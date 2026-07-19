"""
Attention FuncProto viewer (AttentionIdentifier)
────────────────────────────────────────────────

`AttentionIdentifier` gives a structural view over a traced Attention FunctionProto:
  - opset-23 `Attention` node inputs: [Q, K, V]
  - from `Attention`, backtraces each path(Q,K,V) until the qkv_proj MatMul marking ...
    * q/k/v_proj MatMuls
    * head-split Reshape (N,L,E -> N,H,L,E`)
    * KV-cache Concat
    * (OPTIONAL)RMSNorm/RotaryEmbedding.
  - from `Attention` forwardtrace output and mark ...
    * head/seqlen permute Transpose (N,L,H,E` -> N,L,H,E`)
    * head-merge Reshape (N,L,H,E` -> N,L,E)
    * o_proj Matmul

GroupQueryAttention fusion (fuse_group_query_attention)
───────────────────────────────────────────────────────
Pattern (Attention block excluding QKV_proj MatMul and Normalization(if QK norm is enabled)):

    q_in ─────────────────── Transpose ─ RotaryEmbedding ────────────────────┐
    k_in ─────────────────── Transpose ─ RotaryEmbedding ─ Concat(past_k,·) ─┤
    v_in ─ Reshape(3d->4d) ─ Transpose ─────────────────── Concat(past_v,·) ─┤
                                                                           Attention
                                                            Transpose ─ Reshape(4d->3d) ─→ out

Replaced with (custom domain):
  GroupQueryAttention(
      q3d,
      k3d,
      v3d,
      past_key,
      past_value,
      seqlens_k,
      total_sequence_length,
      cos_cache,
      sin_cache
  )
"""

import re
from typing import Dict, Tuple

import numpy as np
import onnx
import onnx_ir as ir
import transformers
from onnx_ir.passes.common import RemoveUnusedNodesPass
from onnxscript.rewriter import pattern

from hf2hw.constant import _NORM_OPS, _SEQLENS_K_NAME, _TOTAL_SEQLEN_NAME
from hf2hw.utils.logger import logger
from hf2hw.utils.onnx_helper import update_opset


def _ir_const_tensor(val: ir.Value) -> ir.TensorProtocol | None:
    """From an edge info (ir.Value) retreive ir.TensorProtocol (TensorProto, Const)"""
    if val.const_value is not None:
        return val.const_value  # return initializer
    p_node: ir.Node = val.producer()
    if p_node is not None and p_node.op_type == "Constant":
        attr: ir.Attr = p_node.attributes.get("value", None)
        if attr is not None:
            return attr.as_tensor()  # return Constant
    return None


def _rope_cache_table(cache: ir.Value) -> ir.TensorProtocol | None:
    """The constant full table behind a RotaryEmbedding cos/sin edge: `Gather(table, position_ids)` input.

    Resolved per attention block — models with several RoPE frequencies (e.g. Gemma3's
    global/local tables) have a different `Gather` feeding each layer type.
    """
    gather: ir.Node = cache.producer()  # NOTE: `Gather -> RotaryEmbedding[1,2]`
    if gather is None or gather.op_type != "Gather":
        return None
    return _ir_const_tensor(gather.inputs[0])


def _layer_index(past_key: ir.Value) -> int | None:
    """Layer index of a matched attention block, read off its `past_keys_{i}` graph input name."""
    match = re.fullmatch(r"past_keys_(\d+)", past_key.name or "")
    return int(match.group(1)) if match else None


def _gqa_rule(
    hf_config: transformers.PretrainedConfig,
    shared: Dict[str, ir.Value],
    head_dim: int,
    norm_after_transpose: bool = False,
    with_mask: bool = False,
) -> pattern.RewriteRule:
    """Attention block -> GroupQueryAttention rewrite rule (over `shared` graph-level values).

    `norm_after_transpose` selects where the (optional) per-head QK norm sits in the traced graph:
      * False (Qwen3/Llama): `Reshape(4d) -> [RMSNorm] -> Transpose -> RotaryEmbedding`
      * True  (Gemma3):      `Reshape(4d) -> Transpose -> RMSNorm -> RotaryEmbedding`
    In the True variant the replacement re-emits the norm on the *pre-transpose* `(N, L, H, E')`
    tensor — RMSNorm on `axis=-1` commutes with the `(0, 2, 1, 3)` transpose — then flattens to 3-D.

    `with_mask` matches the 4-input `Attention(Q, K, V, attn_mask)` written by the shaper for
    sliding-window layers; GQA carries the window via `local_window_size`, so the matched mask
    chain goes dead and is removed by DCE.
    """
    num_heads, num_kv_heads = hf_config.num_attention_heads, hf_config.num_key_value_heads
    # sliding-window layers (e.g. Gemma3/Mistral): GQA expresses the window natively via
    #   `local_window_size` — verified to match the HF semantics exactly (each query attends
    #   the last `sliding_window` keys INCLUDING itself; no off-by-one)
    sliding_window = getattr(hf_config, "sliding_window", None)
    layer_types = getattr(hf_config, "layer_types", None)

    def _window(past_key: ir.Value) -> int | None:
        if not sliding_window:
            return None
        i = _layer_index(past_key)
        if layer_types is not None:
            return sliding_window if layer_types[i] == "sliding_attention" else None
        return sliding_window  # no per-layer split: every layer slides (e.g. Mistral v0.1)

    def _gqa_attrs(past_key: ir.Value, attn_out: ir.Value) -> Dict:
        attrs = {
            "num_heads": num_heads,
            "kv_num_heads": num_kv_heads,
            "do_rotary": 1,
            "rotary_interleaved": 0,
        }
        window = _window(past_key)
        if window is not None:
            attrs["local_window_size"] = window
        # inspect sqrt(d) from the Attention node's attribute
        scale: ir.Attr = attn_out.producer().attributes.get("scale", None)
        if scale is not None:
            attrs["scale"] = scale.as_float()
        return attrs

    def _gqa_node(op, q3d, k3d, v3d, past_key, past_value, cos, sin, attn_out):
        """Collapse the matched pattern into a GroupQueryAttention node (com.microsoft)."""
        return op.GroupQueryAttention(
            q3d,
            k3d,
            v3d,
            past_key,
            past_value,
            shared[_SEQLENS_K_NAME],
            shared[_TOTAL_SEQLEN_NAME],
            cos.producer().inputs[0],  # this block's full cos table (`Gather.input[0]`)
            sin.producer().inputs[0],
            _domain="com.microsoft",
            _outputs=3,
            **_gqa_attrs(past_key, attn_out),
        )

    def _tail(op, q_rope, k_rope, v_in, v_shape, o_shape, past_key, past_value, attn_mask=None):
        """Shared pattern tail: KV concat -> Attention[+mask] -> Transpose -> Reshape(4d->3d).

        `attn_mask` is bound for sliding-window layers (the shaper injects an in-graph window
        mask); the GQA replacement expresses the window via `local_window_size` instead, so
        the matched mask chain simply goes dead and is removed by DCE.
        """
        k_cat = op.Concat(past_key, k_rope, _outputs=["k_cat"])
        # {V} -> Reshape(3d->4d) -> Transpose -> Concat -> {new_V}
        v_4d = op.Reshape(v_in, v_shape)
        v_t = op.Transpose(v_4d, perm=[0, 2, 1, 3])
        v_cat = op.Concat(past_value, v_t, _outputs=["v_cat"])
        # {Q},{K},{V} -> Attention -> Transpose -> Reshape(4d->3d) -> {out3d}
        attn_inputs = (q_rope, k_cat, v_cat) if attn_mask is None else (q_rope, k_cat, v_cat, attn_mask)
        attn = op.Attention(*attn_inputs, _outputs=["attn_out"])
        attn_t = op.Transpose(attn, perm=[0, 2, 1, 3])
        out3d = op.Reshape(attn_t, o_shape)
        return out3d, k_cat, v_cat

    if not norm_after_transpose:

        def _pat_base(op, q_in, k_in, v_in, v_shape, o_shape, cos, sin, past_key, past_value, attn_mask=None):
            # {Q} -> Transpose -> RotaryEmbedding -> {Q}
            q_t = op.Transpose(q_in, perm=[0, 2, 1, 3])
            q_rope = op.RotaryEmbedding(q_t, cos, sin)
            # {K} -> Trnaspose -> RotaryEmbedding -> Concat -> {new_K}
            k_t = op.Transpose(k_in, perm=[0, 2, 1, 3])
            k_rope = op.RotaryEmbedding(k_t, cos, sin)
            return _tail(op, q_rope, k_rope, v_in, v_shape, o_shape, past_key, past_value, attn_mask)

        if with_mask:

            def pat(op, q_in, k_in, v_in, v_shape, o_shape, cos, sin, past_key, past_value, attn_mask):
                return _pat_base(op, q_in, k_in, v_in, v_shape, o_shape, cos, sin, past_key, past_value, attn_mask)

        else:

            def pat(op, q_in, k_in, v_in, v_shape, o_shape, cos, sin, past_key, past_value):
                return _pat_base(op, q_in, k_in, v_in, v_shape, o_shape, cos, sin, past_key, past_value)

        def repl(
            op: pattern.RewriterContext,
            q_in: ir.Value,
            k_in: ir.Value,
            v_in: ir.Value,
            cos: ir.Value,
            sin: ir.Value,
            past_key: ir.Value,
            past_value: ir.Value,
            attn_out: ir.Value,
            **_,
        ):
            def _to_3d(val: ir.Value, flat_shape: ir.Value) -> ir.Value:
                """Make tensor into 3d if not already in 3d shape (insert Reshape(NHLE`->NLE)"""
                prod: ir.Node = val.producer()
                if prod.op_type in _NORM_OPS:
                    # flatten the per-head norm (QK norm) output back to 3d
                    return op.Reshape(val, flat_shape)
                return prod.inputs[0]  # no norm. undo the 4-D head split, the projection output is 3-D

            q3d = _to_3d(q_in, shared["q_3d_shape"])
            k3d = _to_3d(k_in, shared["kv_3d_shape"])
            return _gqa_node(op, q3d, k3d, v_in, past_key, past_value, cos, sin, attn_out)

    else:

        def _pat_base(
            op, q_in, k_in, v_in, v_shape, o_shape, q_scale, k_scale, cos, sin, past_key, past_value, attn_mask=None
        ):
            # {Q} -> Transpose -> RMSNorm -> RotaryEmbedding -> {Q}
            q_t = op.Transpose(q_in, perm=[0, 2, 1, 3])
            q_normed = op.RMSNormalization(q_t, q_scale, _outputs=["q_normed"])
            q_rope = op.RotaryEmbedding(q_normed, cos, sin)
            # {K} -> Transpose -> RMSNorm -> RotaryEmbedding -> Concat -> {new_K}
            k_t = op.Transpose(k_in, perm=[0, 2, 1, 3])
            k_normed = op.RMSNormalization(k_t, k_scale, _outputs=["k_normed"])
            k_rope = op.RotaryEmbedding(k_normed, cos, sin)
            return _tail(op, q_rope, k_rope, v_in, v_shape, o_shape, past_key, past_value, attn_mask)

        if with_mask:

            def pat(
                op, q_in, k_in, v_in, v_shape, o_shape, q_scale, k_scale, cos, sin, past_key, past_value, attn_mask
            ):
                return _pat_base(
                    op, q_in, k_in, v_in, v_shape, o_shape, q_scale, k_scale, cos, sin, past_key, past_value, attn_mask
                )

        else:

            def pat(op, q_in, k_in, v_in, v_shape, o_shape, q_scale, k_scale, cos, sin, past_key, past_value):
                return _pat_base(
                    op, q_in, k_in, v_in, v_shape, o_shape, q_scale, k_scale, cos, sin, past_key, past_value
                )

        def repl(
            op: pattern.RewriterContext,
            q_in: ir.Value,
            k_in: ir.Value,
            v_in: ir.Value,
            q_scale: ir.Value,
            k_scale: ir.Value,
            q_normed: ir.Value,
            k_normed: ir.Value,
            cos: ir.Value,
            sin: ir.Value,
            past_key: ir.Value,
            past_value: ir.Value,
            attn_out: ir.Value,
            **_,
        ):
            def _norm_to_3d(val_4d: ir.Value, scale: ir.Value, normed: ir.Value, flat_shape: ir.Value) -> ir.Value:
                """Re-emit the per-head norm on the pre-transpose (N, L, H, E`) tensor, flatten to 3d."""
                norm_attrs = {name: attr.value for name, attr in normed.producer().attributes.items()}
                out_4d = op.RMSNormalization(val_4d, scale, **norm_attrs)
                return op.Reshape(out_4d, flat_shape)

            q3d = _norm_to_3d(q_in, q_scale, q_normed, shared["q_3d_shape"])
            k3d = _norm_to_3d(k_in, k_scale, k_normed, shared["kv_3d_shape"])
            return _gqa_node(op, q3d, k3d, v_in, past_key, past_value, cos, sin, attn_out)

    def cond(
        context: "MatchContext",
        q_in: ir.Value,
        k_in: ir.Value,
        v_in: ir.Value,
        cos: ir.Value,
        sin: ir.Value,
        k_cat: ir.Value,
        v_cat: ir.Value,
        past_key: ir.Value,
        **_,
    ) -> bool:
        # KV cache appends along the L axis of (N, H, L, E`)
        for cat in (k_cat, v_cat):
            if cat.producer().attributes["axis"].as_int() not in (2, -2):
                return False
        # cos/sin must backtrace to constant full tables of width head_dim//2
        for cache in (cos, sin):
            table = _rope_cache_table(cache)
            if table is None or table.shape[-1] != head_dim // 2:
                return False
        # a sliding-window model needs the layer index (past_keys_{i}) to pick this block's window
        if sliding_window and layer_types is not None and _layer_index(past_key) is None:
            return False
        return True

    return pattern.RewriteRule(pat, repl, cond, name="GroupQueryAttention")


def fuse_group_query_attention(
    model: onnx.ModelProto,
    hf_config: transformers.PretrainedConfig,
) -> Tuple[onnx.ModelProto, int]:
    """
    Rewrite every attention block around `com.microsoft.GroupQueryAttention`.

    `RotaryEmbedding` nodes and cos/sin_cache tables (`Gather`s) are removed.
    main graph loses `position_ids` and gains `seqlens_k`(int32, [batch]) and `total_sequence_length`(int32, [1])
    => GQA applies RoPE internally (`do_rotary=1`) positions implied by `seqlens_k`.

    ```
    seqlens_k[b] = kv_len + q_len - 1
    total_sequence_length = kv_len + q_len
    ```

    Args:
        model: final inlined ONNX model.
        hf_config: HF model config providing `num_attention_heads` / `num_key_value_heads` / `head_dim`.
    Returns:
        `(patched_model, num_fused_attention_blocks)` — the input `model` is returned untouched when nothing matches.
    """
    head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)

    model_ir = ir.from_proto(model)
    graph_ir: ir.Graph = model_ir.graph

    def _shape_initializer(name: str, embed_dim: int) -> ir.Value:
        arr = np.array([1, -1, embed_dim], np.int64)
        value = ir.Value(
            name=name, const_value=ir.tensor(arr), type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape(arr.shape)
        )
        graph_ir.register_initializer(value)
        return value

    shared: Dict[str, ir.Value] = {
        "q_3d_shape": _shape_initializer("gqa_q_3d_shape", hf_config.num_attention_heads * head_dim),
        "kv_3d_shape": _shape_initializer("gqa_kv_3d_shape", hf_config.num_key_value_heads * head_dim),
        _SEQLENS_K_NAME: ir.Value(name=_SEQLENS_K_NAME, type=ir.TensorType(ir.DataType.INT32), shape=ir.Shape([1])),
        _TOTAL_SEQLEN_NAME: ir.Value(
            name=_TOTAL_SEQLEN_NAME, type=ir.TensorType(ir.DataType.INT32), shape=ir.Shape([1])
        ),
    }
    graph_ir.inputs.extend([shared[_SEQLENS_K_NAME], shared[_TOTAL_SEQLEN_NAME]])  # NOTE: add new model inputs

    rules = [
        _gqa_rule(hf_config, shared, head_dim, norm_after_transpose=norm_pos, with_mask=mask)
        for norm_pos in (False, True)
        for mask in (False, True)
    ]
    n_fused = pattern.RewriteRuleSet(rules).apply_to_model(model_ir)
    if not n_fused:
        logger.warning("fuse_group_query_attention: 0 attention pattern matched.")
        return model, 0

    # NOTE: the cos/sin_cache Gathers are dangling (unused) -> delete "position_ids"
    RemoveUnusedNodesPass()(model_ir)
    for graph_input in list(graph_ir.inputs):
        if graph_input.name == "position_ids" and not graph_input.uses():
            graph_ir.inputs.remove(graph_input)

    new_model: onnx.ModelProto = ir.to_proto(model_ir)
    update_opset(new_model, "com.microsoft", 1)
    logger.debug(f"fuse_group_query_attention: fused {n_fused} attention block(s) into GroupQueryAttention")
    return new_model, n_fused


__all__ = ["fuse_group_query_attention"]
