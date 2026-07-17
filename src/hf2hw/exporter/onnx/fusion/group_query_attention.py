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

from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
import onnx_ir as ir
import transformers
from onnx_ir.passes.common import RemoveUnusedNodesPass
from onnxscript.rewriter import pattern

from hf2hw.constant import _NORM_OPS, _SEQLENS_K_NAME, _TOTAL_SEQLEN_NAME
from hf2hw.utils.logger import logger
from hf2hw.utils.onnx_helper import get_bwd_dict, get_fwd_dict, update_opset


# ════════════════════════════ fusion helper ════════════════════════════ #
# TODO: use onnx_ir instead of using bwd_dict, fwd_dict
class AttentionIdentifier:
    """Structural identification of a traced Attention FunctionProto.

    e.g. (Qwen3 generation case)
      {args_0} ─ MatMul ─ Reshape(4d) ─ RMSNorm ─ Transpose ─ RotaryEmbedding ─────┐
      {args_0} ─ MatMul ─ Reshape(4d) ─ RMSNorm ─ Transpose ─ RotaryEmbedding ── Concat ── {prev_K}
      {args_0} ─ MatMul ─ Reshape(4d) ─ Transpose ────────────────────────────── Concat ── {prev_V}
                                                                                   |
                                                                             Attention(Q,K,V)
                                                                 Transpose ─ Reshape(3d) ─ MatMul -> {OUT}
    """

    _PASSTHROUGH_OPS = ("Cast", "Identity", "Add", "Mul")

    def __init__(self, func: onnx.FunctionProto, hf_config: transformers.PreTrainedConfig):
        self.hf_config = hf_config

        self.func = func
        self.fwd_dict: Dict[str, List[onnx.NodeProto]] = get_fwd_dict(func.node)
        self.bwd_dict: Dict[str, onnx.NodeProto] = get_bwd_dict(func.node)
        self.edge2const: Dict[str, onnx.TensorProto] = {
            n.output[0]: attr.t
            for n in func.node
            if n.op_type == "Constant" and n.output
            for attr in n.attribute
            if attr.name == "value"
        }

        _attns = [n for n in func.node if n.op_type == "Attention"]
        if len(_attns) != 1:
            raise ValueError(f"`{func.name}` holds {len(_attns)} Attention node(s); expected exactly 1.")
        self.attention: onnx.NodeProto = _attns[0]

        # NOTE: the following _walk_bwd and _walk_fwd calls are assuming certain graph topologies in advacne
        # [bwd]
        #   Q, K, V MatMul weights are not merged
        #   RotaryEmbedding must have been fused in advance
        #   (OPTIONAL) RMSNormalization, SimplifiedLayerNormalization, LayerNormalization must have been fused in advance
        # [fwd]
        #   {O -> Trnaspose -> Reshape} must be a pathological tree
        self._q = self._walk_bwd(self.attention.input[0])
        self._k = self._walk_bwd(self.attention.input[1])
        self._v = self._walk_bwd(self.attention.input[2])
        self._o = self._walk_fwd(self.attention.output[0])
        for tag, path in (("Q", self._q), ("K", self._k), ("V", self._v)):
            if "matmul" not in path or "reshape" not in path:
                raise ValueError(f"`{func.name}` {tag} path is missing its projection MatMul/Reshape.")
        if "matmul" not in self._o or "reshape" not in self._o:
            raise ValueError(f"`{func.name}` output path is missing its o_proj MatMul/Reshape.")

    @classmethod
    def is_attention_func(cls, func: onnx.FunctionProto) -> bool:
        return sum(1 for n in func.node if n.op_type == "Attention") == 1

    # ================ fwd bwd inspector ================ #
    def _walk_bwd(self, edge: str) -> Dict[str, onnx.NodeProto]:
        """
        Using `slef.bwd_dict` backtrace Attention block nodes on Q, K or V branch.
        Return dictionary containing references to ...
            `matmul` (Q, K or V)
            `concat` (past_KV + cur_KV (IF EXIST))
            `rope`   (assumes RotaryEmbedding has already been fused)
            `norm`   (assums RMSNormalization, SimplifiedLayerNormalization, LayerNormalization are fused already (IF EXIST))
            `transpose`, `reshape` (3d -> 4d head split)
        NodeProtos
        """
        parents: Dict[str, onnx.NodeProto] = {}
        while True:
            node = self.bwd_dict.get(edge)
            if node is None:
                break  # NOTE: graph input reached (end of the loop)

            if node.op_type in ("MatMul", "Gemm"):
                parents["matmul"] = node
                break  # NOTE: MM reached (end of the loop)
            elif node.op_type == "Concat":
                parents["concat"] = node
                edge = next((_ for _ in node.input if _ in self.bwd_dict), None)
                if edge is None:
                    break  # NOTE: concat reached but does not have parent node
            elif node.op_type == "RotaryEmbedding":
                parents["rope"] = node
                edge = node.input[0]
            elif node.op_type in _NORM_OPS:
                parents["norm"] = node
                edge = node.input[0]
            elif node.op_type == "Reshape":
                parents.setdefault("reshape", node)
                edge = node.input[0]
            elif node.op_type == "Transpose":
                parents["transpose"] = node
                edge = node.input[0]
            elif node.op_type in self._PASSTHROUGH_OPS:
                edge = next((_ for _ in node.input if _ in self.bwd_dict), node.input[0])
            else:
                break  # NOTE: no more case to cover exit the loop
        return parents

    def _walk_fwd(self, edge: str) -> Dict[str, onnx.NodeProto]:
        """
        Using `slef.fwd_dict` forwardtrace Attention block nodes from sftmx((QK.T)/sqrt(d))V.output[0]
        Return dictionary containing references to ...
            `matmul` (O)
            `transpose`, `reshape` (4d -> 3d head merge)
        NodeProtos
        """
        children: Dict[str, onnx.NodeProto] = {}
        while True:
            nexts = self.fwd_dict.get(edge, [])
            if len(nexts) != 1:
                break  # NOTE: graph output reached (end of the loop)
            assert (
                len(nexts) == 1
            ), f"`{edge}` directs to multiple NodeProtos `{[n.name for n in nexts]}`. Please pass an edge name that does not branch out (pathological)."
            node = nexts[0]
            if node.op_type in ("MatMul", "Gemm"):
                children["matmul"] = node
                break
            elif node.op_type == "Reshape":
                children["reshape"] = node
            elif node.op_type == "Transpose":
                children["transpose"] = node
            elif node.op_type not in _PASSTHROUGH_OPS:
                break
            edge = node.output[0]  # NOTE: always follow output[0]
        return children

    # ================ shape references ================ #
    @property
    def query4dShape(self) -> List[int]:
        return [1, -1, self.num_heads, self.head_dim]

    @property
    def key4dShape(self) -> List[int]:
        return [1, -1, self.kv_num_heads, self.head_dim]

    @property
    def value4dShape(self) -> List[int]:
        return [1, -1, self.kv_num_heads, self.head_dim]

    @property
    def out3dShape(self) -> List[int]:
        return [1, -1, self.num_heads * self.head_dim]

    # ================ node references ================ #
    @property
    def queryMM(self) -> onnx.NodeProto:
        return self._q["matmul"]

    @property
    def keyMM(self) -> onnx.NodeProto:
        return self._k["matmul"]

    @property
    def valueMM(self) -> onnx.NodeProto:
        return self._v["matmul"]

    @property
    def query4dReshape(self) -> onnx.NodeProto:
        return self._q["reshape"]

    @property
    def key4dReshape(self) -> onnx.NodeProto:
        return self._k["reshape"]

    @property
    def value4dReshape(self) -> onnx.NodeProto:
        return self._v["reshape"]

    @property
    def queryNorm(self) -> Optional[onnx.NodeProto]:
        return self._q.get("norm")

    @property
    def keyNorm(self) -> Optional[onnx.NodeProto]:
        return self._k.get("norm")

    @property
    def valueNorm(self) -> Optional[onnx.NodeProto]:
        return self._v.get("norm")

    @property
    def queryRope(self) -> Optional[onnx.NodeProto]:
        return self._q.get("rope")

    @property
    def keyRope(self) -> Optional[onnx.NodeProto]:
        return self._k.get("rope")

    @property
    def keyConcat(self) -> Optional[onnx.NodeProto]:
        return self._k.get("concat")

    @property
    def valueConcat(self) -> Optional[onnx.NodeProto]:
        return self._v.get("concat")

    @property
    def out3dReshape(self) -> onnx.NodeProto:
        return self._o["reshape"]

    @property
    def outMM(self) -> onnx.NodeProto:
        return self._o["matmul"]

    # ================ derived attention hyper-params ================ #
    def _get_shape_tp_from_reshape(self, reshape_node: onnx.NodeProto) -> onnx.TensorProto | None:
        assert (
            reshape_node.op_type == "Reshape"
        ), f"`{reshape_node.op_type}` is not an allowed op_type for `_get_shape_tp_from_reshape`"
        return self.edge2const.get(reshape_node.input[1], None)

    @property
    def head_dim(self) -> int:
        """Last element of the Q head-split Reshape, e.g. [1, L, H, head_dim] -> head_dim"""
        shape_tp = self._get_shape_tp_from_reshape(self.query4dReshape)
        head_dim = int(onnx.numpy_helper.to_array(shape_tp).ravel()[-1])
        if self.hf_config is not None:
            config_head_dim = getattr(
                self.hf_config, "head_dim", self.hf_config.hidden_size // self.hf_config.num_attention_heads
            )
            head_dim = head_dim if head_dim <= 0 else config_head_dim
            assert head_dim == config_head_dim
        return head_dim

    @property
    def num_heads(self) -> int:
        num_heads = self._mm_out_features(self.queryMM) // self.head_dim
        if self.hf_config is not None:
            assert num_heads == self.hf_config.num_attention_heads
        return num_heads

    @property
    def kv_num_heads(self) -> int:
        num_heads = self._mm_out_features(self.keyMM) // self.head_dim
        if self.hf_config is not None:
            assert num_heads == self.hf_config.num_key_value_heads
        return num_heads

    def _mm_out_features(self, mm: onnx.NodeProto) -> int:
        """Out-features of a projection MatMul/Gemm: weight is `[in, out]` (`[out, in]` if transB)."""
        trans_b = next((a.i for a in mm.attribute if a.name == "transB"), 0) if mm.op_type == "Gemm" else 0
        for edge_name in mm.input:
            tp = self.edge2const.get(edge_name, None)
            if tp is not None and len(tp.dims) == 2:
                return int(tp.dims[0] if trans_b else tp.dims[1])
        raise ValueError(f"`{self.func.name}` {mm.name!r} has non 2D Constant weight input.")


# ════════════════════════════ GQA fusion ════════════════════════════ #


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


def _find_rope_caches(graph: ir.Graph) -> Tuple[ir.Value, ir.Value] | None:
    """The full cos/sin tables shared by every `RotaryEmbedding`'s `Gather(table, position_ids)` input."""
    cos_table_candidates, sin_table_candidates = set(), set()
    for node in graph:
        if node.op_type != "RotaryEmbedding":
            continue
        for cache_idx, table_candidates in ((1, cos_table_candidates), (2, sin_table_candidates)):
            gather: ir.Node = node.inputs[cache_idx].producer()  # NOTE: `Gather -> RotaryEmbedding[1,2]`
            if gather is None or gather.op_type != "Gather" or _ir_const_tensor(gather.inputs[0]) is None:
                return None
            table_candidates.add(gather.inputs[0])
    if len(cos_table_candidates) != 1 or len(sin_table_candidates) != 1:
        return None
    return cos_table_candidates.pop(), sin_table_candidates.pop()


def _gqa_rule(hf_config: transformers.PretrainedConfig, shared: Dict[str, ir.Value]) -> pattern.RewriteRule:
    """Attention block -> GroupQueryAttention rewrite rule (over `shared` graph-level values)."""
    num_heads, num_kv_heads = hf_config.num_attention_heads, hf_config.num_key_value_heads

    def pat(
        op: pattern.OpsetPatternBuilder,
        q_in: pattern.Var,
        k_in: pattern.Var,
        v_in: pattern.Var,
        v_shape: pattern.Var,
        o_shape: pattern.Var,
        cos: pattern.Var,
        sin: pattern.Var,
        past_key: pattern.Var,
        past_value: pattern.Var,
    ):
        # {Q} -> Transpose -> RotaryEmbedding -> {Q}
        q_t = op.Transpose(q_in, perm=[0, 2, 1, 3])
        q_rope = op.RotaryEmbedding(q_t, cos, sin)
        # {K} -> Trnaspose -> RotaryEmbedding -> Concat -> {new_K}
        k_t = op.Transpose(k_in, perm=[0, 2, 1, 3])
        k_rope = op.RotaryEmbedding(k_t, cos, sin)
        k_cat = op.Concat(past_key, k_rope, _outputs=["k_cat"])
        # {V} -> Reshape(3d->4d) -> Transpose -> Concat -> {new_V}
        v_4d = op.Reshape(v_in, v_shape)
        v_t = op.Transpose(v_4d, perm=[0, 2, 1, 3])
        v_cat = op.Concat(past_value, v_t, _outputs=["v_cat"])
        # {Q},{K},{V} -> Attention -> Transpose -> Reshape(4d->3d) -> {out3d}
        attn = op.Attention(q_rope, k_cat, v_cat, _outputs=["attn_out"])
        attn_t = op.Transpose(attn, perm=[0, 2, 1, 3])
        out3d = op.Reshape(attn_t, o_shape)
        return out3d, k_cat, v_cat

    def repl(
        op: pattern.RewriterContext,
        q_in: ir.Value,
        k_in: ir.Value,
        v_in: ir.Value,
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

        attrs = {
            "num_heads": num_heads,
            "kv_num_heads": num_kv_heads,
            "do_rotary": 1,
            "rotary_interleaved": 0,
        }

        # inspect sqrt(d) from the Attention node's attribute
        scale: ir.Attr = attn_out.producer().attributes.get("scale", None)
        if scale is not None:
            attrs["scale"] = scale.as_float()

        # collapse pattern into a GroupQueryAttention node (com.microsoft)
        return op.GroupQueryAttention(
            _to_3d(q_in, shared["q_3d_shape"]),
            _to_3d(k_in, shared["kv_3d_shape"]),
            v_in,
            past_key,
            past_value,
            shared[_SEQLENS_K_NAME],
            shared[_TOTAL_SEQLEN_NAME],
            shared["cos_cache"],
            shared["sin_cache"],
            _domain="com.microsoft",
            _outputs=3,
            **attrs,
        )

    def cond(
        context: "MatchContext",
        q_in: ir.Value,
        k_in: ir.Value,
        v_in: ir.Value,
        cos: ir.Value,
        sin: ir.Value,
        k_cat: ir.Value,
        v_cat: ir.Value,
        **_,
    ) -> bool:
        # KV cache appends along the L axis of (N, H, L, E`)
        for cat in (k_cat, v_cat):
            if cat.producer().attributes["axis"].as_int() not in (2, -2):
                return False
        # ir.Value (cos_cache, sin_cache) must satisfy `Gather.input[0] == shared["cos/sin_cache"]`
        for cache, table in ((cos, shared["cos_cache"]), (sin, shared["sin_cache"])):
            gather: ir.Node = cache.producer()
            if gather is None or gather.op_type != "Gather" or gather.inputs[0] is not table:
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

    caches: Tuple[ir.Value, ir.Value] | None = _find_rope_caches(graph_ir)
    if caches is None:
        logger.warning("fuse_group_query_attention: cannot locate RotaryEmbedding cache tables (cos, sin).")
        return model, 0
    cos_cache, sin_cache = caches
    _table_width = _ir_const_tensor(cos_cache).shape[-1]
    if _table_width != head_dim // 2:
        logger.warning(
            f"fuse_group_query_attention: cos table width `{_table_width} != head_dim//2` "
            f"(hf_config.head_dim={head_dim // 2})."
        )
        return model, 0

    def _shape_initializer(name: str, embed_dim: int) -> ir.Value:
        arr = np.array([1, -1, embed_dim], np.int64)
        value = ir.Value(
            name=name, const_value=ir.tensor(arr), type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape(arr.shape)
        )
        graph_ir.register_initializer(value)
        return value

    shared: Dict[str, ir.Value] = {
        "cos_cache": cos_cache,  # the existing full-table constants are consumed directly
        "sin_cache": sin_cache,
        "q_3d_shape": _shape_initializer("gqa_q_3d_shape", hf_config.num_attention_heads * head_dim),
        "kv_3d_shape": _shape_initializer("gqa_kv_3d_shape", hf_config.num_key_value_heads * head_dim),
        _SEQLENS_K_NAME: ir.Value(name=_SEQLENS_K_NAME, type=ir.TensorType(ir.DataType.INT32), shape=ir.Shape([1])),
        _TOTAL_SEQLEN_NAME: ir.Value(
            name=_TOTAL_SEQLEN_NAME, type=ir.TensorType(ir.DataType.INT32), shape=ir.Shape([1])
        ),
    }
    graph_ir.inputs.extend([shared[_SEQLENS_K_NAME], shared[_TOTAL_SEQLEN_NAME]])  # NOTE: add new model inputs

    n_fused = pattern.RewriteRuleSet([_gqa_rule(hf_config, shared)]).apply_to_model(model_ir)
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


__all__ = ["AttentionIdentifier", "fuse_group_query_attention"]
