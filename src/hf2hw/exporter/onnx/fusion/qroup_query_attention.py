"""GroupQueryAttention fusion (fuse_group_query_attention) + AttentionIdentifier.

`AttentionIdentifier` gives a structural view over a traced Attention FunctionProto:
starting from the (single) opset-23 `Attention` node — whose inputs are ordered Q, K, V —
it backtraces each path to the projection MatMul and the 4-D head-split Reshape, picking
up the optional RMSNorm / RotaryEmbedding / KV-cache Concat nodes along the way, and
walks forward from the Attention output to the Transpose → Reshape(3-D) → MatMul(o_proj)
chain. `head_dim` / `num_heads` / `kv_num_heads` are derived from the graph itself
(reshape target shape + projection weight widths), so no HF config is needed.

`fuse_group_query_attention` rewrites a merged two-case model (e.g.
`2026-07-14_01-23-48__ORT__Qwen-Qwen3-0.6B/case2.onnx`) to use the ORT
`com.microsoft.GroupQueryAttention` contrib op:

  per attention FunctionProto (paths located via AttentionIdentifier)
      keep:    q/k/v projection MatMuls (+ per-head RMSNorm with its 4-D reshape, when
               present — the norm output is flattened back to 3-D for GQA), o_proj MatMul
      drop:    Transposes, RotaryEmbedding nodes, KV Concats, the Attention node and the
               output-side Transpose/Reshape — GQA does head split/merge, RoPE, cache
               append and grouped SDPA internally
      GQA:     GroupQueryAttention(q3d, k3d, v3d, past_key, past_value,
                                   seqlens_k, total_sequence_length, cos_cache, sin_cache)
               attrs: num_heads, kv_num_heads, do_rotary=1, rotary_interleaved=0, scale

  main graph
      * the RoPE FunctionProto call is removed (GQA derives positions from `seqlens_k`);
        its full cos/sin tables — `Gather.input[0]` inside the RoPE function body — are
        lifted into graph initializers and passed to every GQA call
      * `position_ids` graph input is dropped; `seqlens_k` (int32, [batch]) and
        `total_sequence_length` (int32, scalar) graph inputs are added
        (`seqlens_k[b] = past_len + new_len - 1`, `total_sequence_length = past_len + new_len`)

NOTE: the surgery spans function bodies, function signatures, call sites and graph IO,
which is beyond a local pattern rewrite — so the identification is done with
`AttentionIdentifier` directly instead of the onnxscript/onnx_ir pattern matcher used by
the single-node fusions (fuse_rope / fuse_rms_norm).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
import transformers
from onnx import numpy_helper

from hf2hw.utils.logger import logger
from hf2hw.utils.onnx_helper import drop_vi_by_name, get_bwd_dict, get_fwd_dict, update_opset

_MSFT_DOMAIN = "com.microsoft"
_GQA_OP = "GroupQueryAttention"
_NORM_OPS = ("RMSNormalization", "SimplifiedLayerNormalization", "LayerNormalization")
_PASSTHROUGH_OPS = ("Cast", "Identity", "Add", "Mul")  # bias/scale/dtype hops on a projection path

# names used for the lifted cos/sin tables and the new seq-length inputs (graph + function scope)
_COS_CACHE, _SIN_CACHE = "cos_cache", "sin_cache"
_SEQLENS_K, _TOTAL_SEQLEN = "seqlens_k", "total_sequence_length"


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

    # ── path walkers ──────────────────────────────────────────────────────────
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
            elif node.op_type in _PASSTHROUGH_OPS:
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

    # ── shape references ──────────────────────────────────────────────────────
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

    # ── node references ──────────────────────────────────────────────────────
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
    def oMM(self) -> onnx.NodeProto:
        return self._o["matmul"]

    # ── derived attention hyper-params (graph-derived; no HF config needed) ───
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
        head_dim = head_dim if head_dim <= 0 else self.hf_config.head_dim
        assert head_dim == self.hf_config.head_dim
        return head_dim

    @property
    def num_heads(self) -> int:
        num_heads = self._mm_out_features(self.queryMM) // self.head_dim
        assert num_heads == self.hf_config.num_attention_heads
        return num_heads

    @property
    def kv_num_heads(self) -> int:
        num_heads = self._mm_out_features(self.keyMM) // self.head_dim
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


# ════════════════════════════ GQA fusion ════════════════════════════


def _is_rope_func(func: onnx.FunctionProto) -> bool:
    """A RoPE lookup function: every non-Constant node is a Gather driven by a function input."""
    gathers = [n for n in func.node if n.op_type != "Constant"]
    return (
        len(func.output) == 2
        and len(gathers) >= 2
        and all(n.op_type == "Gather" and len(n.input) >= 2 and n.input[1] in func.input for n in gathers)
    )


def _rope_tables(func: onnx.FunctionProto) -> Tuple[onnx.TensorProto, onnx.TensorProto]:
    """cos/sin full tables = `Gather.input[0]` Constants, matched to the (cos, sin) output order."""
    consts = {
        n.output[0]: attr.t
        for n in func.node
        if n.op_type == "Constant"
        for attr in n.attribute
        if attr.name == "value"
    }
    out2gather = {n.output[0]: n for n in func.node if n.op_type == "Gather"}
    tables = []
    for out in func.output:  # function outputs are (cos, sin) in order
        gather = out2gather.get(out)
        assert gather is not None, f"RoPE function output {out!r} is not produced by a Gather."
        tables.append(consts[gather.input[0]])
    return tables[0], tables[1]


def _cast_table(tp: onnx.TensorProto, elem_type: int, name: str) -> onnx.TensorProto:
    """Re-type a cos/sin table to the activation dtype expected by GQA (T-constraint)."""
    arr = numpy_helper.to_array(tp)
    if elem_type == onnx.TensorProto.FLOAT:
        return numpy_helper.from_array(arr.astype(np.float32), name)
    if elem_type == onnx.TensorProto.FLOAT16:
        return numpy_helper.from_array(arr.astype(np.float16), name)
    if elem_type == onnx.TensorProto.BFLOAT16:
        import ml_dtypes

        raw = arr.astype(ml_dtypes.bfloat16).tobytes()
        return onnx.helper.make_tensor(name, onnx.TensorProto.BFLOAT16, list(arr.shape), raw, raw=True)
    raise ValueError(f"unsupported activation elem_type={elem_type} for GQA cos/sin caches.")


def _rewrite_attention_function(func: onnx.FunctionProto, ident: AttentionIdentifier) -> Tuple[int, int]:
    """Rebuild `func` around a single GroupQueryAttention node (in place).

    Returns the (cos, sin) positions in the OLD input list so the caller can rewrite
    call sites positionally.
    """
    head_dim, n_heads, n_kv_heads = ident.head_dim, ident.num_heads, ident.kv_num_heads

    cos_in, sin_in = ident.queryRope.input[1], ident.queryRope.input[2]
    assert (
        cos_in in func.input and sin_in in func.input
    ), f"`{func.name}` RotaryEmbedding cos/sin are not function inputs; cannot re-route to GQA."
    past_key, past_value = ident.keyConcat.input[0], ident.valueConcat.input[0]
    new_key, new_value = ident.keyConcat.output[0], ident.valueConcat.output[0]

    compute: List[onnx.NodeProto] = []

    def _3d_path(tag: str, mm, reshape4d, norm, width: int) -> str:
        """GQA takes pre-head-split (B, S, H*E); keep the 4-D reshape + norm only when needed."""
        compute.append(mm)
        if norm is None:
            return mm.output[0]
        compute.extend([reshape4d, norm])  # per-head norm needs the (B, S, H, E) layout
        shape_name, out_name = f"{tag}_gqa_flat_shape", f"{tag}_gqa_3d"
        shape_tp = numpy_helper.from_array(np.array([1, -1, width], np.int64), shape_name)
        compute.append(onnx.helper.make_node("Constant", [], [shape_name], value=shape_tp, name=f"const_{shape_name}"))
        compute.append(
            onnx.helper.make_node("Reshape", [norm.output[0], shape_name], [out_name], name=f"node_{out_name}")
        )
        return out_name

    q3d = _3d_path("q", ident.queryMM, ident.query4dReshape, ident.queryNorm, n_heads * head_dim)
    k3d = _3d_path("k", ident.keyMM, ident.key4dReshape, ident.keyNorm, n_kv_heads * head_dim)
    v3d = _3d_path("v", ident.valueMM, ident.value4dReshape, ident.valueNorm, n_kv_heads * head_dim)

    gqa_attrs = dict(num_heads=n_heads, kv_num_heads=n_kv_heads, do_rotary=1, rotary_interleaved=0)
    scale = next((a.f for a in ident.attention.attribute if a.name == "scale"), 0.0)
    if scale:
        gqa_attrs["scale"] = scale
    compute.append(
        onnx.helper.make_node(
            _GQA_OP,
            [q3d, k3d, v3d, past_key, past_value, _SEQLENS_K, _TOTAL_SEQLEN, _COS_CACHE, _SIN_CACHE],
            ["gqa_out", new_key, new_value],  # present KV keep the original output edge names
            name="node_GroupQueryAttention",
            domain=_MSFT_DOMAIN,
            **gqa_attrs,
        )
    )
    o_mm = ident.oMM
    o_mm.input[0] = "gqa_out"  # GQA output is already (B, S, H*E): o_proj consumes it directly
    compute.append(o_mm)

    # keep only the Constants the new body still needs (weights, norm scales, reshape shapes)
    needed = {i for n in compute for i in n.input}
    old_consts = [n for n in func.node if n.op_type == "Constant" and n.output and n.output[0] in needed]

    del func.node[:]
    func.node.extend(old_consts + compute)

    # inputs: cos → seqlens_k, sin → total_sequence_length (position preserved), caches appended
    cos_pos, sin_pos = list(func.input).index(cos_in), list(func.input).index(sin_in)
    func.input[cos_pos], func.input[sin_pos] = _SEQLENS_K, _TOTAL_SEQLEN
    func.input.extend([_COS_CACHE, _SIN_CACHE])

    if not any(oi.domain == _MSFT_DOMAIN for oi in func.opset_import):
        func.opset_import.append(onnx.helper.make_opsetid(_MSFT_DOMAIN, 1))
    return cos_pos, sin_pos


def fuse_group_query_attention(model: onnx.ModelProto) -> Tuple[onnx.ModelProto, int]:
    """
    Rewrite every KV-cache attention FunctionProto around `com.microsoft.GroupQueryAttention`.

    The RoPE FunctionProto and its main-graph call are removed — GQA applies RoPE internally
    (`do_rotary=1`) from the lifted full cos/sin tables and the positions implied by
    `seqlens_k`. The main graph loses `position_ids` and gains `seqlens_k` (int32, [batch])
    and `total_sequence_length` (int32 scalar):
        `seqlens_k[b] = past_len + new_len - 1`, `total_sequence_length = past_len + new_len`

    Args:
        model: merged model (modified in place, then returned).
    Returns:
        `(patched_model, num_fused_attention_functions)`
    """
    rope_funcs = [f for f in model.functions if _is_rope_func(f)]
    attn_funcs = [f for f in model.functions if AttentionIdentifier.is_attention_func(f)]
    if not attn_funcs:
        logger.warning("fuse_group_query_attention: no Attention FunctionProto found; nothing to do.")
        return model, 0
    assert len(rope_funcs) == 1, f"expected exactly 1 RoPE lookup function, found {len(rope_funcs)}."
    rope_func = rope_funcs[0]

    # ── lift the full cos/sin tables into graph initializers (activation dtype) ──
    ident0 = AttentionIdentifier(attn_funcs[0])
    act_type = next(
        tp.data_type for i in ident0.queryMM.input if (tp := ident0.edge2const.get(i)) is not None and len(tp.dims) == 2
    )
    cos_tp, sin_tp = _rope_tables(rope_func)
    model.graph.initializer.extend(
        [_cast_table(cos_tp, act_type, _COS_CACHE), _cast_table(sin_tp, act_type, _SIN_CACHE)]
    )

    # ── rewrite the attention functions ──
    n_fused = 0
    cos_pos = sin_pos = None
    rewritten_fn_names = set()
    for func in attn_funcs:
        ident = AttentionIdentifier(func)
        if ident.keyConcat is None or ident.valueConcat is None or ident.queryRope is None:
            logger.warning(f"fuse_group_query_attention: `{func.name}` has no KV cache/RoPE; skipped.")
            continue
        cos_pos, sin_pos = _rewrite_attention_function(func, ident)
        rewritten_fn_names.add(func.name)
        n_fused += 1
    if not n_fused:
        return model, 0

    # ── main graph: rewire call sites, drop the RoPE call/function and position_ids ──
    rope_call = next(n for n in model.graph.node if n.op_type == rope_func.name)
    cos_edge, sin_edge = rope_call.output[0], rope_call.output[1]

    for node in model.graph.node:
        if node.op_type in rewritten_fn_names:
            node.input[cos_pos], node.input[sin_pos] = _SEQLENS_K, _TOTAL_SEQLEN
            node.input.extend([_COS_CACHE, _SIN_CACHE])

    model.graph.node.remove(rope_call)
    model.functions.remove(rope_func)
    drop_vi_by_name(model.graph.input, {"position_ids"})  # only the RoPE call consumed it
    drop_vi_by_name(model.graph.value_info, {cos_edge, sin_edge})

    batch = model.graph.input[0].type.tensor_type.shape.dim[0].dim_value or 1
    model.graph.input.extend(
        [
            onnx.helper.make_tensor_value_info(_SEQLENS_K, onnx.TensorProto.INT32, [batch]),
            onnx.helper.make_tensor_value_info(_TOTAL_SEQLEN, onnx.TensorProto.INT32, []),
        ]
    )
    update_opset(model, _MSFT_DOMAIN, 1)
    logger.debug(f"fuse_group_query_attention: fused {n_fused} attention function(s) into {_GQA_OP}")
    return model, n_fused


__all__ = ["AttentionIdentifier", "fuse_group_query_attention"]
