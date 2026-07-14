"""Dynamic-shape post-processing for the exported two-case CausalLM ONNX graphs.

After ``torch.onnx.export`` every dimension is frozen from the traced example.
``CausalLMONNXShaper`` rewrites the input/output shapes to symbolic dims and
re-runs ``onnx.shape_inference`` so the sequence length propagates through the
main graph.

Prefill vs generation share the same Reshape nodes (same node/tensor names) inside
their attention ``FunctionProto`` bodies — only the enclosing function name differs
by the ``____case{N}`` suffix. That lets us treat the two graphs as one: we trace with
seq = ``constant.TRACE_L`` (a unique sentinel) so every Reshape shape constant holding
``TRACE_L`` marks the frozen sequence axis. ``make_dynamic_onnx(case1, "prefill")``
**registers** those axes, keyed by (normalized function name, shape-source tensor);
``make_dynamic_onnx(case2, "generation")`` **replays** the registry onto the same-named
Reshape shapes (its own seq dim is a frozen ``1`` with no sentinel to scan for).

How the registered seq axis is rewritten depends on ``allow_zero``:

  allow_zero=0 (default)
    Set the seq axis to ``0`` ("copy the input dim at that axis") and reset the node's
    ``allowzero=0`` so the copy is honored. The head-count axis stays ``-1`` (inferred),
    so a single shape constant can stay shared across Q/K/V (different head counts).

  allow_zero=1
    Some runtimes reject the ``0``-copy. Instead make the seq axis the inferred ``-1``
    and bake the previously-inferred axis explicitly from ``hf_config``. Because Q/K/V
    share one constant but need different head counts, each Reshape gets its **own**
    private shape constant (un-shared):
        [1, 0, -1, 128] (Q)   -> [1, -1, num_attention_heads,  head_dim]
        [1, 0, -1, 128] (K/V) -> [1, -1, num_key_value_heads,  head_dim]
        [1, 0, -1]      (out) -> [1, -1, num_attention_heads * head_dim]

Symbolic dims set explicitly (shape inference cannot cross FunctionProto boundaries):

  prefill (case1)
    inputs:   input_ids[1, L], position_ids[1, L]
    dropped:  attention_mask  (no consumers)

  generation (case2)
    inputs:   past_keys_*[1, H, L_prev, E], past_values_*[1, H, L_prev, E]
    dropped:  attention_mask
    outputs:  past_keys_*_out[1, H, L_prev+1, E], past_values_*_out[1, H, L_prev+1, E]
              ("L_prev+1" is a literal dim_param label — ONNX dim_param is a free
               string, not arithmetic — documenting the past+current relation)
"""

import logging
import shutil
from abc import ABC, abstractmethod
from os import PathLike
from typing import Dict, Iterator, List, Literal, Sequence, Set, Tuple

import numpy as np
import onnx
import transformers

from hf2hw.constant import _CASE_SUFFIX_RE, ONNX_DOMAIN_NAME, SUBGRAPH_MAP_TYPE, TRACE_L
from hf2hw.exporter.onnx.merge import merge_subgraphs_into_model
from hf2hw.utils.hf_helper import get_head_dim
from hf2hw.utils.logger import logger
from hf2hw.utils.onnx_helper import (
    drop_vi_by_name,
    filter_shape_metadata,
    get_vi_axis,
    set_vi_axis,
    update_node_attribute,
)


def _normalize_funcproto_name(name: str) -> str:
    return _CASE_SUFFIX_RE.sub("", name)


def _projection_out_features(reshape_node: onnx.NodeProto, bwd_dict: dict, edge2shape: dict) -> int | None:
    """
    Out-features of the MatMul/Gemm feeding a Reshape's data input (skips a trailing Add/Mul if exist).
    e.g. The weight is `[in, out]` (or `[out, in]` when a Gemm sets `transB=1`).
      Return `out` if reshape_node does not have MatMul/Gemm parent return None
    """
    p_node = bwd_dict.get(reshape_node.input[0])
    if (p_node is not None) and (p_node.op_type in ("Add", "Mul")):
        # NOTE: skip Add, Mul node if parent's parent is MatMul or Gemm
        p_node = next(
            (bwd_dict[i] for i in p_node.input if bwd_dict.get(i) and bwd_dict[i].op_type in ("MatMul", "Gemm")), p_node
        )
    if p_node is None or p_node.op_type not in ("MatMul", "Gemm"):
        return None
    trans_b = next((a.i for a in p_node.attribute if a.name == "transB"), 0) if p_node.op_type == "Gemm" else 0
    for inp in p_node.input:
        shape = edge2shape.get(inp, None)
        if shape and len(shape) == 2:
            return shape[0] if trans_b else shape[1]
    return None


class CausalLMONNXShaper(ABC):
    def __init__(self, hf_config: transformers.PreTrainedConfig) -> None:
        self.hf_config = hf_config
        self._reshape_seqlen_axis: Dict[Tuple[str, str], Set[int]] = {}

    # ================ ABSTRACT METHODS  ================ #
    @abstractmethod
    def _post_process_final_onnx(self, case_idx: int, onnx_path: str | PathLike):
        """Postprocess method that optimizes the final merged ONNX graph"""
        pass

    # ================ subgraph merge  ================ #
    def merge_subgraphs_into_main_graph(
        self,
        case_paths: List[str],
        subgraph_map: SUBGRAPH_MAP_TYPE,
    ) -> None:
        """
        Merge the ONNX subgraphs into the main ONNX graphs

        Args:
            case_paths: main ONNX graph paths representing unique input cases
            subgraph_map: `subgraph_map[case_idx]` maps torchlib_op_name -> subgraph onnx path
        """
        for case_idx, case_path in enumerate(case_paths):
            mapping = subgraph_map.get(case_idx, {})
            if mapping:
                logger.info(f"  merging {len(mapping)} Subgraph(s) into {case_path}")
                merge_subgraphs_into_model(
                    case_path=case_path,
                    torchlib_op2subgraph_path=mapping,
                    domain=ONNX_DOMAIN_NAME,
                )
            else:
                logger.warning(f"No plugin subgraph to merge for case {case_idx + 1}.")

            self._post_process_final_onnx(case_idx, case_path)

        # clean up subgraph onnx
        if (logger.isEnabledFor(logging.DEBUG) is False) and (self._subgraph_dir.exists()):
            shutil.rmtree(self._subgraph_dir)

    # ================ dynamic shaping  ================ #
    def make_dynamic_onnx(
        self,
        onnx_path: str | PathLike,
        mode: Literal["prefill", "generation"],
        *,
        allowzero: int = 0,
        seq_sym: str = "L",
        prev_sym: str = "L_prev",
        curr_sym: str = "L_prev+1",
    ) -> None:
        """
        Rewrite the CausalLM ONNX graph so that it can take dynamic seq_len.

        `mode="prefill"`      locates every `TRACE_L` seq axis in the Reshape shapes and
                              registers it so the generation graph can mirror it.
        `mode="generation"`   replays the registered seq axes onto the same-named Reshape
                              shapes (its own seq dim is a frozen `1` with no sentinel).

        `allowzero=0` writes the seq axis as a `0`-copy (default).
        `allowzero=1` makes the seq axis the inferred `-1` and bakes the head-count / hidden axis explicitly from `self.hf_config`.

        The model is loaded from / saved back to ``onnx_path`` in place.
        """
        if allowzero not in (0, 1):
            raise ValueError(f"`{allowzero=}` is invalid; expected 0 or 1.")

        model: onnx.ModelProto = onnx.load(onnx_path, load_external_data=True)
        drop_vi_by_name(model.input, {"attention_mask"})  # TODO: attention_bias? ALiBi?

        # NOTE: IO ValueInfoProto update (make QueryL dynamic)
        for vi in model.graph.input:
            if vi.name in ("input_ids", "position_ids"):
                set_vi_axis(vi, 1, seq_sym)
        for vi in model.graph.output:
            if vi.name in ("logits",):
                set_vi_axis(vi, 1, seq_sym)
        # TODO: CausalLMExporter._post_process_final_onnx
        # will make prefill graph to output KV cache in future. make KV output dynamic too.

        # NOTE: IO KV ValueInfoProto update (make KeyL, ValueL dynamic)
        if mode == "prefill":
            self._reshape_seqlen_axis = self._scan_and_register_seq_axes(model)
        else:
            for vi in model.graph.input:
                if vi.name.startswith(("past_keys_", "past_values_")):
                    set_vi_axis(vi, 2, prev_sym)
            for vi in model.graph.output:
                if vi.name.startswith(("past_keys_", "past_values_")) and vi.name.endswith("_out"):
                    set_vi_axis(vi, 2, curr_sym)

        if allowzero == 0:
            # NOTE: loop over FuncProtos and set Reshape seqlen axis to 0
            for func in model.functions:
                fn_key = _normalize_funcproto_name(func.name)
                for name, tp in filter_shape_metadata(func.node, ()):
                    idxs_tobe_zero = self._reshape_seqlen_axis.get((fn_key, name), None)
                    assert idxs_tobe_zero, f"Key `{(fn_key, name)}` is not registered under self._reshape_seqlen_axis."
                    if idxs_tobe_zero:
                        shape_npy = onnx.numpy_helper.to_array(tp).copy().ravel()
                        for i in idxs_tobe_zero:
                            shape_npy[i] = 0
                        tp.CopyFrom(onnx.numpy_helper.from_array(shape_npy, tp.name))
        elif allowzero == 1:
            # NOTE: loop over FuncProtos and set Reshape seqlen axis to -1, replace original -1 to fixed size (e.g. head_dim, num_qkv_heads).
            head_dim = get_head_dim(self.hf_config)
            q_heads = self.hf_config.num_attention_heads
            kv_heads = getattr(self.hf_config, "num_key_value_heads", q_heads)

            for func in model.functions:
                fn_key = _normalize_funcproto_name(func.name)
                edge2Lidx_map: Dict[str, Set[int]] = {
                    src: seqlen_idx for (fk, src), seqlen_idx in self._reshape_seqlen_axis.items() if fk == fn_key
                }
                if not edge2Lidx_map:
                    continue

                bwd_dict: Dict[str, onnx.NodeProto] = {out: node for node in func.node for out in node.output}
                edge2value = {
                    name: onnx.numpy_helper.to_array(tp).tolist() for name, tp in filter_shape_metadata(func.node, ())
                }  # NOTE: edge here represents both `Constant` and `TensorProto`
                edge2shape = {
                    node.output[0]: list(attr.t.dims)
                    for node in func.node
                    if node.op_type == "Constant" and node.output
                    for attr in node.attribute
                    if attr.name == "value"
                }  # NOTE: edge here represents `Constant` output

                new_consts: List[onnx.NodeProto] = []
                for reshape_n in (n for n in func.node if n.op_type == "Reshape"):
                    axes = edge2Lidx_map.get(reshape_n.input[1])
                    if not axes:
                        continue

                    new_shape = list(edge2value[reshape_n.input[1]])
                    if len(new_shape) == 3:
                        # attn-output merge: [1, seq, num_attention_heads * head_dim]
                        explicit = q_heads * head_dim
                    else:
                        # QKV split: head-count from the projection weight out-features (Q=q_heads, K/V=kv_heads)
                        width = _projection_out_features(reshape_n, bwd_dict, edge2shape)
                        if width == q_heads * head_dim:
                            explicit = q_heads
                        elif width == kv_heads * head_dim:
                            explicit = kv_heads
                        elif width is not None:
                            explicit = width // head_dim
                        else:
                            raise RuntimeError(
                                f"`allowzero=1`: cannot resolve head-count for Reshape {reshape_n.name!r} in {func.name!r}."
                            )

                    for i, v in enumerate(new_shape):  # previously-inferred axis → explicit fixed size
                        if v == -1:
                            new_shape[i] = explicit
                    for i in axes:  # seq axis → inferred -1
                        new_shape[i] = -1

                    out_name = f"{reshape_n.name}__shape_azoff"
                    value = onnx.numpy_helper.from_array(np.array(new_shape, dtype=np.int64), out_name)
                    new_consts.append(
                        onnx.helper.make_node("Constant", [], [out_name], value=value, name=f"const_{out_name}")
                    )
                    reshape_n.input[1] = out_name  # un-share: point at the private constant

                if new_consts:  # prepend (Constants have no inputs → topological order preserved)
                    existing = list(func.node)
                    del func.node[:]
                    func.node.extend(new_consts + existing)

        # NOTE: set Reshape(allowzero = allowzero)
        for func in model.functions:
            for node in (n for n in func.node if n.op_type == "Reshape"):
                update_node_attribute(node, attribute_name="allowzero", value=allowzero)

        # NOTE: shapes inference from scratch
        del model.graph.value_info[:]
        onnx.save(model, str(onnx_path))
        model = onnx.shape_inference.infer_shapes_path(onnx_path, check_type=True, strict_mode=False)

    # ── prefill: locate + register the TRACE_L seq axes ──────────────────────
    @staticmethod
    def _scan_and_register_seq_axes(model: onnx.ModelProto) -> Dict[Tuple[str, str], Set[int]]:
        """
        Fill `_reshape_seqlen_axis`.
            * key=(normalized function name, edge name)
            * value=nonzero_indices
        """
        assert not any(n.op_type == "Reshape" for n in model.graph.node), (
            "main graph holds Reshape node(s); CausalLMONNXShaper assumes every Reshape lives "
            "inside a FuncProto (else its seq axis would keep the baked TRACE_L)."
        )
        _reshape_seqlen_axis: Dict[Tuple[str, str], Set[int]] = {}
        for func in model.functions:  # NOTE: `FunctionProto` has no graph-level initializers
            fn_key = _normalize_funcproto_name(func.name)
            for name, tp in filter_shape_metadata(func.node, ()):
                shape_npy = onnx.numpy_helper.to_array(tp).ravel()
                idxs_tobe_zero = {int(i) for i in np.flatnonzero(shape_npy == TRACE_L)}
                if idxs_tobe_zero:
                    _reshape_seqlen_axis.setdefault((fn_key, name), set()).update(idxs_tobe_zero)
        return _reshape_seqlen_axis


__all__ = ["CausalLMONNXShaper"]
