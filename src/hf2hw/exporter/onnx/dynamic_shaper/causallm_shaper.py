"""Dynamic-shape post-processing for the exported two-case CausalLM ONNX graphs.

After ``torch.onnx.export`` every dimension is frozen from the traced example.
``CausalLMONNXShaper`` rewrites the input/output shapes to symbolic dims and
re-runs ``onnx.shape_inference`` so the sequence length propagates through the
main graph.

The frozen sequence axes inside the attention FunctionProtos are located
**structurally** via ``AttentionIdentifier``: the Q/K/V head-split Reshapes and the
output-merge Reshape are found by backtracing from the ``Attention`` node, and their
seq axis is axis 1 by construction (the head split runs before the Transpose, in
``(B, S, ...)`` layout). No trace sentinel or prefill→generation registry is needed —
each case graph is rewritten independently.

How the seq axis is rewritten depends on ``allowzero``:

  allowzero=0 (default)
    Set the seq axis to ``0`` ("copy the input dim at that axis") and reset the node's
    ``allowzero=0`` so the copy is honored. The head-count axis stays ``-1`` (inferred),
    so a single shape constant can stay shared across Q/K/V (different head counts).

  allowzero=1
    Some runtimes reject the ``0``-copy. Instead make the seq axis the inferred ``-1``
    and bake the head-count / hidden axis explicitly (derived from the graph by
    ``AttentionIdentifier``). Because Q/K/V share one constant but need different head
    counts, each Reshape gets its **own** private shape constant (un-shared):
        query4dReshape -> [1, -1, num_heads,    head_dim]
        key4dReshape   -> [1, -1, kv_num_heads, head_dim]
        value4dReshape -> [1, -1, kv_num_heads, head_dim]
        out3dReshape   -> [1, -1, num_heads * head_dim]

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
from typing import List, Literal

import numpy as np
import onnx
import transformers

from hf2hw.constant import ONNX_DOMAIN_NAME, SUBGRAPH_MAP_TYPE
from hf2hw.exporter.onnx.fusion import AttentionIdentifier
from hf2hw.exporter.onnx.merge import merge_subgraphs_into_model
from hf2hw.utils.logger import logger
from hf2hw.utils.onnx_helper import drop_vi_by_name, set_vi_axis, update_node_attribute


class CausalLMONNXShaper(ABC):
    def __init__(self, hf_config: transformers.PreTrainedConfig) -> None:
        self.hf_config = hf_config

    # ================ subgraph merge  ================ #
    @abstractmethod
    def _post_process_final_onnx(self, case_idx: int, onnx_path: str | PathLike):
        """Postprocess method that optimizes the final merged ONNX graph"""
        pass

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

        `allowzero=0` writes the seq axis as a `0`-copy (default).
        `allowzero=1` makes the seq axis the inferred `-1` and bakes the head-count / hidden axis explicitly.
        """
        if allowzero not in (0, 1):
            raise ValueError(f"`{allowzero=}` is invalid; expected 0 or 1.")

        model: onnx.ModelProto = onnx.load(onnx_path, load_external_data=True)
        assert not any(
            n.op_type == "Reshape" for n in model.graph.node
        ), "Main graph holds Reshape node(s); CausalLMONNXShaper assumes every Reshape lives inside a FuncProto."
        drop_vi_by_name(model.graph.input, {"attention_mask"})  # TODO: attention_bias? ALiBi?

        # NOTE: IO ValueInfoProto update (make QueryL dynamic)
        for vi in model.graph.input:
            if vi.name in ("input_ids", "position_ids"):
                set_vi_axis(vi, 1, seq_sym)
        for vi in model.graph.output:
            if vi.name in ("logits",):
                set_vi_axis(vi, 1, seq_sym)
        # TODO: CausalLMExporter._post_process_final_onnx
        #   will make prefill graph to output KV cache in future. make KV output dynamic too.

        # NOTE: IO KV ValueInfoProto update (make KeyL, ValueL dynamic)
        if mode == "generation":
            for vi in model.graph.input:
                if vi.name.startswith(("past_keys_", "past_values_")):
                    set_vi_axis(vi, 2, prev_sym)
            for vi in model.graph.output:
                if vi.name.startswith(("past_keys_", "past_values_")) and vi.name.endswith("_out"):
                    set_vi_axis(vi, 2, curr_sym)

        for func in model.functions:
            if not AttentionIdentifier.is_attention_func(func):
                continue
            attention = AttentionIdentifier(func, hf_config=self.hf_config)
            head_dim, n_heads, n_kv_heads = attention.head_dim, attention.num_heads, attention.kv_num_heads
            reshape2shape = {
                attention.query4dReshape.name: (attention.query4dReshape, [1, -1, n_heads, head_dim]),
                attention.key4dReshape.name: (attention.key4dReshape, [1, -1, n_kv_heads, head_dim]),
                attention.value4dReshape.name: (attention.value4dReshape, [1, -1, n_kv_heads, head_dim]),
                attention.o3dReshape.name: (attention.o3dReshape, [1, -1, n_heads * head_dim]),
            }

            if allowzero == 0:
                # NOTE: seq axis (axis 1) -> 0-copy; the shape constant may be shared across
                #   Q/K/V — the rewrite is identical for all of them, so setting it repeatedly is fine
                for reshape_node in (
                    attention.query4dReshape,
                    attention.key4dReshape,
                    attention.value4dReshape,
                    attention.out3dReshape,
                ):
                    tp = attention._get_shape_tp_from_reshape(reshape_node)
                    shape_npy = onnx.numpy_helper.to_array(tp).copy()
                    shape_npy[1] = 0
                    tp.CopyFrom(onnx.numpy_helper.from_array(shape_npy, tp.name))
            elif allowzero == 1:
                # NOTE: seq axis -> inferred -1, head/hidden axes -> explicit; Q/K/V need
                #   different head counts so each Reshape gets its own private shape constant
                new_consts: List[onnx.NodeProto] = []
                for reshape_node, new_shape in (
                    (attention.query4dReshape, attention.query4dShape),
                    (attention.key4dReshape, attention.key4dShape),
                    (attention.value4dReshape, attention.value4dShape),
                    (attention.out3dReshape, attention.out3dShape),
                ):
                    out_name = f"{reshape_node.name}__shape_NEW"
                    value = onnx.numpy_helper.from_array(np.array(new_shape, dtype=np.int64), out_name)
                    new_consts.append(
                        onnx.helper.make_node("Constant", [], [out_name], value=value, name=f"const_{out_name}")
                    )
                    reshape_node.input[1] = out_name

                new_nodes = new_consts + list(func.node)
                del func.node[:]
                func.node.extend(new_nodes)

        # NOTE: set Reshape(allowzero = allowzero)
        for func in model.functions:
            for node in (n for n in func.node if n.op_type == "Reshape"):
                update_node_attribute(node, attribute_name="allowzero", value=allowzero)

        # NOTE: shapes inference from scratch
        del model.graph.value_info[:]
        onnx.save(model, str(onnx_path))
        model = onnx.shape_inference.infer_shapes_path(onnx_path, check_type=True, strict_mode=False)


__all__ = ["CausalLMONNXShaper"]
