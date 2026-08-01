from typing import List

import numpy as np
import onnx_ir as ir
import transformers

from hf2mobile.exporter.onnx import AttentionIdentifier
from hf2mobile.utils.onnx_helper import (
    drop_graph_io_by_name,
    get_const_tensor,
    make_constant_node,
    prepend_nodes_to_graph,
    set_value_axis,
    update_node_attribute,
)

from ._base_postprocessor import _ONNXPostprocessor


class CausalLMONNXPostprocessor(_ONNXPostprocessor):
    def __init__(self, hf_config: transformers.PreTrainedConfig) -> None:
        self.hf_config = hf_config

    def make_dynamic_onnx(
        self,
        model_ir: ir.Model,
        *,
        allowzero: int = 0,
        seq_sym: str = "L",
        prev_sym: str = "L_prev",
        curr_sym: str = "L_prev+1",
    ) -> None:
        """
        Rewrite the CausalLM ONNX graph so that it can take dynamic seq_len.
        IMPORTANT: ONNX must still be in FunctionProto layout (call it before `optimize_onnx`)

        `allowzero=0` writes the seq axis as a `0`-copy (default).
        `allowzero=1` makes the seq axis the inferred `-1` and bakes the head-count / hidden axis explicitly.
        """
        if allowzero not in (0, 1):
            raise ValueError(f"`{allowzero=}` is invalid; expected 0 or 1.")

        graph: ir.Graph = model_ir.graph
        assert not any(
            n.op_type == "Reshape" for n in graph
        ), "Main graph holds Reshape node(s); CausalLMONNXPostprocessor assumes every Reshape lives inside a function."
        drop_graph_io_by_name(graph.inputs, {"attention_mask"})  # TODO: attention_bias? ALiBi?

        # NOTE: IO update (make QueryL dynamic)
        for value in graph.inputs:
            if value.name in ("input_ids", "position_ids"):
                set_value_axis(value, 1, seq_sym)
        for value in graph.outputs:
            if value.name in ("logits",):
                set_value_axis(value, 1, seq_sym)

        # NOTE: IO KV update (make KeyL, ValueL dynamic)
        for value in graph.inputs:
            if value.name.startswith(("past_keys_", "past_values_")):
                set_value_axis(value, 2, prev_sym)
        for value in graph.outputs:
            if value.name.startswith(("past_keys_", "past_values_")) and value.name.endswith("_out"):
                set_value_axis(value, 2, curr_sym)

        for attn_func in [f for f in model_ir.functions.values() if AttentionIdentifier.is_attention_func(f)]:
            attention = AttentionIdentifier(attn_func, hf_config=self.hf_config)

            # ===== Reshape(split_head, merge_head) update ===== #
            if allowzero == 0:
                # NOTE: seq axis (axis 1) -> 0-copy; the shape constant may be shared across
                #   Q/K/V — the rewrite is identical for all of them, so setting it repeatedly is fine
                for reshape_node in (
                    attention.query4dReshape,
                    attention.key4dReshape,
                    attention.value4dReshape,
                    attention.out3dReshape,
                ):
                    shape_value = reshape_node.inputs[1]
                    tensor = get_const_tensor(shape_value)
                    shape_npy = tensor.numpy().copy()
                    shape_npy[1] = 0
                    new_tensor = ir.tensor(shape_npy, name=tensor.name)
                    producer = shape_value.producer()
                    if producer is not None and producer.op_type == "Constant":
                        producer.attributes["value"] = ir.AttrTensor("value", new_tensor)
                    else:
                        shape_value.const_value = new_tensor
            elif allowzero == 1:
                # NOTE: seq axis -> inferred -1, head/hidden axes -> explicit; Q/K/V need
                #   different head counts so each Reshape gets its own private shape constant
                new_consts: List[ir.Node] = []
                for reshape_node, new_shape in (
                    (attention.query4dReshape, attention.query4dShape),
                    (attention.key4dReshape, attention.key4dShape),
                    (attention.value4dReshape, attention.value4dShape),
                    (attention.out3dReshape, attention.out3dShape),
                ):
                    out_name = f"{reshape_node.name}__shape_NEW"
                    const_node = make_constant_node(out_name, np.array(new_shape, dtype=np.int64))
                    new_consts.append(const_node)
                    reshape_node.replace_input_with(1, const_node.outputs[0])

                prepend_nodes_to_graph(attn_func, new_consts)

            # ===== set Reshape(allowzero = allowzero) ===== #
            for node in (n for n in attn_func if n.op_type == "Reshape"):
                update_node_attribute(node, attribute_name="allowzero", value=allowzero)
            # ===== set Attention(is_causal = 1) ===== #
            for node in (n for n in attn_func if n.op_type == "Attention"):
                update_node_attribute(node, attribute_name="is_causal", value=1)


__all__ = ["CausalLMONNXPostprocessor"]
