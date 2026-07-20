from os import PathLike
from typing import List

import numpy as np
import onnx
import transformers

from hf2hw.exporter.onnx import AttentionIdentifier
from hf2hw.utils.onnx_helper import drop_vi_by_name, save_onnx, set_vi_axis, update_node_attribute

from ._base_postprocessor import _ONNXPostprocessor


class CausalLMONNXPostprocessor(_ONNXPostprocessor):
    def __init__(self, hf_config: transformers.PreTrainedConfig) -> None:
        self.hf_config = hf_config

    def make_dynamic_onnx(
        self,
        onnx_path: str | PathLike,
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
        ), "Main graph holds Reshape node(s); CausalLMONNXPostprocessor assumes every Reshape lives inside a FuncProto."
        drop_vi_by_name(model.graph.input, {"attention_mask"})  # TODO: attention_bias? ALiBi?

        # NOTE: IO ValueInfoProto update (make QueryL dynamic)
        for vi in model.graph.input:
            if vi.name in ("input_ids", "position_ids"):
                set_vi_axis(vi, 1, seq_sym)
        for vi in model.graph.output:
            if vi.name in ("logits",):
                set_vi_axis(vi, 1, seq_sym)

        # NOTE: IO KV ValueInfoProto update (make KeyL, ValueL dynamic)
        for vi in model.graph.input:
            if vi.name.startswith(("past_keys_", "past_values_")):
                set_vi_axis(vi, 2, prev_sym)
        for vi in model.graph.output:
            if vi.name.startswith(("past_keys_", "past_values_")) and vi.name.endswith("_out"):
                set_vi_axis(vi, 2, curr_sym)

        for attn_func in (f for f in model.functions if AttentionIdentifier.is_attention_func(f)):
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

                new_nodes = new_consts + list(attn_func.node)
                del attn_func.node[:]
                attn_func.node.extend(new_nodes)

            # ===== set Reshape(allowzero = allowzero) ===== #
            for node in (n for n in attn_func.node if n.op_type == "Reshape"):
                update_node_attribute(node, attribute_name="allowzero", value=allowzero)
            # ===== set Attention(is_causal = 1) ===== #
            for node in (n for n in attn_func.node if n.op_type == "Attention"):
                update_node_attribute(node, attribute_name="is_causal", value=1)

        del model.graph.value_info[:]
        save_onnx(model, onnx_path)


__all__ = ["CausalLMONNXPostprocessor"]
