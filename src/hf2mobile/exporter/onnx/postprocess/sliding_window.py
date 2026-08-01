from typing import Dict, List

import numpy as np
import onnx_ir as ir
import torch

from hf2mobile.constant import ONNX_DOMAIN_NAME, _swa_onnx_template_ir
from hf2mobile.exporter.onnx import AttentionIdentifier
from hf2mobile.utils.logger import logger
from hf2mobile.utils.onnx_helper import append_node_input


def _get_sliding_window_mask_function(func_name: str, sliding_window: int, opset_version: int) -> ir.Function:
    """
    Turn the template graph into a `SlidingWindowMask_{window_size}` FunctionProto with the window_size baked in.
    NOTE: window_size=0 doesn't mean sliding window attention is turned off.
      In order to ignore the sliding window attention please DO NOT include this graph.

    [PREFILL]
      A = arange(Lq).unsqueeze(0)
      B = (arange(Lq) - window_size).unsqueeze(1)
      (A > B)[None, None, ...] -> (1, 1, Lq, Lq)
      e.g. window_size=3
          [1 1 1 1 1]
          [1 1 1 1 1]
          [1 1 1 1 1]    +    causal_mask   ->   final_mask
          [0 1 1 1 1]
          [0 0 1 1 1]
    [GENERATION]
      A = [[0]]
      B = (arange(Lkv+1) - window_size).unsqueeze(1)
      (A > B)[None, None, ...] -> (1, 1, 1, Lkv+1)
      e.g. window_size=3
          [0 0 1 1 1]    +    causal_mask   ->   final_mask
    """
    graph = _swa_onnx_template_ir().graph  # a fresh clone per call — see `_swa_onnx_template_ir`

    # NOTE: overwrite the `swa_windowsize` Constant
    for node in graph:
        if node.op_type == "Constant" and node.outputs and node.outputs[0].name == "swa_windowsize":
            node.attributes["value"] = ir.AttrTensor(
                "value", ir.tensor(np.array(sliding_window, np.int64), name="swa_windowsize")
            )

    func = ir.Function(domain=ONNX_DOMAIN_NAME, name=func_name, graph=graph, attributes=())
    func.opset_imports[""] = opset_version
    return func


# NOTE: SlidingWindowMask require Lq and Lkv which can be obtained from "input_ids" and "past_keys_0"
_Q_LEN_SRC, _KV_LEN_SRC = "input_ids", "past_keys_0"


def attach_sliding_window_mask_onnx(model_ir: ir.Model, name2module: Dict[str, torch.nn.Module]) -> None:
    """
    Construct and attach sliding window mask (takes Lq and Lkv -> construct (N, 1, Lq, Lq+Lkv)) to model_ir
    (see `src/hf2mobile/exporter/onnx/postprocess/sliding_window_mask.onnx`)
    Sliding window info can be obtained directly from name2module filtered by using FunctionProto name
    (e.g. "Qwen3Attention____model__layers__0__self_attn____case1")
    """
    funcname2winsize: Dict[str, int] = {}
    for attn_func in [f for f in model_ir.functions.values() if AttentionIdentifier.is_attention_func(f)]:
        _m = name2module[attn_func.name.split("____")[1].replace("__", ".")]
        if getattr(_m, "layer_type", "") == "sliding_attention":
            funcname2winsize[attn_func.name] = _m.sliding_window
    if not funcname2winsize:
        return  # no sliding window found

    opset_version = model_ir.opset_imports.get("", None) or model_ir.opset_imports["ai.onnx"]
    gen_mask_edge = lambda winsize: f"swa_mask_{winsize}"  # noqa: E731

    graph: ir.Graph = model_ir.graph
    by_name: Dict[str, ir.Value] = {v.name: v for v in graph.inputs}
    q_len_src, kv_len_src = (
        by_name[_Q_LEN_SRC],
        by_name[_KV_LEN_SRC],
    )  # WARNING: fails if _Q_LEN_SRC, _KV_LEN_SRC not found

    # one shared mask computation per distinct winsize, prepended to the main graph
    mask_nodes: List[ir.Node] = []
    mask_values: Dict[int, ir.Value] = {}
    for winsize in sorted(set(funcname2winsize.values())):
        func_name = f"SlidingWindowMask_{winsize}"
        func_identifier = (ONNX_DOMAIN_NAME, func_name, "")
        if func_identifier not in model_ir.functions:
            model_ir.functions[func_identifier] = _get_sliding_window_mask_function(func_name, winsize, opset_version)
        out = ir.Value(name=gen_mask_edge(winsize))
        mask_values[winsize] = out
        mask_nodes.append(
            ir.Node(
                ONNX_DOMAIN_NAME,
                func_name,
                inputs=[q_len_src, kv_len_src],
                outputs=[out],
                name=f"node_{func_name}",
            )
        )

    # add the shared mask as an extra `Attention` input inside each sliding attention function
    for attn_func in model_ir.functions.values():
        winsize = funcname2winsize.get(attn_func.name, None)
        if winsize is None:
            continue  # filter non sliding window attention blocks
        mask_in = ir.Value(name=gen_mask_edge(winsize))
        attn_func.inputs.append(mask_in)  # NOTE: extend the function signature
        attn_node = next(n for n in attn_func if n.op_type == "Attention")
        append_node_input(attn_node, mask_in)  # NOTE: (!!ORDER MATTERS!!) extend Attention(Q, K, V, attn_mask)
        logger.debug(f"  routed {gen_mask_edge(winsize)} into `{attn_func.name}`")

    # pass the shared mask at each sliding attention call site in the main graph
    for node in graph:
        winsize = funcname2winsize.get(node.op_type, None) if node.domain == ONNX_DOMAIN_NAME else None
        if winsize is not None:
            append_node_input(node, mask_values[winsize])

    # mask calls only depend on graph inputs -> valid at the front of the main graph
    for node in reversed(mask_nodes):
        graph.insert_before(next(iter(graph)), node)


__all__ = ["attach_sliding_window_mask_onnx"]
