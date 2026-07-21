import copy
from typing import Dict

import numpy as np
import onnx
import torch

from hf2hw.constant import ONNX_DOMAIN_NAME, _swa_onnx_template
from hf2hw.exporter.onnx import AttentionIdentifier
from hf2hw.utils.logger import logger


def _get_sliding_window_mask_funcproto(func_name: str, sliding_window: int, opset_version: int) -> onnx.FunctionProto:
    """
    Turn the template graph into `SlidingWindowMask_{window_size}` FuncProto with the window_size baked in.
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
    # TODO: onnx_to_function
    graph: onnx.GraphProto = _swa_onnx_template().graph
    nodes = [copy.deepcopy(n) for n in graph.node]

    # NOTE: overwrite the `swa_windowsize` Constant
    for node in nodes:
        if node.op_type == "Constant" and node.output[0] == "swa_windowsize":
            win_tp = next(a.t for a in node.attribute if a.name == "value")
            win_tp.CopyFrom(onnx.numpy_helper.from_array(np.array(sliding_window, np.int64), "swa_windowsize"))

    return onnx.helper.make_function(
        domain=ONNX_DOMAIN_NAME,
        fname=func_name,
        inputs=[i.name for i in graph.input],
        outputs=[o.name for o in graph.output],
        nodes=nodes,
        opset_imports=[onnx.helper.make_opsetid("", opset_version)],
    )


# NOTE: SlidingWindowMask require Lq and Lkv which can be obtained from "input_ids" and "past_keys_0"
_Q_LEN_SRC, _KV_LEN_SRC = "input_ids", "past_keys_0"


def attach_sliding_window_mask_onnx(model: onnx.ModelProto, name2module: Dict[str, torch.nn.Module]):
    funcname2winsize: Dict[str, int] = {}
    for attn_func in (f for f in model.functions if AttentionIdentifier.is_attention_func(f)):
        _m = name2module[attn_func.name.split("____")[1].replace("__", ".")]
        if getattr(_m, "layer_type", "") == "sliding_attention":
            funcname2winsize[attn_func.name] = _m.sliding_window
    if not funcname2winsize:
        return

    opset_version = next(oi.version for oi in model.opset_import if oi.domain in ("", "ai.onnx"))
    gen_mask_edge = lambda winsize: f"swa_mask_{winsize}"  # noqa: E731

    # one shared mask computation per distinct winsize, prepended to the main graph
    mask_func_nodes: List[onnx.NodeProto] = []
    for winsize in sorted(set(funcname2winsize.values())):
        func_name = f"SlidingWindowMask_{winsize}"
        if not any(f.name == func_name and f.domain == ONNX_DOMAIN_NAME for f in model.functions):
            model.functions.append(_get_sliding_window_mask_funcproto(func_name, winsize, opset_version))
        mask_func_nodes.append(
            onnx.helper.make_node(
                func_name,
                [_Q_LEN_SRC, _KV_LEN_SRC],
                [gen_mask_edge(winsize)],
                domain=ONNX_DOMAIN_NAME,
                name=f"node_{func_name}",
            )
        )

    # add the shared mask as an extra `Attention` input inside each sliding attention FuncProto
    for attn_func in model.functions:
        winsize = funcname2winsize.get(attn_func.name)
        if winsize is None:
            continue
        attn_func.input.append(gen_mask_edge(winsize))  # NOTE: extend the FuncProto signature
        attn_node = next(n for n in attn_func.node if n.op_type == "Attention")
        attn_node.input.append(gen_mask_edge(winsize))  # NOTE: extend Attention(Q, K, V, attn_mask)
        logger.debug(f"  routed {gen_mask_edge(winsize)} into `{attn_func.name}`")

    # pass the shared mask at each sliding attention call site in the main graph
    for node in model.graph.node:
        winsize = funcname2winsize.get(node.op_type, None) if node.domain == ONNX_DOMAIN_NAME else None
        if winsize is not None:
            node.input.append(gen_mask_edge(winsize))

    # mask calls only depend on graph inputs -> valid at the front of the main graph
    main_nodes = mask_func_nodes + list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(main_nodes)


__all__ = ["attach_sliding_window_mask_onnx"]
