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


def _inject_sliding_window_mask(model: onnx.ModelProto, attn_func: onnx.FunctionProto, sliding_window: int) -> None:
    """Route a `SlidingWindowMask_{window}` FuncProto call into this attention FuncProto's `Attention`.

    Adds the (shared) mask FuncProto to `model.functions` once, then inserts a call
    `SlidingWindowMask(hidden_states, past_key) -> mask` right before the `Attention` node and
    appends `mask` as its 4th input. Both functions get flattened together by the later inliner.
    """
    attn_identifier = AttentionIdentifier(attn_func, hf_config=None)  # rebuilt post-reshape-rewrite (stale-ref safe)
    func_name = f"SlidingWindowMask_{sliding_window}"
    if not any(f.name == func_name and f.domain == ONNX_DOMAIN_NAME for f in model.functions):
        opset_version = next(oi.version for oi in model.opset_import if oi.domain in ("", "ai.onnx"))
        model.functions.append(_get_sliding_window_mask_funcproto(func_name, sliding_window, opset_version))

    inner_func_node = onnx.helper.make_node(
        func_name,
        [attn_identifier.queryMM.input[0], attn_identifier.keyConcat.input[0]],
        ["swa_mask"],
        domain=ONNX_DOMAIN_NAME,
        name="node_SlidingWindowMask",
    )
    attn_idx = next(i for i, n in enumerate(attn_func.node) if n.op_type == "Attention")
    attn_func.node[attn_idx].input.append("swa_mask")  # Attention(Q, K, V, attn_mask)
    nodes = list(attn_func.node)
    nodes.insert(attn_idx, inner_func_node)
    del attn_func.node[:]
    attn_func.node.extend(nodes)


def attach_sliding_window_mask_onnx(model: onnx.ModelProto, name2module: Dict[str, torch.nn.Module]):
    for attn_func in (f for f in model.functions if AttentionIdentifier.is_attention_func(f)):
        _m = name2module[attn_func.name.split("____")[1].replace("__", ".")]
        if getattr(_m, "layer_type", "") == "sliding_attention":
            _inject_sliding_window_mask(model, attn_func, _m.sliding_window)
            logger.debug(f"  injected sliding-window mask (window={_m.sliding_window}) into `{attn_func.name}`")


__all__ = ["attach_sliding_window_mask_onnx"]
