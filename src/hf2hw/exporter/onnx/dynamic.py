"""Dynamic-shape post-processing for exported ONNX case files.

After ``torch.onnx.export`` all dimensions are frozen from the traced example.
This module rewrites specific input/output shapes to symbolic names and runs
``onnx.shape_inference`` to propagate them through the main graph.

Shapes set explicitly (inference cannot cross FunctionProto boundaries):

  case1 – prefill
    inputs:   input_ids[1,L], position_ids[1,L]
    dropped:  attention_mask  (no consumers in either case)
    outputs:  unchanged (already static 1-token logit)

  case2 – generation
    inputs:   past_keys_*[1,H,L_prev,E], past_values_*[1,H,L_prev,E]
    dropped:  attention_mask
    outputs:  past_keys_*_out[1,H,L_prev+1,E], past_values_*_out[1,H,L_prev+1,E]
              ("L_prev+1" is used as a literal dim_param label — ONNX dim_param is a
               free string, not arithmetic — to document the past+current relation)

Frozen reshapes (two coupled fixes):
  ``torch.onnx.export`` bakes the traced sequence length into its Reshape target shapes
  (e.g. ``[1, 123, -1, 64]``) AND stamps the node with ``allowzero=1``. To un-freeze the
  seq dim we:
    1. rewrite the sentinel trace length (``constant.TRACE_L``) to ``0`` in Reshape shape
       constants — a ``0`` means "copy the input's dim at that axis"; and
    2. reset ``allowzero=0`` so that ``0`` is honored as a copy instead of a literal
       zero-size dim.
  With both, the reshape copies the (now dynamic) input seq dim and inference propagates
  it instead of the frozen value.
"""

from typing import Sequence

import onnx
from onnx import numpy_helper

from hf2hw.constant import TRACE_L
from hf2hw.utils.logger import logger

# ── helpers ──────────────────────────────────────────────────────────────────


def _set_dim(vi: onnx.ValueInfoProto, axis: int, symbol: str) -> None:
    """Replace a fixed dim on ``vi`` with a symbolic name."""
    dim = vi.type.tensor_type.shape.dim[axis]
    dim.ClearField("dim_value")
    dim.dim_param = symbol


def _drop_inputs(model: onnx.ModelProto, names: Sequence[str]) -> None:
    """Remove named entries from graph.input (assumes they have no consumers)."""
    drop = set(names)
    keep = [vi for vi in model.graph.input if vi.name not in drop]
    del model.graph.input[:]
    model.graph.input.extend(keep)


# ── per-case logic ────────────────────────────────────────────────────────────


def _apply_prefill(model: onnx.ModelProto, seq_sym: str) -> None:
    """Rewrite case1 (prefill) inputs in-place."""
    for vi in model.graph.input:
        if vi.name in ("input_ids", "position_ids"):
            _set_dim(vi, 1, seq_sym)
    _drop_inputs(model, ["attention_mask"])


def _apply_generation(
    model: onnx.ModelProto,
    prev_sym: str,
    curr_sym: str,
) -> None:
    """Rewrite case2 (generation) inputs and outputs in-place."""
    # Inputs: past KV dim-2 → L_prev
    for vi in model.graph.input:
        if vi.name.startswith(("past_keys_", "past_values_")):
            _set_dim(vi, 2, prev_sym)
    _drop_inputs(model, ["attention_mask"])

    # Outputs: updated KV dim-2 → "L_prev+1" (past + current token)
    for vi in model.graph.output:
        if (
            vi.name.endswith(("_keys_out", "_values_out"))
            or ("_keys_" in vi.name and vi.name.endswith("_out"))
            or ("_values_" in vi.name and vi.name.endswith("_out"))
        ):
            _set_dim(vi, 2, curr_sym)


def _is_generation(model: onnx.ModelProto) -> bool:
    return any(vi.name.startswith("past_keys_") for vi in model.graph.input)


def _rebind_reshape_trace_seq(model: onnx.ModelProto, trace_len: int = TRACE_L) -> int:
    """Rewrite the baked trace sequence length to ``0`` in Reshape shape constants.

    We trace with the unique sentinel ``trace_len`` (``constant.TRACE_L``), so any
    ``trace_len`` inside a Reshape's shape tensor is the frozen sequence dim. Rewriting it
    to ``0`` makes ONNX copy that dim from the (dynamic) Reshape *input* at the same axis
    (paired with ``allowzero=0``; see ``_reshape_allowzero_off``). Both the main graph and
    every ``FunctionProto`` body are scanned — the attention/RoPE reshapes live inside the
    merged subblock functions. Returns the number of rewritten values.
    """
    n_vals = 0

    def _patch(tp: onnx.TensorProto) -> None:
        nonlocal n_vals
        arr = numpy_helper.to_array(tp)
        if trace_len not in arr:
            return
        n_vals += int((arr == trace_len).sum())
        patched = arr.copy()
        patched[patched == trace_len] = 0
        tp.CopyFrom(numpy_helper.from_array(patched, tp.name))

    def _rebind(nodes, initializers=()) -> None:
        shape_inputs = {n.input[1] for n in nodes if n.op_type == "Reshape" and len(n.input) >= 2}
        for init in initializers:
            if init.name in shape_inputs:
                _patch(init)
        for node in nodes:
            if node.op_type == "Constant" and node.output and node.output[0] in shape_inputs:
                for attr in node.attribute:
                    if attr.name == "value":
                        _patch(attr.t)

    _rebind(model.graph.node, model.graph.initializer)
    for func in model.functions:  # FunctionProto has no graph-level initializers
        _rebind(func.node)

    if n_vals:
        logger.debug(f"make_dynamic_shapes: rebound {n_vals} baked seq ({trace_len}) → 0 in Reshape shape(s)")
    return n_vals


def _reshape_allowzero_off(model: onnx.ModelProto) -> int:
    """Reset ``allowzero=0`` on every ``Reshape`` node (main graph + FunctionProto bodies).

    ``torch.onnx.export`` puts a ``0`` at the (dynamic) sequence axis of its Reshape target
    shapes — which normally means "copy the input's dim at that axis" — but ships the node
    with ``allowzero=1``, turning that ``0`` into a literal zero-size dim. Shape inference
    then can't resolve the reshape (NO-SHAPE) once the input seq dim is symbolic. Setting
    ``allowzero=0`` restores copy semantics so the sequence dim follows the dynamic input.

    Attention/RoPE reshapes live inside the merged subblock functions, so the function
    bodies are scanned too. Returns the number of nodes changed.
    """
    n = 0

    def _fix(nodes) -> None:
        nonlocal n
        for node in nodes:
            if node.op_type != "Reshape":
                continue
            for attr in node.attribute:
                if attr.name == "allowzero" and attr.i != 0:
                    attr.i = 0
                    n += 1

    _fix(model.graph.node)
    for func in model.functions:
        _fix(func.node)

    if n:
        logger.debug(f"make_dynamic_shapes: reset allowzero=0 on {n} Reshape node(s)")
    return n


# ── public API ────────────────────────────────────────────────────────────────


def make_dynamic_shapes(
    model: onnx.ModelProto,
    *,
    seq_sym: str = "L",
    prev_sym: str = "L_prev",
    curr_sym: str = "L_prev+1",
) -> onnx.ModelProto:
    """Rewrite static traced shapes to symbolic dims and re-run shape inference.

    Automatically detects prefill vs generation by the presence of
    ``past_keys_*`` inputs.

    Args:
        model:    fully-merged ONNX model (modified in-place, then returned).
        seq_sym:  symbol for the prefill sequence length (case1 inputs).
        prev_sym: symbol for the past-KV sequence length (case2 inputs).
        curr_sym: symbol for the updated-KV sequence length (case2 outputs,
                  = prev_sym + 1 semantically).

    Returns:
        The patched model after shape inference.
    """
    gen = _is_generation(model)
    if gen:
        _apply_generation(model, prev_sym, curr_sym)
        logger.debug(
            f"make_dynamic_shapes: generation — past KV → {prev_sym!r}, "
            f"output KV → {curr_sym!r}, attention_mask dropped"
        )
    else:
        _apply_prefill(model, seq_sym)
        logger.debug(f"make_dynamic_shapes: prefill — seq → {seq_sym!r}, attention_mask dropped")

    # Un-freeze the baked seq dim in Reshape shapes: rewrite TRACE_L → 0 (copy-from-input),
    # then reset allowzero=0 so that 0 is honored as a copy. Both are needed together.
    _rebind_reshape_trace_seq(model)
    _reshape_allowzero_off(model)

    # Clear stale intermediate shapes and re-propagate.
    # Note: shape inference cannot cross FunctionProto subblock boundaries,
    # so only main-graph intermediates between subblocks are updated.
    del model.graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=False)

    n_vi = len(model.graph.value_info)
    logger.debug(f"make_dynamic_shapes: shape inference filled {n_vi} value_info entries")
    return model


__all__ = ["make_dynamic_shapes"]
