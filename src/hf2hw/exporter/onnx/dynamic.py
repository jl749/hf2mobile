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
    outputs:  past_keys_*_out[1,H,L_curr,E], past_values_*_out[1,H,L_curr,E]
              (L_curr = L_prev+1; expressed as a distinct symbol because ONNX
               dim_param is a free string, not arithmetic)
"""

from typing import Sequence

import onnx

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

    # Outputs: updated KV dim-2 → L_curr  (= L_prev + 1)
    for vi in model.graph.output:
        if (
            vi.name.endswith(("_keys_out", "_values_out"))
            or ("_keys_" in vi.name and vi.name.endswith("_out"))
            or ("_values_" in vi.name and vi.name.endswith("_out"))
        ):
            _set_dim(vi, 2, curr_sym)


def _is_generation(model: onnx.ModelProto) -> bool:
    return any(vi.name.startswith("past_keys_") for vi in model.graph.input)


# ── public API ────────────────────────────────────────────────────────────────


def make_dynamic_shapes(
    model: onnx.ModelProto,
    *,
    seq_sym: str = "L",
    prev_sym: str = "L_prev",
    curr_sym: str = "L_curr",
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

    # Clear stale intermediate shapes and re-propagate.
    # Note: shape inference cannot cross FunctionProto subblock boundaries,
    # so only main-graph intermediates between subblocks are updated.
    del model.graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=False)

    n_vi = len(model.graph.value_info)
    logger.debug(f"make_dynamic_shapes: shape inference filled {n_vi} value_info entries")
    return model


__all__ = ["make_dynamic_shapes"]
