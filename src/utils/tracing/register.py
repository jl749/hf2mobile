from typing import Any, Dict, List, Optional, Tuple

import onnx_ir
import torch
from torch.onnx._internal.exporter import _core as _onnx_core

from .tensor_metadata import INPUT_SPECS_TYPE, OUTPUT_SPECS_TYPE
from .inspect import FwdSpec, fwdspecs2kwargs
from utils.constant import CUSTOM_LIB_NAME


_custom_lib = torch.library.Library(CUSTOM_LIB_NAME, "DEF")

# op_name -> (fwdspecs, num_outputs)
_registered_ops: Dict[str, Tuple[List[FwdSpec], int]] = {}

# id(module) -> callable(**tensor_kwargs) -> tensor(s)
_module_fwd_registry: Dict[int, Any] = {}

# op_overload -> ONNX translation callable; passed as custom_translation_table
CUSTOM_ONNX_TRANSLATIONS: Dict[Any, Any] = {}


def _schema_str(op_name: str, fwdspecs: List[FwdSpec], num_outputs: int) -> str:
    parts = ["int module_id"]
    for s in fwdspecs:
        if s.kind == "tensor":
            parts.append(f"Tensor {s.name}")
        elif s.kind == "optional_tensor":
            parts.append(f"Tensor? {s.name}")
        elif s.kind == "tuple_tensor":
            for i in range(s.count):
                parts.append(f"Tensor {s.name}_{i}")
    ret = "Tensor" if num_outputs == 1 else f"({', '.join('Tensor' for _ in range(num_outputs))})"
    return f"{op_name}({', '.join(parts)}) -> {ret}"


def _ensure_op_registered(
    module: torch.nn.Module,
    fwdspecs: List[FwdSpec],
    input_specs: INPUT_SPECS_TYPE,
    output_specs: OUTPUT_SPECS_TYPE,
) -> str:
    """Register a torch.library op for (cls_name, mode) if not already done.

    op_name is ``{cls_name}_fwd_{mode}`` so prefill and decode get separate ops
    with separate abstract impls (and therefore correct static output shapes).
    Returns op_name.
    """
    op_name = f"{module.__class__.__name__}_fwd_{mode}"
    if op_name in _registered_ops:
        return op_name
    custom_op_name = f"{module.__class__.__name__}Plugin"
    num_outputs = len(output_spec)

    schema = _schema_str(op_name, fwdspecs, num_outputs)
    _custom_lib.define(schema)

    _ps = fwdspecs
    _no = num_outputs
    _out_specs = output_specs  # List[(shape, dtype)] or None

    @torch.library.impl(_custom_lib, op_name, "CPU")
    def _cpu_impl(module_id, *flat_tensors):
        fwd = _module_fwd_registry[module_id]
        kwargs = myparamlist2kwargs(_ps, list(flat_tensors))
        return fwd(**kwargs)

    @torch.library.register_fake(f"{CUSTOM_LIB_NAME}::{op_name}")
    def _abstract_impl(module_id, *flat_tensors):
        device = next(t for t in flat_tensors if t is not None).device
        if _out_specs:
            outs = tuple(torch.empty(shape, dtype=dtype, device=device)
                         for shape, dtype in _out_specs)
        else:
            first = next(t for t in flat_tensors if t is not None)
            outs = tuple(torch.empty_like(first) for _ in range(_no))
        return outs[0] if _no == 1 else outs

    _ot = custom_op_name
    _no2 = num_outputs

    def _onnx_translation(module_id, *flat_tensors):
        inputs = [t for t in flat_tensors if t is not None]
        node = onnx_ir.Node("com.jerry", _ot, inputs, num_outputs=_no2)
        _onnx_core.current_tracer.nodes.append(node)
        return node.outputs[0] if _no2 == 1 else tuple(node.outputs)

    op_overload = getattr(getattr(torch.ops.hf_module2plugin, op_name), "default")
    CUSTOM_ONNX_TRANSLATIONS[op_overload] = _onnx_translation

    _registered_ops[op_name] = (fwdspecs, num_outputs)
    return op_name


__all__ = [
    "CUSTOM_LIB_NAME",
    "CUSTOM_ONNX_TRANSLATIONS",
    "_ensure_op_registered",
    "_module_fwd_registry",
]
