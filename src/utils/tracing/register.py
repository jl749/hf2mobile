from typing import Any, Dict, List, Tuple

import onnx_ir
import torch
from torch.onnx._internal.exporter import _core as _onnx_core

from .tensor_metadata import TensorSpec, INPUT_SPECS_TYPE, OUTPUT_SPECS_TYPE
from .inspect import FwdSpec, fwdspecs2kwargs
from utils.constant import CUSTOM_LIB_NAME, CUSTOM_LIB, ONNX_DOMAIN_NAME


# op_name -> (fwdspecs, num_outputs)
_registered_ops: Dict[str, Tuple[List[FwdSpec], int]] = {}

# id(module) -> callable(**tensor_kwargs) -> tensor(s)
_module_fwd_registry: Dict[int, Any] = {}

# op_overload -> ONNX translation callable; passed as custom_translation_table
CUSTOM_ONNX_TRANSLATIONS: Dict[Any, Any] = {}





def _input_seq_len(input_spec: INPUT_SPECS_TYPE) -> int:
    """Query length from an input dict — used to discriminate prefill vs decode."""
    pid = input_spec.get("position_ids")
    if isinstance(pid, TensorSpec) and pid.shape and len(pid.shape) >= 2:
        return pid.shape[1]
    for spec in input_spec.values():
        if isinstance(spec, TensorSpec) and spec.shape and len(spec.shape) >= 2:
            return spec.shape[1]
    return 0


def _ensure_op_registered(
    module: torch.nn.Module,
    fwdspecs: List[FwdSpec],
    input_spec: INPUT_SPECS_TYPE,
    output_spec: OUTPUT_SPECS_TYPE,
) -> str:
    """Register a torch.library op for a (cls_name, query_seq_len) pair.

    The op name is ``{cls_name}_fwd_q{seq_len}`` so prefill (seq_len > 1) and
    decode (seq_len == 1) get separate ops, each with its own abstract impl and
    therefore correct static output shapes.  All module instances of the same
    class share the registered op — dispatch happens via the leading
    ``module_id`` argument at call time.

    Args:
        module: module instance — only its ``__class__.__name__`` is used here.
        fwdspecs: forward-signature metadata (name / kind / count per parameter).
                  Does not carry shape or dtype — those come from `input_spec`
                  and `output_spec`.
        input_spec: per-call input dict ({param_name: TensorSpec | nested}).
                    Used to compute the query seq_len for op naming.
        output_spec: per-call output structure, possibly nested e.g.
                     ``(TensorSpec, [TensorSpec, TensorSpec])``.  Recursively
                     flattened to drive both num_outputs and per-output
                     shape/dtype in the abstract impl.
    Returns:
        op_name
    """
    cls_name = module.__class__.__name__
    # seq_len = _input_seq_len(input_spec)
    op_name = f"{cls_name}_fwd"
    if op_name in _registered_ops:
        return op_name
    custom_op_name = f"{cls_name}Plugin"

    _leaves, _ = torch.utils._pytree.tree_flatten(output_spec)
    flat_os = [v for v in _leaves if isinstance(v, TensorSpec) and not v.is_empty]
    num_outputs = len(flat_os)
    assert num_outputs > 0, f"`output_spec` passed is empty. Please check what `ModuleIOSpec.unique_ios` returns."

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
    schema = _schema_str(op_name, fwdspecs, num_outputs)
    CUSTOM_LIB.define(schema)

    @torch.library.impl(CUSTOM_LIB, op_name, "CPU")
    def _cpu_impl(module_id, *flat_tensors):
        fwd = _module_fwd_registry[module_id]
        kwargs = fwdspecs2kwargs(fwdspecs, list(flat_tensors))
        return fwd(**kwargs)

    @torch.library.register_fake(f"{CUSTOM_LIB_NAME}::{op_name}")
    def _abstract_impl(module_id, *flat_tensors):
        device = next(t for t in flat_tensors if t is not None).device
        outs = tuple(torch.empty(ts.shape, dtype=ts.torch_dtype, device=device) for ts in flat_os)
        return outs

    def _onnx_translation(module_id, *flat_tensors):
        inputs = [t for t in flat_tensors if t is not None]
        node = onnx_ir.Node(ONNX_DOMAIN_NAME, custom_op_name, inputs, num_outputs=num_outputs)
        _onnx_core.current_tracer.nodes.append(node)
        return node.outputs[0] if num_outputs == 1 else tuple(node.outputs)

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
