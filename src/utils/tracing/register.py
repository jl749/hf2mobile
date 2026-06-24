from abc import ABC
from typing import Any, Callable, Dict, List, Tuple
import inspect

import onnx_ir
import torch
from torch.onnx._internal.exporter import _core as _onnx_core

from .tensor_metadata import TensorSpec, INPUT_SPECS_TYPE, OUTPUT_SPECS_TYPE
from .inspect import FwdSpec, fwdspecs2kwargs
from utils.constant import CUSTOM_LIB_NAME, CUSTOM_LIB, ONNX_DOMAIN_NAME





# def _input_seq_len(input_spec: INPUT_SPECS_TYPE) -> int:
#     """Query length from an input dict — used to discriminate prefill vs decode."""
#     pid = input_spec.get("position_ids")
#     if isinstance(pid, TensorSpec) and pid.shape and len(pid.shape) >= 2:
#         return pid.shape[1]
#     for spec in input_spec.values():
#         if isinstance(spec, TensorSpec) and spec.shape and len(spec.shape) >= 2:
#             return spec.shape[1]
#     return 0

def _get_torchlib_impl_module_fwd(
    orig_cls: Any,
    module: torch.nn.Module,
):
    """
    factory fn to create module.forward wrapper called inside _cpu_impl
    Args:
        orig_cls: original class (e.g. Qwen3Attention)
        module: original module instance (e.g. Qwen3Attention())
            which .forward will be overwritten
    Returns:
        custom fwd function for model inferencing (not related to tracing)
    """
    sig = inspect.signature(orig_cls.forward)
    defaults = {
        name: param.default
        for name, param in sig.parameters.items()
        if param.default is not inspect.Parameter.empty
    }
    def fwd(**tensor_kwargs): 
        kwargs = dict(defaults)
        kwargs.update(tensor_kwargs)
        output = orig_cls.forward(module, **kwargs)
        # TODO: flatten output? e.g. pytree
        return output
    return fwd

def _make_plugin_forward(
    orig_cls: Any,
    module: torch.nn.Module,
    sig: inspect.Signature,
    mode_ops: Dict[str, Any],
    mode_specs: Dict[str, List[FwdSpec]],
):
    sig = inspect.signature(orig_cls.forward)
    params_no_self = [p for n, p in sig.parameters.items() if n != "self"]
    sig_wo_self = sig.replace(parameters=params_no_self)

    def plugin_forward(self_mod, *_args, **_kwargs):
        mode = export_mode.get()

        if mode is None:
            # Generation: always use the original (unpatched) forward so KV
            # caching works correctly without routing through the custom op.
            return orig_cls.forward(module, *_args, **_kwargs)

        op = mode_ops.get(mode)
        specs = mode_specs.get(mode)
        if op is None or specs is None:
            raise RuntimeError(
                f"No op registered for mode '{mode}' on {orig_cls.__name__}. "
                f"Available: {list(mode_ops)}"
            )

        bound = sig_wo_self.bind_partial(*_args, **_kwargs)
        bound.apply_defaults()
        flat = _flatten_call_args(specs, bound.arguments)
        result = op(id(self_mod), *flat)

        ret_ann = sig.return_annotation
        if isinstance(ret_ann, type) and issubclass(ret_ann, torch.Tensor):
            return result
        if isinstance(result, tuple):
            return result + (None,) * (2 - len(result))
        return result, None

    return plugin_forward

class PluginRegisterInterface(ABC):
    def __init__(self):
        self._registered_torchlib_opname = set()
        self._id2fwd = dict()
        self.custom_onnx_translation: Dict[Any, Any] = dict()
        check_field(self, "_id2module")
        check_field(self, "_module2name")

    @staticmethod
    def create_schema_str(op_name: str, fwdspecs: List[FwdSpec], num_outputs: int) -> str:
        parts = ["int module_id"]
        for fs in fwdspecs:
            if fs.kind == "tensor":
                parts.append(f"Tensor {fs.name}")
            elif fs.kind == "optional_tensor":
                parts.append(f"Tensor? {fs.name}")
            elif fs.kind == "tuple_tensor":
                for i in range(fs.count):
                    parts.append(f"Tensor {fs.name}_{i}")
        ret = "Tensor" if num_outputs == 1 else f"({', '.join('Tensor' for _ in range(num_outputs))})"
        return f"{op_name}({', '.join(parts)}) -> {ret}"

    def _register_torchlib_op(
        self,
        torchlib_op_name: str,
        onnx_op_name: str,
        fwdspecs: List[FwdSpec],
        input_spec: INPUT_SPECS_TYPE,
        output_spec: OUTPUT_SPECS_TYPE,
    ) -> None:
        """
        Register a torch.library op
        Args:
            torchlib_op_name: unique custom torchlib op name for each module
            fwdspecs: forward-signature metadata (name / kind / count per parameter).
            input_spec: per-call input dict ({param_name: TensorSpec | nested}).
            output_spec: per-call output structure, possibly nested
        """
        if torchlib_op_name in self._registered_torchlib_opname:
            # TODO: logger warning already registered
            return
        self._registered_torchlib_opname.add(torchlib_op_name)

        _leaves, _ = torch.utils._pytree.tree_flatten(output_spec)
        flat_os = [v for v in _leaves if isinstance(v, TensorSpec) and not v.is_empty]
        num_outputs = len(flat_os)
        assert num_outputs > 0, f"`output_spec` passed is empty. Please check what `ModuleIOSpec.unique_ios` returns."

        fwdspecs = [fs.resolve_unknown(input_spec) for fs in fwdspecs]
        schema = self.create_schema_str(torchlib_op_name, fwdspecs, num_outputs)
        CUSTOM_LIB.define(schema)

        @torch.library.register_fake(f"{CUSTOM_LIB_NAME}::{torchlib_op_name}")
        def _abstract_impl(module_id, *flat_tensors):
            device = next(t for t in flat_tensors if t is not None).device
            outs = tuple(torch.empty(ts.shape, dtype=ts.torch_dtype, device=device) for ts in flat_os)
            return outs

        @torch.library.impl(CUSTOM_LIB, torchlib_op_name, "CPU")
        def _cpu_impl(module_id, *flat_tensors):
            fwd = self._id2fwd[module_id]
            kwargs = fwdspecs2kwargs(fwdspecs, list(flat_tensors))
            return fwd(**kwargs)

        def _onnx_translation(module_id, *flat_tensors):
            module = self._id2module[module_id]  # TODO: fill attr
            inputs = [t for t in flat_tensors if t is not None]
            node = onnx_ir.Node(
                domain=ONNX_DOMAIN_NAME,
                op_type=onnx_op_name,
                inputs=inputs,
                attributes=[
                    onnx_ir.AttrString("torchlib_op_name", torchlib_op_name),
                    # onnx_ir.AttrInt64(""),
                ],
                num_outputs=num_outputs
            )
            _onnx_core.current_tracer.nodes.append(node)
            return node.outputs[0] if num_outputs == 1 else tuple(node.outputs)

        op_overload = getattr(getattr(torch.ops.hf_module2plugin, torchlib_op_name), "default")
        self.custom_onnx_translation[op_overload] = _onnx_translation


    def register_plugins(self):
        """Apply the plugin registrations"""
        name2modules_dict: Dict[str, List[torch.nn.Module]] = self.get_plugin_modules(plugin_suffix=self.plugin_suffix)
        for suffix, modules in name2modules_dict.items():
            onnx_op_name = f"Custom{suffix}"
            if not modules:
                continue  # TODO: logger warning
            cls_names = set(m.__class__.__name__ for m in modules)
            if len(cls_names) > 1:
                raise RuntimeError(
                    f"More than one class candidate for suffix '{suffix}': {cls_names}"
                )

            for orig_m in modules:
                orig_cls = orig_m.__class__
                name: str = self._module2name[orig_m]

                fwdspecs: List[FwdSpec] = sig2fwdspecs(sig)

                # mode_ops: Dict[str, Any] = {}
                # mode_specs: Dict[str, List[FwdSpec]] = {}

                # e.g. for CausalLM it will be len 2 [prefill, decode]
                unique_io_cases = self.plugin_ios[f"{orig_cls.__name__}::{name}"].unique_ios()
                self._id2module[id(orig_m)]._trace_metadata = unique_io_cases
                for input_specs, output_specs in unique_io_cases:
                    breakpoint()
                    torchlib_op_name = f"{orig_cls.__name__}::{name}::forward"
                    self._register_torchlib_op(
                        torchlib_op_name=torchlib_op_name,
                        onnx_op_name=onnx_op_name,
                        fwdspecs=fwdspecs,
                        input_spec=input_specs,
                        output_spec=output_specs,
                    )
                    if id(orig_m) not in self._id2fwd:
                        # can be skipped in case "inference fwd" is already registered
                        # this is un-related to the statically shaped "tracing fwd"
                        self._id2fwd[id(orig_m)] = _get_torchlib_impl_module_fwd(orig_cls, orig_m)

                    # mode_ops[mode] = getattr(torch.ops.hf_module2plugin, torchlib_op_name)
                    # mode_specs[mode] = resolved

                # make fwd function traceable
                traceable_fwd = _make_plugin_forward(
                    module=orig_module,
                    orig_cls=orig_cls,
                    sig=sig,
                    mode_ops=mode_ops,
                    mode_specs=mode_specs,
                )
                module.__class__ = type(
                    f"Traceable{orig_cls.__name__}",
                    (orig_cls,),
                    {"forward": traceable_fwd},
                )


__all__ = [
    "CUSTOM_LIB_NAME",
    "CUSTOM_ONNX_TRANSLATIONS",
    "register_torchlib_op",
]
