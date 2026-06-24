import contextvars
import inspect
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import transformers

from utils import convert_dtype, FwdSpec, sig2fwdspecs, sig2num_outputs, TensorSpec, ModuleIOSpec
from utils.constant import INPUT_SPECS_TYPE, OUTPUT_SPECS_TYPE
from utils.tracing.inspect import _flatten_call_args
from utils.tracing.register import (
    CUSTOM_ONNX_TRANSLATIONS,
    _ensure_op_registered,
    _module_fwd_registry,
)


# ---------------------------------------------------------------------------
# Export-mode context variable
# ---------------------------------------------------------------------------
# None  → generation (plugin_forward always bypasses to original forward)
# "prefill" | "decode" → ONNX export; plugin_forward calls the matching op
export_mode: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "export_mode", default=None
)


# ---------------------------------------------------------------------------
# Wrapper class
# ---------------------------------------------------------------------------

class CausalLMWrapper:
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        model_inputs: Dict[str, Any],
        plugin_suffix: Sequence[str] = ("Attention", "RotaryEmbedding"),
        **generate_kwargs,
    ):
        self.model = model
        self.plugin_suffix = plugin_suffix

        self._module2name = {mod: name for name, mod in model.named_modules()}
        self._hook_handles: List[torch.utils.hooks.RemovableHook] = []

        self.plugin_ios: Dict[str, ModuleIOSpec] = {}

        # Phase 1: observe original (unpatched) modules during generation
        self._attach_hooks()
        self.model.generate(**model_inputs, **generate_kwargs)
        self._detach_hooks()

        # Phase 2: patch modules using captured IO shapes
        self._apply_plugin_wrappers()

    # ------------------------------------------------------------------
    # Plugin wrappers
    # ------------------------------------------------------------------

    def _apply_plugin_wrappers(self):
        name2modules_dict: Dict[str, List[torch.nn.Module]] = self.get_plugin_modules(plugin_suffix=self.plugin_suffix)
        for suffix, modules in name2modules_dict.items():
            custom_op_name = f"Custom{suffix}"
            if not modules:
                continue  # TODO: logger warning
            cls_names = set(m.__class__.__name__ for m in modules)
            if len(cls_names) > 1:
                raise RuntimeError(
                    f"More than one class candidate for suffix '{suffix}': {cls_names}"
                )

            for module in modules:
                orig_cls = module.__class__
                _module_name: str = self._module2name[module]

                sig = inspect.signature(orig_cls.forward)
                fwdspecs: List[FwdSpec] = sig2fwdspecs(sig)

                mode_ops: Dict[str, Any] = {}
                mode_specs: Dict[str, List[FwdSpec]] = {}

                prefill, decode = self.plugin_ios[f"{orig_cls.__name__}::{_module_name}"].unique_ios()
                for input_specs, output_specs in (prefill, decode):
                    # TODO: fwdspecs.resolve_unknown(input_spec)
                    _num_outputs = len(output_spec)
                    breakpoint()

                    op_name = _ensure_op_registered(
                        module=module,
                        fwdspecs=fwdspecs,
                        input_spec=input_specs,
                        output_spec=output_specs,
                    )

                    # All modes share the same CPU forward (use_cache=False during export)
                    if id(module) not in _module_fwd_registry:
                        _module_fwd_registry[id(module)] = self._make_module_fwd(
                            module=module,
                            orig_cls=orig_cls,
                            sig=sig,
                            fwdspecs=resolved,
                            num_outputs=_num_outputs,
                        )

                    mode_ops[mode] = getattr(torch.ops.hf_module2plugin, op_name)
                    mode_specs[mode] = resolved

                traceable_fwd = self._make_plugin_forward(
                    module=module,
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

    @staticmethod
    def _make_module_fwd(
        module: torch.nn.Module,
        orig_cls,
        sig: inspect.Signature,
        fwdspecs: List[FwdSpec],
        num_outputs: int,
    ):
        defaults = {
            name: param.default
            for name, param in sig.parameters.items()
            if param.default is not inspect.Parameter.empty
        }

        def fwd(**tensor_kwargs):
            kwargs = dict(defaults)
            kwargs.update(tensor_kwargs)
            result = orig_cls.forward(module, **kwargs)
            if isinstance(result, tuple):
                tensors = [v for v in result if isinstance(v, torch.Tensor)]
            else:
                tensors = [result]
            assert len(tensors) == num_outputs, (
                f"{orig_cls.__name__}: expected {num_outputs} tensor output(s), got {len(tensors)}"
            )
            return tensors[0] if num_outputs == 1 else tuple(tensors[:num_outputs])

        return fwd

    @staticmethod
    def _make_plugin_forward(
        module,
        orig_cls,
        sig: inspect.Signature,
        mode_ops: Dict[str, Any],
        mode_specs: Dict[str, List[FwdSpec]],
    ):
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

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_graphs(
        self,
        prefill_inputs: Dict[str, Any],
        decode_inputs: Dict[str, Any],
        prefill_path: str = "prefill.onnx",
        decode_path: str = "decode.onnx",
        opset_version: int = 21,
    ):
        """Export one ONNX graph per mode (prefill / decode) with static shapes."""
        configs = [
            ("prefill", prefill_inputs, prefill_path),
            ("decode",  decode_inputs,  decode_path),
        ]
        for mode, inputs, path in configs:
            # Check that at least one class has an op for this mode
            has_ops = any(
                self._get_io(cls_name, mode) is not None
                for cls_name in self.plugin_ios
            )
            if not has_ops:
                print(f"[export_graphs] skipping '{mode}': no observations captured")
                continue

            token = export_mode.set(mode)
            try:
                torch.onnx.export(
                    self.model,
                    args=(),
                    kwargs={**inputs, "use_cache": False},
                    f=path,
                    opset_version=opset_version,
                    custom_translation_table=CUSTOM_ONNX_TRANSLATIONS,
                )
            finally:
                export_mode.reset(token)
            print(f"ONNX export successful ({mode}): {path}")

    # ------------------------------------------------------------------
    # Module enumeration
    # ------------------------------------------------------------------

    def get_plugin_modules(
        self,
        *,
        plugin_suffix: Optional[str | Sequence[str]] = None,
    ) -> Dict[str, List[torch.nn.Module]] | List[torch.nn.Module]:
        plugin_suffix = plugin_suffix or self.plugin_suffix
        if isinstance(plugin_suffix, str):
            plugin_suffix = [plugin_suffix]
        suffix2modules = {}
        for suffix in plugin_suffix:
            suffix2modules[suffix] = [
                m for m in self.model.modules() if suffix in m.__class__.__name__
            ]
        if len(suffix2modules) == 1:
            return next(iter(suffix2modules.values()))
        return suffix2modules

    # ------------------------------------------------------------------
    # Observation hooks
    # ------------------------------------------------------------------

    def _input_pre_hook(self, module, hook_args, hook_kwargs):
        """Pre-hook: capture inputs into _pending_inputs before the forward runs."""
        _cls_name: str = module.__class__.__name__
        _module_name: str = self._module2name[module]

        _sig = inspect.signature(module.forward)
        _bound_args = _sig.bind(*hook_args, **hook_kwargs)
        _bound_args.apply_defaults()
        name2val = _bound_args.arguments
        name2val = name2val | name2val.pop("kwargs", {})
        assert name2val, f"Empty `{_cls_name}.forward` input"

        obsvd_inputs: INPUT_SPECS_TYPE = {
            param_name: TensorSpec.from_tensor(value, module=module)
            for param_name, value in name2val.items()
        }
        ms = self.plugin_ios.setdefault(f"{_cls_name}::{_module_name}", ModuleIOSpec(_cls_name, _module_name))
        ms.input_specs.append(obsvd_inputs)

    def _output_post_hook(self, module, hook_args, hook_kwargs, output):
        """Post-hook: append this call's output to the class accumulator."""
        _cls_name: str = module.__class__.__name__
        _module_name: str = self._module2name[module]

        _ts = TensorSpec.from_tensor(output)
        obsvd_outputs: OUTPUT_SPECS_TYPE = (_ts,) if isinstance(_ts, TensorSpec) else _ts

        self.plugin_ios[f"{_cls_name}::{_module_name}"].output_specs.append(obsvd_outputs)

    def _attach_hooks(self):
        self.plugin_ios.clear()
        self._hook_handles = []
        for module in (m for ml in self.get_plugin_modules().values() for m in ml):
            self._hook_handles.append(
                module.register_forward_pre_hook(self._input_pre_hook, with_kwargs=True)
            )
            self._hook_handles.append(
                module.register_forward_hook(self._output_post_hook, with_kwargs=True)
            )

    def _detach_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
