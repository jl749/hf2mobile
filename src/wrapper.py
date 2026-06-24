import contextvars
import inspect
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import transformers

from utils import register_torchlib_op, convert_dtype, FwdSpec, sig2fwdspecs, sig2num_outputs, TensorSpec, ModuleIOSpec, PluginRegisterInterface
from utils.constant import INPUT_SPECS_TYPE, OUTPUT_SPECS_TYPE
from utils.tracing.inspect import _flatten_call_args
from utils.tracing.register import (
    CUSTOM_ONNX_TRANSLATIONS,
    
    _ensure_op_registered,
    MODULE2FWD_REGISTRY,
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

class CausalLMTracer(PluginRegisterInterface, HookRegisterInterface):
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        
        plugin_suffix: Sequence[str] = ("Attention", "RotaryEmbedding"),
        **generate_kwargs,
    ):
        self.plugin_suffix = plugin_suffix
        self.plugin_ios: Dict[str, ModuleIOSpec] = {}

        self.model = model
        self._module2name = {mod: name for name, mod in model.named_modules()}
        self._id2module = {id(mod): module for mod in model.modules()}

        PluginRegisterInterface.__init__(self)
        HookRegisterInterface.__init__(self)


    def get_plugin_modules(
        self,
        *,
        plugin_suffix: Optional[str | Sequence[str]] = None,
    ) -> Dict[str, List[torch.nn.Module]] | List[torch.nn.Module]:
        """Helper method that filters torch modules based on `plugin_suffix` from self.model"""
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

    def trace_graph(self, model_inputs: Dict[str, Any]):
        # Phase 1: observe original (unpatched) modules during generation
        self._attach_hooks()
        self.model.generate(**model_inputs, **generate_kwargs)
        self._detach_hooks()

        # Phase 2: patch modules using captured IO shapes
        self._apply_plugin_wrappers()

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
