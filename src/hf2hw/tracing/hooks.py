import functools
import inspect
from abc import ABC
from typing import Dict, List

import torch

from hf2hw.utils.py_helper import check_parent_field
from .tensor_metadata import TensorSpec, ModuleIOSpec, INPUT_SPECS_TYPE, OUTPUT_SPECS_TYPE


class HookRegisterInterface(ABC):
    def __init__(self):
        self.model_ios: ModuleIOSpec | None = None
        self.plugin_ios: Dict[str, ModuleIOSpec] = {}
        self._hook_handles: List[torch.utils.hooks.RemovableHook] = []
        check_parent_field(self, "_module2name")
        check_parent_field(self, "model")

    def _model_pre_hook(self, module, hook_args, hook_kwargs):
        """Capture model input"""
        _cls_name: str = module.__class__.__name__
        _module_name: str = self._module2name[module]

        sig = inspect.signature(module.forward)
        bound_args = sig.bind(*hook_args, **hook_kwargs)
        bound_args.apply_defaults()
        name2val = bound_args.arguments
        name2val = name2val | name2val.pop("kwargs", {})

        obsvd_inputs: INPUT_SPECS_TYPE = {
            param_name: TensorSpec.from_tensor(value, module=module)
            for param_name, value in name2val.items()
        }
        if self.model_ios is None:
            self.model_ios = ModuleIOSpec(_cls_name, _module_name)
        self.model_ios.obsvd_input_specs.append(obsvd_inputs)

    def _model_post_hook(self, module, hook_args, hook_kwargs, output):
        """Capture model output"""
        _ts = TensorSpec.from_tensor(output)
        obsvd_outputs: OUTPUT_SPECS_TYPE = (_ts,) if isinstance(_ts, TensorSpec) else _ts
        self.model_ios.obsvd_output_specs.append(obsvd_outputs)

    def _input_pre_hook(self, module, hook_args, hook_kwargs):
        """Capture plugin input"""
        _cls_name: str = module.__class__.__name__
        _module_name: str = self._module2name[module]

        sig = inspect.signature(module.forward)
        bound_args = sig.bind(*hook_args, **hook_kwargs)
        bound_args.apply_defaults()
        name2val = bound_args.arguments
        name2val = name2val | name2val.pop("kwargs", {})
        assert name2val, f"Empty `{_cls_name}.forward` input"

        obsvd_inputs: INPUT_SPECS_TYPE = {
            param_name: TensorSpec.from_tensor(value, module=module)
            for param_name, value in name2val.items()
        }
        ms = self.plugin_ios.setdefault(
            f"{_cls_name}::{_module_name}",
            ModuleIOSpec(_cls_name, _module_name),
        )
        ms.obsvd_input_specs.append(obsvd_inputs)

    def _output_post_hook(self, module, hook_args, hook_kwargs, output):
        """Capture plugin output"""
        _cls_name: str = module.__class__.__name__
        _module_name: str = self._module2name[module]

        _ts = TensorSpec.from_tensor(output)
        obsvd_outputs: OUTPUT_SPECS_TYPE = (_ts,) if isinstance(_ts, TensorSpec) else _ts

        self.plugin_ios[f"{_cls_name}::{_module_name}"].obsvd_output_specs.append(obsvd_outputs)

    def _attach_hooks(self):
        self.model_ios = None
        self.plugin_ios.clear()
        self._hook_handles = []

        # NOTE: model IO catcher
        self._hook_handles.append(
            self.model.register_forward_pre_hook(self._model_pre_hook, with_kwargs=True)
        )
        self._hook_handles.append(
            self.model.register_forward_hook(self._model_post_hook, with_kwargs=True)
        )

        # NOTE: plugin IO catcher
        suffix2modules = self.get_plugin_modules()
        if isinstance(suffix2modules, list):
            iter_modules = suffix2modules
        else:
            iter_modules = (m for ml in suffix2modules.values() for m in ml)
        for module in iter_modules:
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

    @staticmethod
    def register_plugin_io_hooks():
        def decorator(fn):
            @functools.wraps(fn)
            def wrapper(self, *args, **kwargs):
                self._attach_hooks()
                try:
                    return fn(self, *args, **kwargs)
                finally:
                    self._detach_hooks()
            return wrapper
        return decorator


__all__ = ["HookRegisterInterface"]
