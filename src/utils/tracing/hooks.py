

from abc import ABC

from utils.py_helper import check_field


class HookRegisterInterface(ABC):
    def __init__(self):
        self.plugin_ios: Dict[str, ModuleIOSpec] = {}
        self._hook_handles: List[torch.utils.hooks.RemovableHook] = []
        check_field(self, "_module2name")

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
