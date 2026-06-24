from typing import Any, Dict, List, Optional, Sequence

import torch
import transformers

from utils import ModuleIOSpec, PluginRegisterInterface
from utils.tracing.hooks import HookRegisterInterface
from utils.tracing.register import export_case


class CausalLMTracer(PluginRegisterInterface, HookRegisterInterface):
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        plugin_suffix: Sequence[str] = ("Attention", "RotaryEmbedding"),
        **generate_kwargs,
    ):
        self.plugin_suffix = plugin_suffix
        self.plugin_ios: Dict[str, ModuleIOSpec] = {}
        self.generate_kwargs = generate_kwargs

        self.model = model
        self._module2name = {mod: name for name, mod in model.named_modules()}
        self._id2module = {id(mod): mod for mod in model.modules()}

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
        self.model.generate(**model_inputs, **self.generate_kwargs)
        self._detach_hooks()

        # Phase 2: patch modules using captured IO shapes
        self.register_plugins()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_graphs(
        self,
        case_inputs: List[Dict[str, Any]],
        path_template: str = "case{i}.onnx",
        opset_version: int = 21,
    ):
        """Export one ONNX graph per case in ``case_inputs``.

        ``case_inputs[i]`` is the kwargs dict used to drive the trace for case
        ``i`` — its shapes must match the i-th entry of every plugin module's
        ``unique_ios()`` (e.g. case 0 = prefill kwargs, case 1 = decode kwargs
        with a populated KV cache).  The library makes no assumption about what
        "case i" means — caller picks the inputs that match the captured profile.
        """
        for i, inputs in enumerate(case_inputs):
            path = path_template.format(i=i + 1)
            token = export_case.set(i)
            try:
                torch.onnx.export(
                    self.model,
                    args=(),
                    kwargs=inputs,
                    f=path,
                    opset_version=opset_version,
                    custom_translation_table=self.custom_onnx_translation,
                )
            finally:
                export_case.reset(token)
            print(f"ONNX export successful (case {i + 1}): {path}")
