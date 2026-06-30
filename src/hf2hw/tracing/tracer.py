from abc import ABC, abstractmethod
from typing import Sequence, Dict, List, Any

import torch
import transformers
from .tensor_metadata import ModuleIOSpec


class TracerInterface(ABC):
    def __init__(self, model: transformers.PreTrainedModel, plugin_suffix: Sequence[str]):
        self.plugin_suffix = (plugin_suffix,) if isinstance(plugin_suffix, str) else plugin_suffix 
        self.plugin_ios: Dict[str, ModuleIOSpec] = {}

        self.model = model
        self._module2name = {mod: name for name, mod in model.named_modules()}
        self._id2module = {id(mod): mod for mod in model.modules()}

    def get_plugin_modules(
        self,
        *,
        plugin_suffix: str | Sequence[str] | None = None,
    ) -> Dict[str, List[torch.nn.Module]] | List[torch.nn.Module]:
        """Filter torch modules from `self.model` based on `plugin_suffix`"""
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

    @abstractmethod
    def trace_plugin_ios(self, model_inputs: Dict[str, Any], **kwargs) -> None:
        """Inference steps to export"""
        pass

