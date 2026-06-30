from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import Any, Dict, List, Sequence

import torch
import transformers

from hf2hw.constant import INPUT_KWARGS
from hf2hw.utils.logger import logger

from .tensor_metadata import ModuleIOSpec


class TracerInterface(ABC):
    def __init__(self, model: transformers.PreTrainedModel, plugin_suffix: Sequence[str]):
        self.plugin_suffix = (plugin_suffix,) if isinstance(plugin_suffix, str) else plugin_suffix
        self.plugin_ios: Dict[str, ModuleIOSpec] = {}

        self.model = model
        self._module2name = {mod: name for name, mod in model.named_modules()}
        self._id2module = {id(mod): mod for mod in model.modules()}
        logger.info(
            f"{self.__class__.__name__} initialized on {model.__class__.__name__} "
            f"(plugin_suffix={tuple(self.plugin_suffix)}, {len(self._module2name)} named modules)"
        )

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
            suffix2modules[suffix] = [m for m in self.model.modules() if suffix in m.__class__.__name__]
        if len(suffix2modules) == 1:
            return next(iter(suffix2modules.values()))
        return suffix2modules

    @abstractmethod
    def trace_plugin_ios(self, model_inputs: Dict[str, Any], **kwargs) -> None:
        """Inference steps to export"""
        pass

    @abstractmethod
    def _adapt_model_for_case(self, input_dict: INPUT_KWARGS) -> AbstractContextManager:
        """Return a context manager that adapts `self.model` for the given case.

        Subclasses typically implement this with `@contextmanager`; the context
        block yields ``(export_kwargs, output_names)`` and restores any model
        mutations on exit.
        """
        ...
