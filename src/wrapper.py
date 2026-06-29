from typing import Any, Dict, List, Optional, Sequence

import torch
import transformers
from transformers.cache_utils import DynamicCache

from utils import ModuleIOSpec, PluginRegisterInterface
from constant import _NON_HASHABLE_PARAMS
from utils.tracing.hooks import HookRegisterInterface
from utils.tracing.register import export_case, register_dynamic_cache_pytree


class _DecodeWrapper(torch.nn.Module):
    """Wrap a model so its KV cache is passed as flat ``list[Tensor]`` kwargs.

    torch.export struggles with ``DynamicCache`` as a top-level input — the
    deep pytree flattening (28 layers × K/V = 56 leaves) trips an IndexError
    in the FX → ONNX decomposition pass.  This wrapper takes ``past_keys`` and
    ``past_values`` as plain list-of-Tensor kwargs, rebuilds the cache inside
    its own ``forward``, and forwards everything else through to the wrapped
    model.  The 56 K/V tensors become named, flat graph inputs.
    """

    def __init__(self, model: torch.nn.Module, pkv_kwarg: str = "past_key_values"):
        super().__init__()
        self.model = model
        self._pkv_kwarg = pkv_kwarg

    def forward(self, past_keys, past_values, **kwargs):
        cache = DynamicCache(ddp_cache_data=list(zip(past_keys, past_values)))
        kwargs[self._pkv_kwarg] = cache
        out = self.model(**kwargs)
        # Pull the per-layer K, V back out so they become named graph outputs.
        # plugin_forward writes the op's K_new/V_new into cache.layers[i] at
        # trace time, so reading them here lifts those tensors to the model's
        # return — without this they get DCE'd.
        new_keys = [layer.keys for layer in cache.layers]
        new_values = [layer.values for layer in cache.layers]
        logits = out.logits if hasattr(out, "logits") else out[0]
        return logits, new_keys, new_values


class CausalLMTracer(PluginRegisterInterface, HookRegisterInterface):
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        plugin_suffix: Sequence[str] = ("Attention", "RotaryEmbedding"),
    ):
        self.plugin_suffix = (plugin_suffix,) if isinstance(plugin_suffix, str) else plugin_suffix 
        self.plugin_ios: Dict[str, ModuleIOSpec] = {}

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

    def trace_graph(self, model_inputs: Dict[str, Any], **generate_kwargs):
        register_dynamic_cache_pytree()

        # Phase 1: observe original (unpatched) modules during generation
        self._attach_hooks()
        self.model.generate(**model_inputs, **generate_kwargs)
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

        If any case includes a ``DynamicCache`` under a KV key, the model is
        transparently wrapped in ``_DecodeWrapper`` and the cache is splayed
        into ``past_keys`` / ``past_values`` list[Tensor] kwargs so the export
        graph carries them as named, flat inputs.
        """
        for i, inputs in enumerate(case_inputs):
            path = path_template.format(i=i + 1)
            export_target, export_kwargs = self._maybe_unwrap_cache(inputs)
            token = export_case.set(i)
            try:
                torch.onnx.export(
                    export_target,
                    args=(),
                    kwargs=export_kwargs,
                    f=path,
                    opset_version=opset_version,
                    custom_translation_table=self.custom_onnx_translation,
                )
                breakpoint()
            finally:
                export_case.reset(token)
            print(f"ONNX export successful (case {i + 1}): {path}")

    def _maybe_unwrap_cache(self, inputs: Dict[str, Any]):
        """If `inputs` contains a DynamicCache under any KV key, splay it out.

        Returns ``(export_target, export_kwargs)``: when no cache is present,
        ``(self.model, inputs)`` is returned unchanged.  Otherwise the cache is
        replaced by ``past_keys`` and ``past_values`` list[Tensor] kwargs and
        the target becomes a fresh ``_DecodeWrapper``.
        """
        pkv_key = next(
            (k for k in _NON_HASHABLE_PARAMS if isinstance(inputs.get(k), DynamicCache)),
            None,
        )
        if pkv_key is None:
            return self.model, inputs

        cache: DynamicCache = inputs[pkv_key]
        unwrapped = dict(inputs)
        unwrapped.pop(pkv_key)
        unwrapped["past_keys"] = [layer.keys for layer in cache.layers]
        unwrapped["past_values"] = [layer.values for layer in cache.layers]
        return _DecodeWrapper(self.model, pkv_kwarg=pkv_key), unwrapped
