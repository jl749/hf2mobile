from typing import Any, Dict, List, Sequence

import torch
import transformers
from transformers.cache_utils import DynamicCache

from .constant import _NON_HASHABLE_PARAMS, KV_CACHE_PARAM_NAME
from .tracing import (
    TracerInterface, PluginRegisterInterface, HookRegisterInterface,
    export_case, register_dynamic_cache_pytree,
)


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


class CausalLMExporter(TracerInterface, PluginRegisterInterface, HookRegisterInterface):
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        plugin_suffix: Sequence[str] = ("Attention", "RotaryEmbedding"),
    ):
        TracerInterface.__init__(self, model, plugin_suffix)
        PluginRegisterInterface.__init__(self)
        HookRegisterInterface.__init__(self)

        register_dynamic_cache_pytree()

    @HookRegisterInterface.register_plugin_io_hooks()
    def trace_plugin_ios(self, model_inputs: Dict[str, Any], **generate_kwargs) -> None:
        self.model.generate(**model_inputs, **generate_kwargs)

    # def _build_case_inputs(self) -> List[Dict[str, Any]]:
    #     """Build concrete model kwargs for each unique case.

    #     ``model_ios.build_pseudo_unique_inputs()`` already materializes every
    #     captured `TensorSpec` into a `torch.zeros` (or scalar) leaf, preserving
    #     the nested `past_key_values=[(K, V), ...]` structure.  Two model-level
    #     adjustments remain:
    #       - If `past_key_values` is present (decode), wrap the list of pairs
    #         in a `DynamicCache` so the model can consume it.
    #       - Otherwise (prefill), set `use_cache=False` so the model doesn't
    #         emit a `DynamicCache` with empty K/V layers, which breaks FX
    #         decomposition.
    #     """
    #     breakpoint()
    #     cases = self.model_ios.build_pseudo_unique_inputs()
    #     for tensors in cases:
    #         pkv = tensors.get(KV_CACHE_PARAM_NAME)
    #         if isinstance(pkv, list) and pkv and isinstance(pkv[0], tuple):
    #             tensors[KV_CACHE_PARAM_NAME] = DynamicCache(ddp_cache_data=pkv)
    #         else:
    #             tensors["use_cache"] = False
    #     return cases

    def export(
        self,
        model_inputs: Dict[str, Any],
        path_template: str = "case{i}.onnx",
        opset_version: int = 22,
        **kwargs,
    ):
        self.trace_plugin_ios(model_inputs, **kwargs)
        self.register_plugins()  # requires `trace_plugin_ios` to be ran first

        for i, inputs in enumerate(self.model_ios.build_pseudo_unique_inputs()):
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
