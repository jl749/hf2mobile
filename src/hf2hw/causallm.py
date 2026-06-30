from contextlib import contextmanager
from typing import Any, Dict, Sequence

import torch
import transformers

from .constant import INPUT_KWARGS, KV_CACHE_PARAM_NAME
from .tracing import (
    HookRegisterInterface,
    PluginRegisterInterface,
    TracerInterface,
    export_case,
    register_dynamic_cache_pytree,
)


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

    @contextmanager
    def _adapt_model_for_case(self, input_dict: INPUT_KWARGS):
        """By replacing the generation forward prevent DCE from dropping the KV cache IOs."""
        pkv = input_dict.pop(KV_CACHE_PARAM_NAME, None)

        if pkv and isinstance(pkv, list) and isinstance(pkv[0], tuple):
            # =========== trace generation =========== #
            orig_cls = self.model.__class__
            input_dict["past_keys"] = [k for k, _ in pkv]
            input_dict["past_values"] = [v for _, v in pkv]
            n = len(pkv)
            output_names = (
                ["logits"] + [f"past_keys_{i}_out" for i in range(n)] + [f"past_values_{i}_out" for i in range(n)]
            )

            def _traceable_forward(self_inner, past_keys, past_values, **kwargs):
                cache = transformers.DynamicCache(ddp_cache_data=list(zip(past_keys, past_values)))
                kwargs[KV_CACHE_PARAM_NAME] = cache
                out = orig_cls.forward(self_inner, **kwargs)
                logits = out.logits if hasattr(out, "logits") else out[0]
                new_keys = [layer.keys for layer in cache.layers]
                new_values = [layer.values for layer in cache.layers]
                return logits, new_keys, new_values

            self.model.__class__ = type(
                f"KVTraced_{orig_cls.__name__}",
                (orig_cls,),
                {"forward": _traceable_forward},
            )
            try:
                yield input_dict, output_names
            finally:
                self.model.__class__ = orig_cls
        else:
            # =========== trace prefill =========== #
            input_dict["use_cache"] = False
            yield input_dict, None
            return

    @HookRegisterInterface.register_plugin_io_hooks()
    def trace_plugin_ios(self, model_inputs: Dict[str, Any], **generate_kwargs) -> None:
        self.model.generate(**model_inputs, **generate_kwargs)

    def export(
        self,
        model_inputs: Dict[str, Any],
        path_template: str = "case{i}.onnx",
        opset_version: int = 22,
        **kwargs,
    ):
        """Export HF model to ONNX (export 2 unique cases - prefill, generation)"""
        self.trace_plugin_ios(model_inputs, **kwargs)
        self.register_plugins()  # requires `trace_plugin_ios` to be ran first

        for i, inputs in enumerate(self.model_ios.pseudo_unique_inputs()):
            path = path_template.format(i=i + 1)
            token = export_case.set(i)
            try:
                with self._adapt_model_for_case(inputs) as (export_kwargs, output_names):
                    torch.onnx.export(
                        self.model,
                        args=(),
                        kwargs=export_kwargs,
                        f=path,
                        opset_version=opset_version,
                        output_names=output_names,
                        custom_translation_table=self.custom_onnx_translation,
                    )
            finally:
                export_case.reset(token)
            print(f"ONNX export successful (case {i + 1}): {path}")
