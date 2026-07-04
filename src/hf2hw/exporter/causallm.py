from contextlib import contextmanager
from typing import Any, Dict, List, Sequence

import onnx
import torch
import transformers

from hf2hw.constant import INPUT_KWARGS_TYPE, KV_CACHE_PARAM_NAME, ONNX_DOMAIN_NAME
from hf2hw.exporter import SubgraphExporterInterface
from hf2hw.tracing import (
    HookRegisterInterface,
    PluginRegisterInterface,
    TracerInterface,
    export_case,
    register_dynamic_cache_pytree,
)
from hf2hw.utils.logger import logger


class CausalLMExporter(TracerInterface, PluginRegisterInterface, HookRegisterInterface, SubgraphExporterInterface):
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        plugin_suffix: Sequence[str] = ("Attention", "RotaryEmbedding"),
    ):
        TracerInterface.__init__(self, model, plugin_suffix)
        PluginRegisterInterface.__init__(self)
        HookRegisterInterface.__init__(self)
        SubgraphExporterInterface.__init__(self)

        register_dynamic_cache_pytree()

    # ================ ABSTRACT METHODS  ================ #
    @HookRegisterInterface.register_plugin_io_hooks()
    def trace_plugin_ios(self, model_inputs: Dict[str, Any], **generate_kwargs) -> None:
        self.model.generate(**model_inputs, **generate_kwargs)

    @contextmanager
    def _adapt_model_for_case(self, input_dict: INPUT_KWARGS_TYPE):
        """By replacing the generation forward prevent DCE from dropping the KV cache IOs."""
        input_dict = dict(input_dict)  # copy: caller's dict left unchanged
        pkv = input_dict.pop(KV_CACHE_PARAM_NAME, None)

        if pkv and isinstance(pkv, list) and isinstance(pkv[0], tuple):
            # =========== trace generation =========== #
            orig_cls = self.model.__class__
            input_dict["past_keys"] = [k for k, _ in pkv]  # global keys
            input_dict["past_values"] = [v for _, v in pkv]  # global values
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

    # ================ CausalLMExporter METHODS  ================ #
    def export(
        self,
        model_inputs: INPUT_KWARGS_TYPE,
        path_template: str = "case{i}.onnx",
        opset_version: int = 25,
        **kwargs,
    ):
        """Export HF model to ONNX (export 2 unique cases - prefill, generation)"""
        logger.info("Stage 1/5: tracing plugin IOs via model.generate(...)")
        self.trace_plugin_ios(model_inputs, **kwargs)
        uniq_input_dicts: List[INPUT_KWARGS_TYPE] = self.model_ios.pseudo_unique_inputs()
        num_uniq_cases = len(uniq_input_dicts)
        assert (
            num_uniq_cases == 2
        ), f"For CausalLM only 2 trace cases are allowed - Prefill, Generation(`{num_uniq_cases=}`)."

        logger.info("Stage 2/5: exporting attention submodules + merging as SubBlocks")
        subgraph_paths = self._export_plugin_subgraphs(opset_version=opset_version)

        logger.info("Stage 3/5: registering custom plugin ops")
        self.register_plugins()  # requires `trace_plugin_ios` to be ran first

        logger.info(f"Stage 4/5: exporting {len(uniq_input_dicts)} ONNX case(s) — model level")
        case_paths: List[str] = []
        for i, input_dict in enumerate(uniq_input_dicts):
            path = path_template.format(i=i + 1)
            logger.info(f"  case {i + 1}/{num_uniq_cases} → {path}")
            token = export_case.set(i)
            try:
                with self._adapt_model_for_case(input_dict) as (export_kwargs, output_names):
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
            logger.info(f"  case {i + 1}/{num_uniq_cases} ✓ written: {path}")
            case_paths.append(path)

        # TODO: _merge_subgraphs_into_main_graph (this is a new step)

        logger.info("Stage 5/5: making shapes dynamic + re-running shape inference")
        for path in case_paths:
            model = onnx.load(path, load_external_data=True)
            model = make_dynamic_shapes(model)
            onnx.save(model, path)
            logger.info(f"  dynamic shapes written: {path}")
