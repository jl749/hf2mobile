import os
from contextlib import contextmanager
from datetime import datetime
from os import PathLike
from typing import List, Sequence

import onnx
import torch
import transformers

from hf2hw.constant import INPUT_KWARGS_TYPE, KV_CACHE_PARAM_NAME, SUPPORTED_TARGETS, TRACE_L
from hf2hw.tracing import (
    HookRegisterInterface,
    PluginRegisterInterface,
    TracerInterface,
    export_case,
    register_dynamic_cache_pytree,
)
from hf2hw.utils.logger import logger

from .onnx import make_dynamic_shapes
from .onnx.fusion import fuse_rms_norm
from .submodules import SubgraphExporterInterface


class CausalLMExporter(TracerInterface, PluginRegisterInterface, HookRegisterInterface, SubgraphExporterInterface):
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        plugin_suffix: Sequence[str],
    ):
        """
        Every CausalLM model is expected to inherit this class
        Based on target recipe and model choice we pass different hyperparams
            * __init__(..., plugin_suffix)
            * export(..., model_inputs, **input_kwargs)
        """
        TracerInterface.__init__(self, model, plugin_suffix)
        PluginRegisterInterface.__init__(self)
        HookRegisterInterface.__init__(self)
        SubgraphExporterInterface.__init__(self)

        register_dynamic_cache_pytree()

    # ================ ABSTRACT METHODS  ================ #
    @HookRegisterInterface.register_plugin_io_hooks()
    def trace_plugin_ios(self) -> None:
        """Model inference logic for tracing"""
        trace_input_ids = torch.LongTensor([[57] * TRACE_L])
        self.model.generate(input_ids=trace_input_ids, max_new_tokens=3)

    @contextmanager
    def adapt_model_for_case(self, input_dict: INPUT_KWARGS_TYPE):
        """Replace the generation forward to prevent DCE from dropping the KV cache IOs."""
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

    def _post_process_final_onnx(self, case_idx: int, onnx_path: str | PathLike):
        """
        Postprocess method that optimizes the final FuncProto merged ONNX graph
        Called at the end of `self.merge_subgraphs_into_main_graph(...)`
        """
        LOG_PREFIX = "POSTPROCESS:"
        if case_idx >= 2:
            raise RuntimeError(f"`{case_idx=}` is not a valid CausalLM supported only prefill(0) and generation(1).")

        if self.target == "ORT":
            if case_idx == 0:
                os.remove(onnx_path)
            elif case_idx == 1:
                model = onnx.load(onnx_path, load_external_data=True)
                model, _n = fuse_rms_norm(model)
                if _n:
                    logger.info(f"  {LOG_PREFIX} fused {_n} main-graph RMSNorm(s) in {onnx_path}")
                # TODO: GroupQueryAttention, SkipLayerNormalization, SkipSimplifiedLayerNormalization, SimplifiedLayerNormalization fusing
                model = make_dynamic_shapes(model)  # NOTE: make gen graph generic (cover prefill)
                onnx.save(model, onnx_path)
                logger.info(f"  {LOG_PREFIX} dynamic shapes written: {onnx_path}")
        elif self.target == "QNN":
            return NotImplemented
            model = onnx.load(onnx_path, load_external_data=True)
            if case_idx == 0:
                # TODO: make Attention K,V input as ONNX output for prefill caching
                # TODO: copy quant params from the generation graph
                pass
            elif case_idx == 1:
                pass
            # TODO: extract subgraphs excluding GQA
            onnx.save(model, onnx_path)
        else:
            raise RuntimeError(f"`CausalLMExporter.export(...)` has not assigned `self.target`.")

    # ================ CausalLMExporter METHODS  ================ #
    def export(
        self,
        target: str,
        path_template: str = "case{i}.onnx",
        opset_version: int = 25,
    ) -> None:
        """Export HF model to ONNX (export 2 unique cases - prefill, generation)"""
        start_dir = os.getcwd()
        try:
            work_dir = datetime.now().strftime(
                f"%Y-%m-%d_%H-%M-%S__{target}__{self.model.config._name_or_path.replace('/', '-')}"
            )
            os.makedirs(work_dir, exist_ok=True)
            os.chdir(work_dir)

            self.target = target.upper()
            assert (
                self.target in SUPPORTED_TARGETS
            ), f"Unsupported `{target=}`. Currently supported targets are: `{SUPPORTED_TARGETS}`."

            logger.info("Stage 1/5: tracing plugin IOs via model.generate(...)")
            self.trace_plugin_ios()
            uniq_input_dicts: List[INPUT_KWARGS_TYPE] = self.model_ios.pseudo_unique_inputs()
            num_uniq_cases = len(uniq_input_dicts)
            assert (
                num_uniq_cases == 2
            ), f"For CausalLM only 2 trace cases are allowed - Prefill, Generation(`{num_uniq_cases=}`)."

            logger.info("Stage 2/5: exporting submodules as standalone subgraphs")
            subgraph_paths = self.export_plugin_subgraphs(opset_version=opset_version)

            logger.info("Stage 3/5: registering custom plugin ops")
            self.register_plugins()  # requires `trace_plugin_ios` to be ran first

            logger.info(f"Stage 4/5: exporting {len(uniq_input_dicts)} ONNX case(s) — model level")
            case_paths: List[str] = []
            for i, input_dict in enumerate(uniq_input_dicts):
                path = path_template.format(i=i + 1)
                logger.info(f"  case {i + 1}/{num_uniq_cases} → {path}")
                token = export_case.set(i)
                try:
                    with self.adapt_model_for_case(input_dict) as (export_kwargs, output_names):
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

            logger.info("Stage 5/5: merging Subgraphs into the main graphs and postprocessing")
            self.merge_subgraphs_into_main_graph(case_paths, subgraph_paths)
        except Exception as e:
            raise RuntimeError("CausalLMExporter.export(...) failed.") from e
        finally:
            os.chdir(start_dir)


__all__ = ["CausalLMExporter"]
