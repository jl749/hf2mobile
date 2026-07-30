import logging
import os
from contextlib import contextmanager
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import List, Sequence

import onnx
import torch
import transformers

from hf2mobile.constant import (
    GENERATION_CONFIG_FILE,
    INPUT_KWARGS_TYPE,
    KV_CACHE_PARAM_NAME,
    SUPPORTED_TARGETS,
    TOKENIZER_CONFIG_FILE,
    TOKENIZER_FILE,
)
from hf2mobile.exporter.onnx.postprocess.sliding_window import attach_sliding_window_mask_onnx
from hf2mobile.tracing import (
    HookRegisterInterface,
    PluginRegisterInterface,
    TracerInterface,
    export_case,
    register_dynamic_cache_pytree,
)
from hf2mobile.utils.logger import logger
from hf2mobile.utils.onnx_helper import optimize_onnx, save_onnx

from .onnx.fusion import fuse_group_query_attention, fuse_rms_norm
from .onnx.postprocess import CausalLMONNXPostprocessor
from .submodules import SubgraphExporterInterface


class CausalLMExporter(
    TracerInterface,
    PluginRegisterInterface,
    HookRegisterInterface,
    SubgraphExporterInterface,
    CausalLMONNXPostprocessor,
):
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
        CausalLMONNXPostprocessor.__init__(self, model.config)

        register_dynamic_cache_pytree()

    # ================ ABSTRACT METHODS  ================ #
    @HookRegisterInterface.register_plugin_io_hooks()
    def trace_plugin_ios(self) -> None:
        """Model inference logic for tracing"""
        _TRACE_L = 11
        _MAX_GENERATION_STEPS = 3
        assert self.model.config.sliding_window is None or self.model.config.sliding_window >= _TRACE_L, (
            f"{self.model.config.sliding_window=} < {_TRACE_L=}: the window would truncate within the "
            "trace length and mask suppression during export would change traced behavior."
        )
        trace_input_ids = torch.LongTensor([[57] * _TRACE_L])
        self.model.generate(input_ids=trace_input_ids, max_new_tokens=_MAX_GENERATION_STEPS)

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
                if logger.isEnabledFor(logging.DEBUG):
                    self.make_dynamic_onnx(onnx_path, allowzero=1)
                    model = onnx.load(onnx_path, load_external_data=True)
                    attach_sliding_window_mask_onnx(model, self._name2module)
                    model, _n = fuse_rms_norm(model)
                    save_onnx(model, f"debug__{onnx_path}")
                    Path(f"debug__{onnx_path}.data").unlink(missing_ok=True)
                Path(onnx_path).unlink(missing_ok=True)
                Path(onnx_path).with_suffix(".onnx.data").unlink(missing_ok=True)
            elif case_idx == 1:
                # ===== POSTPROCESS: dynamic IO + Reshape ===== #
                self.make_dynamic_onnx(onnx_path, allowzero=1)
                logger.info(f"  {LOG_PREFIX} applied dynamic shaping on `{onnx_path=}`")

                # ===== FUSION: fuse the main graph RMSNorms ===== #
                model = onnx.load(onnx_path, load_external_data=True)
                model, _n = fuse_rms_norm(model)  # fuse main graph RMSNorms (subgraphs are already fused)
                if _n:
                    logger.info(f"  {LOG_PREFIX} fused {_n} main-graph RMSNorm(s) in {onnx_path}")

                # ===== POSTPROCESS: attach sliding window ===== #
                attach_sliding_window_mask_onnx(model, self._name2module)

                # ========================== save debug_case2.onnx ============================ #
                # ===== flat FuncProtos, fold Constants, drop floating nodes(s)/tensor(s) ===== #
                if logger.isEnabledFor(logging.DEBUG):
                    save_onnx(model, f"debug__{onnx_path}")
                    Path(f"debug__{onnx_path}.data").unlink(missing_ok=True)
                model = optimize_onnx(model)

                # ===== FUSION: fuse the main graph GroupQueryAttentions ===== #
                model, _n = fuse_group_query_attention(model, self.hf_config)  # must be called after optimize_onnx
                if _n:
                    logger.info(f"  {LOG_PREFIX} fused {_n} main-graph GroupQueryAttention(s) in {onnx_path}")
                # TODO: SkipLayerNormalization, SkipSimplifiedLayerNormalization, SimplifiedLayerNormalization fusing

                save_onnx(model, onnx_path)
                logger.info(f"  {LOG_PREFIX} flattened model local function and saved under `{onnx_path=}`")
        elif self.target == "QNN":
            return NotImplemented
            # model = onnx.load(onnx_path, load_external_data=True)
            # if case_idx == 0:
            #     # TODO: make Attention K,V input as ONNX output for prefill caching
            #     # TODO: copy quant params from the generation graph
            #     pass
            # elif case_idx == 1:
            #     pass
            # # TODO: extract subgraphs excluding GQA
            # onnx.save(model, onnx_path)
        else:
            raise RuntimeError(f"`CausalLMExporter.export(...)` has not assigned `self.target`.")

    # ================ CausalLMExporter METHODS  ================ #
    def save_tokenizer_config(self) -> List[str]:
        saved: List[str] = []

        repo_id = self.model.config._name_or_path
        tokenizer = transformers.AutoTokenizer.from_pretrained(repo_id, use_fast=True)
        saved += [os.path.basename(path) for path in tokenizer.save_pretrained(".")]
        assert TOKENIZER_FILE in saved, f"`tokenizer.json` not saved. ({repo_id=})"

        try:
            self.model.generation_config.save_pretrained(".")
            saved.append(GENERATION_CONFIG_FILE)
            eos = self.model.generation_config.eos_token_id
        except Exception as e:
            logger.warning(f"  could not save {GENERATION_CONFIG_FILE}: {e}")
            eos = tokenizer.eos_token_id
        finally:
            if eos is not None:
                eos_token_ids = [int(i) for i in (eos if isinstance(eos, (list, tuple)) else [eos])]
            else:
                raise ValueError(f"No EOS tokens found under {TOKENIZER_CONFIG_FILE}, {GENERATION_CONFIG_FILE}")

        if eos_token_ids:
            logger.info(f"  🏁 extracted EOS tokens: {eos_token_ids} 🏁 ")

        logger.info(f"  saved runtime configs: {sorted(set(saved))}")
        return saved

    def export(
        self,
        target: str,
        path_template: str = "case{i}.onnx",
        opset_version: int = 25,
    ) -> List[str]:
        """Export HF model to ONNX (export 2 unique cases - prefill, generation)"""
        case_paths: List[str] = []

        start_dir = os.getcwd()
        try:
            work_dir = datetime.now().strftime(
                f"%Y-%m-%d_%H-%M-%S__{target}__{self.model.config._name_or_path.replace('/', '-')}"
            )
            os.makedirs(work_dir, exist_ok=True)
            os.chdir(work_dir)

            self.save_tokenizer_config()

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

        return [str(Path(work_dir).joinpath(p)) for p in case_paths]


__all__ = ["CausalLMExporter"]
