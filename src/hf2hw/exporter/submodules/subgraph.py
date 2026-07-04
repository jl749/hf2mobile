from abc import ABC
from collections import defaultdict
from typing import Dict, List

import onnx

from ...constant import INPUT_SPECS_TYPE, ONNX_DOMAIN_NAME
from ...utils import check_parent_field, create_torchlib_op_name, suppress_onnx_export_logs
from ...utils.logger import logger
from ..onnx import fuse_rms_norm, merge_subgraphs_into_model
from .attention import export as attn_export


class SubgraphExporterInterface(ABC):
    def __init__(self):
        check_parent_field(self, "plugin_ios")
        check_parent_field(self, "_module2name")
        self._name2module = {name: m for m, name in self._module2name.items()}

    def _merge_subgraphs_into_main_graph(
        self,
        case_paths: List[str],
        subgraph_paths: Dict[int, Dict[str, str]],
    ) -> None:
        """Inline exported plugin subgraphs into each case graph, then fuse main-graph norms.

        ``subgraph_paths[case_idx]`` maps ``torchlib_op_name -> standalone_onnx_path``
        for the placeholders present in ``case_paths[case_idx]``. Norm fusion of the
        subblocks already happened at subgraph export time; here we only fuse the
        RMSNorms that live in the main graph (input_layernorm, post_attention_layernorm, …).
        """
        for case_idx, case_path in enumerate(case_paths):
            mapping = subgraph_paths.get(case_idx, {})
            if mapping:
                logger.info(f"  merging {len(mapping)} SubBlock(s) into {case_path}")
                merge_subgraphs_into_model(
                    case_path=case_path,
                    torchlib_op_to_submodule_path=mapping,
                    domain=ONNX_DOMAIN_NAME,
                )
            else:
                logger.warning(f"No Attention SubBlocks to merge for case {case_idx + 1}.")

            # Main-graph RMSNorm fusion — detached from merge_subgraphs_into_model
            # so it is applied explicitly regardless of subblock merging.
            model = onnx.load(case_path, load_external_data=True)
            model, n_fused = fuse_rms_norm(model)
            onnx.save(model, case_path)
            if n_fused:
                logger.info(f"  fused {n_fused} main-graph RMSNorm(s) in {case_path}")

    def _export_plugin_subgraphs(self, opset_version: int = 25) -> Dict[int, Dict[str, str]]:
        """
        Export plugin modules(subgraphs) to ONNX ...
        e.g.
            `Qwen3Attention____model__layers__0__self_attn____case1`
            `Qwen3Attention____model__layers__0__self_attn____case2`
            `node_qwen3rotary_embedding____model__rotary_emb____case1`
            ...
        """
        if len(self.plugin_ios) == 0:
            raise RuntimeError(
                "`self.plugin_ios` is empty, Nothing to export. Make sure `self.plugin_suffix` is not empty and call `self._trace_plugin_ios(model_inputs, **kwargs)` in advance."
            )
        subgraph_paths: Dict[int, Dict[str, str]] = defaultdict(dict)
        for plugin_name, iospec in self.plugin_ios.items():
            # collect unique input cases
            uniqe_input_specs: List[INPUT_SPECS_TYPE] = [spec for spec, _ in iospec.unique_ios()]

            # find specific plugin module to trace
            cls_name, module_name = plugin_name.split("::", 1)
            module = self._name2module[module_name]

            for case_idx, input_specs in enumerate(uniqe_input_specs):
                torchlib_op_name = create_torchlib_op_name(cls_name, module_name, case_idx + 1)
                onnx_path = f"{torchlib_op_name}.onnx"
                logger.debug(f"Export: {plugin_name=} → {onnx_path}")
                with suppress_onnx_export_logs():
                    if "Attention" in cls_name:
                        attn_export(module, input_specs, onnx_path, opset_version)
                    elif "RotaryEmbedding" in cls_name:
                        # TODO: standalone RotaryEmbedding subgraph export not yet implemented;
                        # placeholders are left in the main graph (unmerged) for now.
                        continue
                    else:
                        raise RuntimeError(f"`{cls_name=}` not recognized")
                subgraph_paths[case_idx][torchlib_op_name] = onnx_path
        return subgraph_paths


__all__ = ["SubgraphExporterInterface"]
