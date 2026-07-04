from abc import ABC
from collections import defaultdict
from typing import List

from hf2hw.constant import INPUT_KWARGS_TYPE
from hf2hw.utils import check_parent_field, suppress_onnx_export_logs
from hf2hw.utils.logger import logger

from .attention import export as attn_export


class SubgraphExporterInterface(ABC):
    def __init__(self):
        check_parent_field(self, "plugin_ios")
        check_parent_field(self, "_module2name")
        self._name2module = {name: m for m, name in self._module2name.items()}

    def _merge_subgraphs_into_main_graph(self, case_paths: List[str], subgraph_paths: Dict[str, Dict[str, str]]):
        for case_idx, case_path in enumerate(case_paths):
            mapping = subgraph_paths[case_idx]
            if not mapping:
                logger.warning(f"No Attention SubBlocks to merge for case {case_idx + 1}; skipping.")
                continue
            logger.info(f"  merging {len(mapping)} SubBlock(s) into {case_path}")
            merge_subgraphs_into_model()  # TODO: resolve

    def _export_plugin_subgraphs(self, opset_version=25) -> Dict[str, Dict[str, str]]:
        subgraph_paths = defaultdict(dict)
        for plugin_name, iospec in self.plugin_ios:
            # collect unique input cases
            uniq_input_dicts: List[INPUT_KWARGS_TYPE] = iospec.pseudo_unique_inputs()

            # find specific plugin module to trace
            cls_name, module_name = plugin_name.split("::", 1)
            module = self._name2module[module_name]

            for case_idx, input_dict in enumerate(uniq_input_dicts):
                onnx_path = f"{cls_name}____{module_name.replace('.', '__')}____case{case_idx}.onnx"
                logger.debug(f"Export: {plugin_name=} → {onnx_path}")
                with suppress_onnx_export_logs():
                    if "Attention" in cls_name:
                        attn_export(module, input_dict, onnx_path, opset_version)
                    elif "RotaryEmbedding" in cls_name:
                        ...
                    else:
                        raise RuntimeError(f"`{cls_name=}` not recognized")
                    subgraph_paths[case_idx][plugin_name] = str(out_path)
        return subgraph_paths


__all__ = ["SubgraphExporterInterface"]
