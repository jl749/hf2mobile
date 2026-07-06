from abc import ABC, abstractmethod
from collections import defaultdict
from os import PathLike
from typing import List

from hf2hw.constant import INPUT_SPECS_TYPE, ONNX_DOMAIN_NAME, SUBGRAPH_MAP_TYPE
from hf2hw.exporter.onnx import merge_subgraphs_into_model
from hf2hw.utils import check_parent_field, create_torchlib_op_name, suppress_onnx_export_logs
from hf2hw.utils.logger import logger

from .attention import export as attn_export


class SubgraphExporterInterface(ABC):
    def __init__(self):
        check_parent_field(self, "plugin_ios")
        check_parent_field(self, "_module2name")
        self._name2module = {name: m for m, name in self._module2name.items()}

    @abstractmethod
    def _post_process_final_onnx(self, onnx_path: str | PathLike):
        """Postprocess method that optimizes the final merged ONNX graph"""
        pass

    def merge_subgraphs_into_main_graph(
        self,
        case_paths: List[str],
        subgraph_map: SUBGRAPH_MAP_TYPE,
    ) -> None:
        """
        Merge the ONNX subgraphs into the main ONNX graphs

        Args:
            case_paths: main ONNX graph paths representing unique input cases
            subgraph_map: `subgraph_map[case_idx]` maps torchlib_op_name -> subgraph onnx path
        """
        for case_idx, case_path in enumerate(case_paths):
            mapping = subgraph_map.get(case_idx, {})
            if mapping:
                logger.info(f"  merging {len(mapping)} Subgraph(s) into {case_path}")
                merge_subgraphs_into_model(
                    case_path=case_path,
                    torchlib_op2subgraph_path=mapping,
                    domain=ONNX_DOMAIN_NAME,
                )
            else:
                logger.warning(f"No plugin subgraph to merge for case {case_idx + 1}.")

            self._post_process_final_onnx(case_path)

    def export_plugin_subgraphs(self, opset_version: int = 25) -> SUBGRAPH_MAP_TYPE:
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
                "`self.plugin_ios` is empty, Nothing to export. Make sure `self.plugin_suffix` is not empty and call `self.trace_plugin_ios(model_inputs, **kwargs)` in advance."
            )
        subgraph_map: SUBGRAPH_MAP_TYPE = defaultdict(dict)
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
                subgraph_map[case_idx][torchlib_op_name] = onnx_path
        return subgraph_map


__all__ = ["SubgraphExporterInterface"]
