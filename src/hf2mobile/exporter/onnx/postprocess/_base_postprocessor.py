import logging
import os
import shutil
from abc import ABC, abstractmethod
from typing import Dict, List

import onnx_ir as ir

from hf2mobile.constant import ONNX_DOMAIN_NAME, ONNX_TORCHLIB_ATTRIBUTE_NAME, SUBGRAPH_MAP_TYPE
from hf2mobile.utils.logger import logger
from hf2mobile.utils.onnx_helper import drop_attributes, graph_to_function, load_onnx_ir, update_opset


def _merge_subgraphs_into_model(
    model_ir: ir.Model,
    torchlib_op2subgraph_path: Dict[str, str],
    domain: str = ONNX_DOMAIN_NAME,
) -> int:
    """
    Replace custom plugin nodes in `model_ir` using `torchlib_op2subgraph_path`.
    Each subgraph replaces the custom plugin nodes(torchlib registered op) and becomes the new `ir.Function`.
    `subgraph_path = torchlib_op2subgraph_path[plugin_node.attributes["torchlib_op_name"]]`

    Args:
        model_ir: main model (merge happens in-place)
        torchlib_op2subgraph_path: `{torchlib_op_name: standalone_onnx_path}`
        domain: ONNX domain for both placeholder nodes and emitted functions
    Returns:
        number of placeholder nodes rewritten
    """
    rewritten = 0
    appended_fn_names: List[str] = []

    for node in model_ir.graph:
        # NOTE: custom registered torchlib_ops are defined with specific domain name (e.g. com.hf2mobile)
        #   this helps determine which nodes to expand (i.e. where to merge the subgraph_path)
        if node.domain != domain:
            continue
        _attr = node.attributes.get(ONNX_TORCHLIB_ATTRIBUTE_NAME)
        if _attr is None:
            continue

        # NOTE: from this point => `node = {...torchlib registered plugin node...}`
        torchlib_op_name = _attr.value
        if isinstance(torchlib_op_name, (bytes, bytearray)):
            torchlib_op_name = torchlib_op_name.decode()

        subgraph_path = torchlib_op2subgraph_path.get(torchlib_op_name, None)
        if subgraph_path is None:
            logger.warning(
                f"merge_subgraphs_into_model: no standalone ONNX subgraph provided for "
                f"{torchlib_op_name!r}; leaving placeholder in place."
            )
            continue

        # build and register the function
        func_identifier = (domain, torchlib_op_name, "")  # (domain, name, overload)
        if func_identifier not in model_ir.functions:
            func: ir.Function = graph_to_function(subgraph_path, function_name=torchlib_op_name, domain=domain)
            logger.debug(
                f"graph_to_function: {os.path.relpath(subgraph_path)} → fn {torchlib_op_name!r} "
                f"{len(func)} node(s), "
                f"{len(func.inputs)} in / {len(func.outputs)} out)"
            )
            for op_domain, version in func.opset_imports.items():
                update_opset(model_ir, op_domain, version)
            model_ir.functions[func_identifier] = func
            appended_fn_names.append(torchlib_op_name)

        # validate input/output arity before mutating the node.
        func = model_ir.functions[func_identifier]
        if len(node.inputs) != len(func.inputs):
            raise ValueError(
                f"Input count mismatch for {torchlib_op_name!r}: "
                f"subgraph has {len(node.inputs)} inputs, function has {len(func.inputs)}."
            )
        if len(node.outputs) != len(func.outputs):
            raise ValueError(
                f"Output count mismatch for {torchlib_op_name!r}: "
                f"subgraph has {len(node.outputs)} outputs, function has {len(func.outputs)}."
            )

        # rewrite placeholder into a function call (domain stays the same).
        node.op_type = torchlib_op_name
        # drop placeholder-only attributes so the node is a clean function call.
        drop_attributes(node, names_to_drop={ONNX_TORCHLIB_ATTRIBUTE_NAME})  # TODO: is this step required?
        rewritten += 1

    logger.info(
        f"merge_subgraphs_into_model: {rewritten} node(s) rewritten, {len(appended_fn_names)} function(s) appended"
    )
    return rewritten


class _ONNXPostprocessor(ABC):
    def merge_subgraphs_into_main_graph(
        self,
        case_paths: List[str],
        subgraph_map: SUBGRAPH_MAP_TYPE,
    ) -> None:
        """
        Merge the ONNX subgraphs into the main ONNX graph

        Args:
            case_paths: main ONNX graph paths representing unique input cases
            subgraph_map: `subgraph_map[case_idx]` maps torchlib_op_name -> subgraph onnx path
        """
        for case_idx, case_path in enumerate(case_paths):
            torchlib_op2subgraph_path = subgraph_map.get(case_idx, {})
            model_ir = load_onnx_ir(case_path)
            if torchlib_op2subgraph_path:
                logger.info(f"  merging {len(torchlib_op2subgraph_path)} Subgraph(s) into {case_path}")
                _merge_subgraphs_into_model(
                    model_ir=model_ir,
                    torchlib_op2subgraph_path=torchlib_op2subgraph_path,
                    domain=ONNX_DOMAIN_NAME,
                )
            else:
                logger.warning(f"No plugin subgraph to merge for case {case_idx + 1}.")

            # NOTE: no save/reload between the merge and the postprocess.
            # * Merged functions store weights as `Constant` nodes, but `ir.save` only externalizes initializers.
            #   A round-trip here would leave `Constant` nodes pointing to temporary subgraph files: `__onnx_subgraphs/*.data`
            #   which are deleted after `_postprocess_final_onnx` (shutil.rmtree).
            # * Henceforth, passing the live model directly to postprocess keeps every mmap valid.
            #   `optimize_onnx` under `_postprocess_final_onnx` subsequently lifts those Constants back to initializers.
            self._postprocess_final_onnx(case_idx, model_ir, case_path)

        # clean up subgraph onnx
        if (logger.isEnabledFor(logging.DEBUG) is False) and (self._subgraph_dir.exists()):
            shutil.rmtree(self._subgraph_dir)

    @abstractmethod
    def _postprocess_final_onnx(self, case_idx: int, model_ir: ir.Model, onnx_path: str) -> None:
        """Postprocess method that optimizes the final merged ONNX graph"""
        pass

    @abstractmethod
    def make_dynamic_onnx(
        self,
        model_ir: ir.Model,
        *,
        allowzero: int = 0,
        seq_sym: str = "L",
        prev_sym: str = "L_prev",
        curr_sym: str = "L_prev+1",
    ) -> None:
        """
        Rewrite the CausalLM ONNX graph so that it can take dynamic seq_len.

        `allowzero=0` writes the seq axis as a `0`-copy (default).
        `allowzero=1` makes the seq axis the inferred `-1` and bakes the head-count / hidden axis explicitly.
        """
        pass


__all__ = ["_ONNXPostprocessor"]
