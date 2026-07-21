import logging
import os
import shutil
from abc import ABC, abstractmethod
from os import PathLike
from typing import Dict, List

import onnx

from hf2hw.constant import ONNX_DOMAIN_NAME, ONNX_TORCHLIB_ATTRIBUTE_NAME, SUBGRAPH_MAP_TYPE
from hf2hw.utils.logger import logger
from hf2hw.utils.onnx_helper import drop_attributes, onnx_to_function, save_onnx, update_opset


def _merge_subgraphs_into_model(
    case_path: str,
    torchlib_op2subgraph_path: Dict[str, str],
    domain: str = ONNX_DOMAIN_NAME,
    output_path: str | None = None,
) -> str:
    """
    Replace custom plugin nodes under the main ONNX graph (`case_path`) using `torchlib_op2subgraph_path`.
    Each subgraph will be represented in FunctionProto.
    `subgraph_path = torchlib_op2subgraph_path[plugin_node.attribute["torchlib_op_name"]]`

    Args:
        case_path: existing `case{i}.onnx` to patch
        torchlib_op2subgraph_path: `{torchlib_op_name: standalone_onnx_path}`
        domain: ONNX domain for both placeholder nodes and emitted functions
        output_path: save path if provided. if None -> overwrite `case_path`
    Returns:
        merged onnx path
    """
    output_path = output_path or case_path
    model: onnx.ModelProto = onnx.load(case_path, load_external_data=True)

    rewritten = 0
    appended_fn_names: List[str] = []
    seen_fn_names = {(f.domain, f.name) for f in model.functions}

    for node in model.graph.node:
        if node.domain != domain:
            continue
        _attr = next((a for a in node.attribute if a.name == ONNX_TORCHLIB_ATTRIBUTE_NAME), None)
        if _attr is None:
            continue

        # NOTE: from this point => `node = {...torchlib registered plugin node...}`
        torchlib_op_name = _attr.s.decode() if isinstance(_attr.s, (bytes, bytearray)) else str(_attr.s)

        subgraph_path = torchlib_op2subgraph_path.get(torchlib_op_name, None)
        if subgraph_path is None:
            logger.warning(
                f"merge_subgraphs_into_model: no standalone ONNX provided for "
                f"{torchlib_op_name!r}; leaving placeholder in place."
            )
            continue

        # build and register the function
        if (domain, torchlib_op_name) not in seen_fn_names:
            func: onnx.FunctionProto = onnx_to_function(subgraph_path, function_name=torchlib_op_name, domain=domain)
            logger.debug(
                f"onnx_to_function: {os.path.relpath(subgraph_path)} → fn {torchlib_op_name!r} "
                f"{len(func.node)} node(s), "
                f"{len(func.input)} in / {len(func.output)} out)"
            )
            for op in func.opset_import:
                update_opset(model, op.domain, op.version)
            model.functions.append(func)
            seen_fn_names.add((domain, torchlib_op_name))
            appended_fn_names.append(torchlib_op_name)

        # validate input/output arity before mutating the node.
        func = next(f for f in model.functions if f.name == torchlib_op_name and f.domain == domain)
        if len(node.input) != len(func.input):
            raise ValueError(
                f"Input count mismatch for {torchlib_op_name!r}: "
                f"placeholder has {len(node.input)} inputs, function has {len(func.input)}."
            )
        if len(node.output) != len(func.output):
            raise ValueError(
                f"Output count mismatch for {torchlib_op_name!r}: "
                f"placeholder has {len(node.output)} outputs, function has {len(func.output)}."
            )

        # rewrite placeholder into a function call (domain stays the same).
        node.op_type = torchlib_op_name
        # drop placeholder-only attributes so the node is a clean function call.
        drop_attributes(node, names_to_drop={ONNX_TORCHLIB_ATTRIBUTE_NAME})  # TODO: is this step required?
        rewritten += 1

    save_onnx(model, output_path)
    logger.info(
        f"merge_subgraphs_into_model: {case_path} → {output_path} "
        f"({rewritten} node(s) rewritten, {len(appended_fn_names)} function(s) appended)"
    )
    return output_path


class _ONNXPostprocessor(ABC):
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
                _merge_subgraphs_into_model(
                    case_path=case_path,
                    torchlib_op2subgraph_path=mapping,
                    domain=ONNX_DOMAIN_NAME,
                )
            else:
                logger.warning(f"No plugin subgraph to merge for case {case_idx + 1}.")

            self._post_process_final_onnx(case_idx, case_path)

        # clean up subgraph onnx
        if (logger.isEnabledFor(logging.DEBUG) is False) and (self._subgraph_dir.exists()):
            shutil.rmtree(self._subgraph_dir)

    @abstractmethod
    def _post_process_final_onnx(self, case_idx: int, onnx_path: str | PathLike):
        """Postprocess method that optimizes the final merged ONNX graph"""
        pass

    @abstractmethod
    def make_dynamic_onnx(
        self,
        onnx_path: str | PathLike,
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
