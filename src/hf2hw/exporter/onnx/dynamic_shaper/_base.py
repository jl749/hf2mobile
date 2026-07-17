from abc import ABC, abstractmethod
from os import PathLike
from typing import List

from hf2hw.constant import SUBGRAPH_MAP_TYPE


class _ONNXShaper(ABC):
    @abstractmethod
    def _post_process_final_onnx(self, case_idx: int, onnx_path: str | PathLike):
        """Postprocess method that optimizes the final merged ONNX graph"""
        pass

    @abstractmethod
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


__all__ = ["_ONNXShaper"]
