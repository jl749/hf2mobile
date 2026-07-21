import functools
from pathlib import Path
from typing import Any, Dict, Tuple

import onnx
import torch

# causallm.py
SUPPORTED_TARGETS = ["ORT", "QNN"]

# register.py
CUSTOM_LIB_NAME = "hf_module2plugin"
CUSTOM_LIB = torch.library.Library(CUSTOM_LIB_NAME, "DEF")
ONNX_DOMAIN_NAME = "com.jerry"
ONNX_TORCHLIB_ATTRIBUTE_NAME = "torchlib_op_name"  # placeholder node attr carrying the registered op name

# tensor_metadata.py
KV_CACHE_PARAM_NAME = "past_key_values"
_NON_HASHABLE_PARAMS: frozenset = frozenset({KV_CACHE_PARAM_NAME, "attention_mask", "cache_position"})
INPUT_KWARGS_TYPE = Dict[str, Any]
INPUT_SPECS_TYPE = Dict[str, "TensorSpec | Any"]
OUTPUT_SPECS_TYPE = Tuple[Any, ...]

# subgraph.py
SUBGRAPH_MAP_TYPE = Dict[int, Dict[str, str]]

# causallm_postprocessor.py
SLIDING_WINDOW_MASK_ONNX = Path(__file__).parent.joinpath("exporter", "onnx", "postprocess", "sliding_window_mask.onnx")


@functools.lru_cache(maxsize=1)
def _swa_onnx_template() -> onnx.ModelProto:
    """`sliding_window_mask.onnx` template graph (cached with Constant weights)."""
    return onnx.load(str(SLIDING_WINDOW_MASK_ONNX))


# group_query_attention.py
_NORM_OPS = ("RMSNormalization", "SimplifiedLayerNormalization", "LayerNormalization")
_SEQLENS_K_NAME = "seqlens_k"
_TOTAL_SEQLEN_NAME = "total_sequence_length"
