import functools
from pathlib import Path
from typing import Any, Dict, Tuple

import onnx
import torch

# export.py
DEBUG_CONFIG = {
    "num_hidden_layers": 2,
    "hidden_size": 512,
    "intermediate_size": 256,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "sliding_window": 12,
    "layer_types": ["full_attention", "sliding_attention"],
    # `vocab_size` is deliberately left untouched to avoid out-of-range token ids while tracing.
    # "attn_implementation": "eager",  # TODO: support eager
}

# exporter/causallm.py
SUPPORTED_TARGETS = ["ORT", "QNN"]

# tracing/register.py
CUSTOM_LIB_NAME = "hf_module2plugin"
CUSTOM_LIB = torch.library.Library(CUSTOM_LIB_NAME, "DEF")
ONNX_DOMAIN_NAME = "com.jerry"
ONNX_TORCHLIB_ATTRIBUTE_NAME = "torchlib_op_name"  # placeholder node attr carrying the registered op name

# tracing/tensor_metadata.py
KV_CACHE_PARAM_NAME = "past_key_values"
_NON_HASHABLE_PARAMS: frozenset = frozenset({KV_CACHE_PARAM_NAME, "attention_mask", "cache_position"})
INPUT_KWARGS_TYPE = Dict[str, Any]
INPUT_SPECS_TYPE = Dict[str, "TensorSpec | Any"]
OUTPUT_SPECS_TYPE = Tuple[Any, ...]

# exporter/submodules/subgraph.py
SUBGRAPH_MAP_TYPE = Dict[int, Dict[str, str]]

# causallm_postprocessor.py
SLIDING_WINDOW_MASK_ONNX = Path(__file__).parent.joinpath("exporter", "onnx", "postprocess", "sliding_window_mask.onnx")


@functools.lru_cache(maxsize=1)
def _swa_onnx_template() -> onnx.ModelProto:
    """`sliding_window_mask.onnx` template graph (cached with Constant weights)."""
    return onnx.load(str(SLIDING_WINDOW_MASK_ONNX))


# exporter/onnx/fusion/
_NORM_OPS = ("RMSNormalization", "SimplifiedLayerNormalization", "LayerNormalization")
_SEQLENS_K_NAME = "seqlens_k"
_TOTAL_SEQLEN_NAME = "total_sequence_length"
