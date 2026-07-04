from typing import Any, Dict, Tuple

import torch

# register.py
CUSTOM_LIB_NAME = "hf_module2plugin"
CUSTOM_LIB = torch.library.Library(CUSTOM_LIB_NAME, "DEF")
ONNX_DOMAIN_NAME = "com.jerry"

# tensor_metadata.py
KV_CACHE_PARAM_NAME = "past_key_values"
_NON_HASHABLE_PARAMS: frozenset = frozenset({KV_CACHE_PARAM_NAME, "attention_mask", "cache_position"})
INPUT_KWARGS_TYPE = Dict[str, Any]
INPUT_SPECS_TYPE = Dict[str, "TensorSpec | Any"]
OUTPUT_SPECS_TYPE = Tuple[Any, ...]

# subgraph.py
SUBGRAPH_MAP_TYPE = Dict[int, Dict[str, str]]
