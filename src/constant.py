from typing import Tuple, Any, Dict
import torch

# register.py
CUSTOM_LIB_NAME = "hf_module2plugin"
CUSTOM_LIB = torch.library.Library(CUSTOM_LIB_NAME, "DEF")
ONNX_DOMAIN_NAME = "com.jerry"

# tensor_metadata.py
KV_CACHE_PARAM_NAME = "past_key_values"
_NON_HASHABLE_PARAMS: frozenset = frozenset({KV_CACHE_PARAM_NAME,})
INPUT_SPECS_TYPE = Dict[str, "TensorSpec | Any"]
OUTPUT_SPECS_TYPE = Tuple[Any, ...]
