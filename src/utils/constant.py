from typing import Tuple, Any, Dict
import torch

# register.py
CUSTOM_LIB_NAME = "hf_module2plugin"
CUSTOM_LIB = torch.library.Library(CUSTOM_LIB_NAME, "DEF")
ONNX_DOMAIN_NAME = "com.jerry"

# tensor_metadata.py
_CACHE_PARAMS: frozenset = frozenset({"past_key_values"})
INPUT_SPECS_TYPE = Dict[str, "TensorSpec | Any"]
OUTPUT_SPECS_TYPE = Tuple[Any, ...]
