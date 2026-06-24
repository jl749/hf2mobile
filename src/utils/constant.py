from typing import Tuple, Any, Dict

# register.py
CUSTOM_LIB_NAME = "hf_module2plugin"

# tensor_metadata.py
_CACHE_PARAMS: frozenset = frozenset({"past_key_values"})
INPUT_SPECS_TYPE = Dict[str, "TensorSpec | Any"]
OUTPUT_SPECS_TYPE = Tuple[Any, ...]
