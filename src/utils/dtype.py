from typing import Union
import torch

_STR_TO_DTYPE = {
    # Floats
    "float32": torch.float32, "float": torch.float32, "torch.float32": torch.float32,
    "float16": torch.float16, "half": torch.float16, "torch.float16": torch.float16,
    "bfloat16": torch.bfloat16, "torch.bfloat16": torch.bfloat16,
    "float64": torch.float64, "double": torch.float64, "torch.float64": torch.float64,
    # Integers
    "int32": torch.int32, "int": torch.int32, "torch.int32": torch.int32,
    "int64": torch.int64, "long": torch.int64, "torch.int64": torch.int64,
    "int16": torch.int16, "short": torch.int16, "torch.int16": torch.int16,
    "int8": torch.int8, "torch.int8": torch.int8,
    "uint8": torch.uint8, "torch.uint8": torch.uint8,
    # Booleans
    "bool": torch.bool, "torch.bool": torch.bool,
}


def convert_dtype(inp: Union[torch.Tensor, torch.dtype, str]) -> Union[str, torch.dtype]:
    """
    Bidirectional dtype converter.
    - If given a tensor or torch.dtype -> returns a clean string (e.g., 'float32').
    - If given a string -> returns the corresponding torch.dtype object.
    """
    # Case 1: Input is a Tensor (extract its dtype first)
    if isinstance(inp, torch.Tensor):
        inp = inp.dtype
        
    # Case 2: Input is a torch.dtype -> Convert to clean String
    if isinstance(inp, torch.dtype):
        return str(inp).replace("torch.", "")

    # Case 3: Input is a String -> Convert to torch.dtype
    if isinstance(inp, str):
        clean_str = inp.strip().lower()
        if clean_str in _STR_TO_DTYPE:
            return _STR_TO_DTYPE[clean_str]
        raise ValueError(f"Unknown dtype string wrapper: '{inp}'")

    raise TypeError(f"Unsupported input type: {type(inp)}")


__all__ = ["convert_dtype"]
