from typing import Union, List, Tuple, Any, Dict
from dataclasses import dataclass

import torch
import transformers

from utils.constant import _CACHE_PARAMS, INPUT_SPECS_TYPE, OUTPUT_SPECS_TYPE


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


@dataclass
class TensorSpec:
    """
    Dataclass storing each param info observed from the fwd hook
    Args:
        dtype: observed tensor dtype in normalized string (see `_STR_TO_DTYPE`)
        shape: observed tensor shape
        scalar: in case the observed value is a scalar
    """
    dtype: str | None
    shape: Tuple[int, ...] | None = None
    scalar: int | float | None = None

    @property
    def torch_dtype(self) -> torch.dtype | None:
        if self.dtype:
            return convert_dtype(self.dtype)
        return None

    @property
    def is_scalar(self) -> bool:
        return not (self.scalar is None)

    @property
    def is_empty(self) -> bool:
        return self.dtype is None and self.shape is None and self.scalar is None

    @classmethod
    def from_tensor(cls, value: Any, module=None) -> Any:
        """
        Construct `TensorSpec` from the observed value

        Covers 4 possible cases ...
            - value is None
            - value is Tensor
            - value is Cache (only applicable for the transformers models)
            - value is python obj (list, tuple, dict, ... etc)
        Args:
            value: value passed to the fwd call (fetched by the torch hook)
            module: (optional) used for metadata in order to encode value to TensorSpec
        Returns:
            `TensorSpec` or `TensorSpec` wrapped around python obj
        """
        if value is None:
            return cls(shape=None, dtype=None)
        elif isinstance(value, (int, float, bool)):
            return cls(dtype=torch.tensor(value).dtype, scalar=value)
        elif isinstance(value, torch.Tensor):
            return cls(shape=value.shape, dtype=value.dtype)
        elif isinstance(value, transformers.Cache):
            # NOTE: only works when "Attention" in _cls_name
            _i = getattr(module, "layer_idx")
            try:
                _layer_cache = value.layers[_i]
                k, v = _layer_cache.keys, _layer_cache.values
                if k is not None and v is not None and k.numel() > 0:
                    return (cls(shape=k.shape, dtype=k.dtype), cls(shape=v.shape, dtype=v.dtype))
                else:
                    return cls(shape=None, dtype=None)
            except (IndexError, AttributeError):
                return cls(shape=None, dtype=None)
        else:
            # NOTE: handles Tensors in python native dtypes (list, tuple, nested, ...)
            _flat_leaves, spec = torch.utils._pytree.tree_flatten(value)
            for i, t in enumerate(_flat_leaves):
                if t is None:
                    _flat_leaves[i] = cls(shape=None, dtype=None)
                elif isinstance(t, torch.Tensor):
                    _flat_leaves[i] = cls(shape=t.shape, dtype=t.dtype)
                else:
                    try:
                        t = torch.tensor(t)
                        _flat_leaves[i] = cls(shape=t.shape, dtype=t.dtype)
                    except Exception:
                        # TODO: throw looger warning
                        _flat_leaves[i] = cls(shape=None, dtype=None)
            return torch.utils._pytree.tree_unflatten(_flat_leaves, spec)

    def __post_init__(self):
        if self.is_scalar:
            self.shape = ()  # override None
        else:
            both_none = (self.shape is None) and (self.dtype is None)
            both_valid = self.dtype is not None and isinstance(self.shape, tuple)
            if both_valid:
                self.shape = tuple(self.shape)
                self.dtype = self.dtype if isinstance(self.dtype, str) else convert_dtype(self.dtype)
                assert self.dtype in _STR_TO_DTYPE, f"Invalid dtype `{self.dtype}`."
            elif both_none:
                pass
            else:
                raise ValueError("`TensorSpec` does not allow partially initialized form. Please specify both `shape` and `dtype`.")
        

def _strip_cache_inputs(specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop cache-only keys (e.g., past_key_values) from each captured input dict."""
    return [{k: v for k, v in s.items() if k not in _CACHE_PARAMS} for s in specs]


def _hashable_spec(value: Any) -> Any:
    """Recursively canonicalize an IO spec value into a hashable form"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, TensorSpec):
        shape = value.shape if value.shape is not None else None
        return ("TensorSpec", value.dtype, shape, value.scalar)
    if isinstance(value, dict):
        return tuple(sorted(
            ((k, _hashable_spec(v)) for k, v in value.items()),
            key=lambda kv: kv[0],
        ))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable_spec(v) for v in value)
    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)


@dataclass
class ModuleIOSpec:
    """
    Collection of `TensorSpec`
    Args:
        cls_name: torch.nn.Module class name
        module_name: torch module name
        input_specs: collected input specs
        output_specs: collected output specs
    """
    cls_name: str
    module_name: str
    input_specs: List[INPUT_SPECS_TYPE] = None
    output_specs: List[OUTPUT_SPECS_TYPE] = None

    def _key(self) -> Tuple[Any, ...]:
        """Canonical comparable key — shared by __hash__ and __eq__."""
        return (
            self.cls_name,
            _hashable_spec(_strip_cache_inputs(self.input_specs)),
            _hashable_spec(self.output_specs),
        )

    def __hash__(self) -> int:
        return hash(self._key())

    def __eq__(self, other) -> bool:
        if not isinstance(other, ModuleIOSpec):
            return NotImplemented
        return self._key() == other._key()

    def __post_init__(self):
        self.input_specs = self.input_specs or []
        self.output_specs = self.output_specs or []

    def unique_ios(self) -> Tuple[Tuple[INPUT_SPECS_TYPE, OUTPUT_SPECS_TYPE], ...]:
        """Return unique (input_spec, output_spec) pairs sorted by query seq_len desc."""
        def _drop_empty_dict(d):
            return {k: v for k, v in d.items() if not (isinstance(v, TensorSpec) and v.is_empty)}
        def _drop_empty_tuple(t):
            return tuple(v for v in t if not (isinstance(v, TensorSpec) and v.is_empty))
        _seen = set()
        out: List[Tuple[INPUT_SPECS_TYPE, OUTPUT_SPECS_TYPE]] = []
        for inp, outp in zip(self.input_specs, self.output_specs):
            inp = _drop_empty_dict(inp)
            outp = _drop_empty_tuple(outp)
            inp_filtered = {k: v for k, v in inp.items() if k not in _CACHE_PARAMS}
            _key = (_hashable_spec(inp_filtered), _hashable_spec(outp))
            if _key not in _seen:
                _seen.add(_key)
                out.append((inp, outp))
        return tuple(out)


__all__ = ["convert_dtype", "TensorSpec", "ModuleIOSpec"] 
