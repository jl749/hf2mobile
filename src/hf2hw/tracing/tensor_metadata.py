from typing import Union, List, Tuple, Any, Dict
from dataclasses import dataclass

import torch
import transformers

from hf2hw.constant import KV_CACHE_PARAM_NAME, _NON_HASHABLE_PARAMS, INPUT_SPECS_TYPE, OUTPUT_SPECS_TYPE


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
    Dataclass storing each param info observed from the forward hook
    Args:
        dtype: observed tensor dtype in normalized string (see `_STR_TO_DTYPE`)
        shape: observed tensor shape
        scalar: in case the observed value is a scalar
    Example:
        >> TensorSpec.from_tensor(value={...IO_param_from_hook...})
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
            module: (optional) sometimes metadata is required in order to encode value to TensorSpec
        Returns:
            `TensorSpec` or `TensorSpec` wrapped around python obj
        """
        if value is None:
            return cls(shape=None, dtype=None)
        elif isinstance(value, (int, float, bool)):
            return cls(dtype=torch.tensor(value).dtype, scalar=value)
        elif isinstance(value, torch.Tensor):
            return cls(shape=value.shape, dtype=value.dtype)
        elif isinstance(value, transformers.utils.ModelOutput):
            # only return tensor objects as an output (e.g. Qwen3ForCausalLM)
            return {k: cls(shape=v.shape, dtype=v.dtype) for k, v in value.items() if isinstance(v, torch.Tensor)}
        elif isinstance(value, transformers.Cache):
            flat_leaves, _ = torch.utils._pytree.tree_flatten(value)
            flat_specs = [
                cls(shape=t.shape, dtype=t.dtype)
                if (isinstance(t, torch.Tensor) and t.numel() > 0)
                else cls(shape=None, dtype=None)
                for t in flat_leaves
            ]
            if module is not None and hasattr(module, "layer_idx"):
                # `Attention`: return only this layer's (K, V) pair
                i = module.layer_idx
                return (flat_specs[2 * i], flat_specs[2 * i + 1])
            else:
                # `Model-level`: return [(k1, v1), (k2, v2), ...] per layer
                return [(flat_specs[2 * i], flat_specs[2 * i + 1]) for i in range(len(flat_specs) // 2)]
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
        

@dataclass
class ModuleIOSpec:
    """
    Collection of `TensorSpec`
    Args:
        cls_name: torch.nn.Module class name
        module_name: torch module name
        obsvd_input_specs: collected input specs during the trace
        obsvd_output_specs: collected output specs during the trace
    """
    cls_name: str
    module_name: str
    obsvd_input_specs: List[INPUT_SPECS_TYPE] = None
    obsvd_output_specs: List[OUTPUT_SPECS_TYPE] = None

    @staticmethod
    def flatten_io_specs(specs: INPUT_SPECS_TYPE | OUTPUT_SPECS_TYPE, skip_empty=False) -> Tuple[List[TensorSpec], torch.utils._pytree.TreeSpec]:
        if skip_empty:
            def is_filterable(v) -> bool:
                # empty TensorSpec leaf, OR container that became empty after filtering
                if hasattr(v, "is_empty") and v.is_empty:
                    return True
                if isinstance(v, (list, tuple, dict)) and len(v) == 0:
                    return True
                return False

            def filter_empty_specs(node):
                """Recursively removes empty specs and collections that become empty."""
                if isinstance(node, dict):
                    out = {}
                    for k, v in node.items():
                        fv = filter_empty_specs(v)
                        if not is_filterable(fv):
                            out[k] = fv
                    return out
                elif isinstance(node, (list, tuple)):
                    out = []
                    for v in node:
                        fv = filter_empty_specs(v)
                        if not is_filterable(fv):
                            out.append(fv)
                    return type(node)(out)
                return node
            specs = filter_empty_specs(specs)
        try:
            leaves, _tree_spec = torch.utils._pytree.tree_flatten(specs)
            flatten_specs = [v for v in leaves]
        except:
            raise ValueError(f"`{specs=}` cannot be flattened with pytree")
        assert all(isinstance(v, TensorSpec) for v in flatten_specs), f"Non TensorSpec object inside {specs=}"
        return flatten_specs, _tree_spec

    @staticmethod
    def _get_hashable_obsvd_specs(obsvd_specs: List[INPUT_SPECS_TYPE] | List[OUTPUT_SPECS_TYPE]) -> tuple:
        hashable_obj = []
        for specs in obsvd_specs:
            flatten_specs, _ = ModuleIOSpec.flatten_io_specs(specs, skip_empty=False)
            hashable_obj.append(tuple(("TensorSpec", ts.dtype, ts.shape, ts.scalar) for ts in flatten_specs))
        return tuple(hashable_obj)

    @property
    def unique_obsvd_input_specs(self) -> List[INPUT_SPECS_TYPE]:
        """Some input param names are ignored when considering the uniquness of the module IO (e.g. past_key_values)"""
        return [{k: v for k, v in s.items() if k not in _NON_HASHABLE_PARAMS} for s in self.obsvd_input_specs]

    @property
    def hashable_obsvd_input_specs(self):
        return self._get_hashable_obsvd_specs(self.obsvd_input_specs),

    @property
    def hashable_obsvd_output_specs(self):
        return self._get_hashable_obsvd_specs(self.obsvd_output_specs),

    def _unique_key(self) -> Tuple[Any, ...]:
        """Canonical comparable key — shared by __hash__ and __eq__."""
        return (
            self.cls_name,
            self.module_name,
            self.hashable_obsvd_input_specs,
            self.hashable_obsvd_output_specs,
        )

    def __hash__(self) -> int:
        return hash(self._unique_key())

    def __eq__(self, other) -> bool:
        if not isinstance(other, ModuleIOSpec):
            return NotImplemented
        return self._unique_key() == other._unique_key()

    def __post_init__(self):
        self.obsvd_input_specs = self.obsvd_input_specs or []
        self.obsvd_output_specs = self.obsvd_output_specs or []

    def unique_ios(self) -> Tuple[Tuple[INPUT_SPECS_TYPE, OUTPUT_SPECS_TYPE], ...]:
        """
        Return unique (input_specs, output_specs) pairs
        When considering the uniqness param names under `_NON_HASHABLE_PARAMS` are ignored (e.g. past_key_values)
        Output tuple will always return the metadata pairs in observation order
        e.g. 
            when `generate` is called on the transformers CausalLM models
            prefill runs first taking index 0 for both `obsvd_input_specs` and `obsvd_output_specs`
            this means `unique_ios` returned tuple also contains 
            prefill profile at idx 0 and generation profile at idx 1
        """
        unique_io_paris = []
        _seen = set()
        for _unq_input_specs, input_specs, output_specs in zip(self.unique_obsvd_input_specs, self.obsvd_input_specs, self.obsvd_output_specs):
            # when considering the uniqness use `self.unique_obsvd_input_specs` instead of `self.obsvd_input_specs` 
            _flat_unq_is, _unq_tree = self.flatten_io_specs(specs=_unq_input_specs, skip_empty=True)
            _unq_input_specs: INPUT_SPECS_TYPE = torch.utils._pytree.tree_unflatten(_flat_unq_is, _unq_tree)

            # when returning the unique IOs use `self.obsvd_input_specs` and `self.obsvd_output_specs`
            flat_i_specs, i_tree = self.flatten_io_specs(specs=input_specs, skip_empty=True)
            flat_o_specs, o_tree = self.flatten_io_specs(specs=output_specs, skip_empty=True)
            input_specs: INPUT_SPECS_TYPE = torch.utils._pytree.tree_unflatten(flat_i_specs, i_tree)
            output_specs: OUTPUT_SPECS_TYPE = torch.utils._pytree.tree_unflatten(flat_o_specs, o_tree)
            _key = (*self._get_hashable_obsvd_specs([_unq_input_specs]), *self._get_hashable_obsvd_specs([output_specs]))
            if _key in _seen:
                continue
            else:
                _seen.add(_key)
                unique_io_paris.append((input_specs, output_specs))
        return tuple(unique_io_paris)

    def build_pseudo_unique_inputs(self) -> List[INPUT_SPECS_TYPE]:
        """
        Materialize each unique input_specs into concrete tensors / scalars.
        Walks the nested structure via `tree_map` and replaces every `TensorSpec` leaf with:
          - `None`                                if it is empty
          - its `.scalar` value                   if it is a scalar
          - `torch.zeros(shape, dtype=...)`       otherwise

        Non-`TensorSpec` leaves (already-materialized values) pass through.
        """
        def _materialize(s):
            if not isinstance(s, TensorSpec):
                return s
            if s.is_empty:
                return None
            if s.is_scalar:
                return s.scalar
            return torch.zeros(s.shape, dtype=s.torch_dtype)

        # build unique pseudo inputs
        input_cases: List[Dict[str, torch.Tesnor | int | float | None]] = [
            torch.utils._pytree.tree_map(_materialize, input_specs)
            for input_specs, _ in self.unique_ios()
        ]

        # replace KV_CACHE_PARAM_NAME entry from `[(k, v), (k, v), ...]` to `DynamicCache`
        for input_dict in input_cases:
            pkv = input_dict.get(KV_CACHE_PARAM_NAME, None)
            if pkv and isinstance(pkv, list) and isinstance(pkv[0], tuple):
                input_dict[KV_CACHE_PARAM_NAME] = transformers.DynamicCache(ddp_cache_data=pkv)
            else:
                input_dict["use_cache"] = False
        return input_cases


def get_kv_specs_from_input_specs(input_specs: INPUT_SPECS_TYPE) -> Tuple[TensorSpec, TensorSpec] | Tuple[()]:
    """Search `KV_CACHE_PARAM_NAME` from the provided `input_specs` and return the K,V TensorSpec pair in len=2 tuple"""
    spec = input_specs.get(KV_CACHE_PARAM_NAME, None)
    if spec:
        if isinstance(spec, TensorSpec) and spec.is_empty:
            return ()
        try:
            k_spec, v_spec = spec
        except:
            raise ValueError(f"Expecting tuple of k and v `TensorSpec`s... `{spec=}`")
        return k_spec, v_spec
    else:
        return ()



__all__ = ["convert_dtype", "TensorSpec", "ModuleIOSpec", "get_kv_specs_from_input_specs"] 
