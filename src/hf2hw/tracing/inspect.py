import inspect
import types
import typing
from typing import List, Any
from dataclasses import dataclass

import torch

from .tensor_metadata import TensorSpec, INPUT_SPECS_TYPE


@dataclass
class FwdSpec:
    """
    Encode fwd signatures for tracing
    Args:
        name: fwd parameter name
        kind: "tensor" | "optional_tensor" | "tuple_tensor" | "unknown"
        count: number of flat tensors (>1 only for tuple_tensor)
    """
    name: str
    kind: str
    count: int = 1


def apply_input_specs2fwd_specs(fwd_specs: List[FwdSpec], input_specs: INPUT_SPECS_TYPE) -> List[FwdSpec]:
    """
    Initial `fwd_specs` only contains type hint inferred parameter info. It is not aware of the actual input values.
    We create a new `fwd_specs` based on the `input_specs` variable which represents the actual inference input metadata.
        - update the "unknown" `FwdSpec` kinds by inspecting the input `TensorSpec`s
        - drop the unused params from `fwd_specs` based on `input_specs` observation
    Args:
        fwd_specs: list of the `FwdSpec`s collected by inspecting the forward signatures
        input_specs: list of the `TensorSpec`s containing the input activation info
    Returns:
        new `fwd_specs` now covering the specific `input_specs` case
    """
    updated_fwd_specs: List[FwdSpec] = []
    for fs in fwd_specs:
        spec = input_specs.get(fs.name, None)
        if spec is None:
            continue  # mismatch between `fwd_specs` and `input_specs` -> SKIP
        if fs.kind == "unknown":
            if isinstance(spec, TensorSpec):
                if spec.is_empty:
                    f_spec = FwdSpec(name=fs.name, kind="optional_tensor", count=1)
                else:
                    f_spec = FwdSpec(name=fs.name, kind="tensor", count=1)
            else:
                leaves, _ = torch.utils._pytree.tree_flatten(spec)
                flat_specs = [ts for ts in leaves if isinstance(ts, TensorSpec) and not ts.is_empty]
                if len(flat_specs) == 0:
                    f_spec = FwdSpec(name=fs.name, kind="optional_tensor", count=1)
                else:
                    f_spec = FwdSpec(name=fs.name, kind="tuple_tensor", count=len(flat_specs))
            updated_fwd_specs.append(f_spec)
        else:
            updated_fwd_specs.append(FwdSpec(name=fs.name, kind=fs.kind, count=fs.count))
    return updated_fwd_specs


def fwdspecs2kwargs(fwd_specs: List[FwdSpec], flat: list) -> dict:
    """
    Reconstruct a kwargs dict from the flat op input list
    Args:
        fwd_specs: list of FwdSpec representing function input sig
        flat: function inputs in args format
    Returns:
        reconstructed kwargs in dict
    """
    kwargs = {}
    idx = 0
    for fs in fwd_specs:
        if fs.kind in ("tensor", "optional_tensor"):
            kwargs[fs.name] = flat[idx]
            idx += 1
        elif fs.kind == "tuple_tensor":
            kwargs[fs.name] = tuple(flat[idx:idx + fs.count])
            idx += fs.count
    return kwargs


def fwdspecs2args(fwd_specs: List[FwdSpec], input_dict: dict) -> list:
    """Flatten `input_dict` in a way that it respects `FwdSpec.kind`"""
    flat = []
    for fs in fwd_specs:
        value = input_dict.get(fs.name, None)
        if fs.kind in ("tensor", "optional_tensor"):
            flat.append(value)
        elif fs.kind == "tuple_tensor":
            flat.extend(value)
    return flat


def _classify_ann(ann: type | Any) -> tuple | None:
    """
    Inspect `ann` and return tuple(kind, count)
        whatever: torch.Tensor --> ("tensor", 1)
        whatever: torch.Tensor | None --> ("optional_tensor", 1)
        whatever: Union[torch.Tensor, None] --> ("optional_tensor", 1)
        whatever: Tuple[torch.Tensor, {repeat Tensor N times}] --> ("tuple_tensor", 1)
    """
    if isinstance(ann, type) and issubclass(ann, torch.Tensor):
        return ("tensor", 1)

    origin = typing.get_origin(ann)  # outer type
    args: tuple = typing.get_args(ann)  # innser types

    if (origin is typing.Union or origin is types.UnionType) and type(None) in args and len(args) == 2:
        # type hint is... `{inner} | None` or `Union[{inner}, None]`
        inner = next(a for a in args if a is not type(None))
        inner_cls = _classify_ann(inner)
        if inner_cls and inner_cls[0] == "tensor":
            return ("optional_tensor", 1)
        return None

    if origin is tuple and args and Ellipsis not in args:
        # type hint is ... `Tuple[Tensor x {repeat N times}]`
        if all(isinstance(a, type) and issubclass(a, torch.Tensor) for a in args):
            return ("tuple_tensor", len(args))

    return None


def sig2fwdspecs(sig: inspect.Signature) -> List[FwdSpec]:
    """Based on the passed signature build `List[FwdSpec]`"""
    full_fwdspecs = []
    for name, param in sig.parameters.items():
        if (name == "self") or (param.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)):
            # skip self, *args, **kwargs
            continue

        ann: type | Any = param.annotation

        # NOTE: type hint does not exist
        if ann is inspect.Parameter.empty:
            full_fwdspecs.append(FwdSpec(name=name, kind="unknown"))
            continue

        classified_ann: tuple | None = _classify_ann(ann)

        # NOTE: _classify_ann failed (`Tuple[Tensor, ...]`, `Optional[Cache]`, ...)
        #   keep if "unknown" for now. fix it with `FwdSpec.resolve_unknown` later
        if classified_ann is None:
            full_fwdspecs.append(FwdSpec(name=name, kind="unknown"))
            continue
        full_fwdspecs.append(FwdSpec(name=name, kind=classified_ann[0], count=classified_ann[1]))
    return full_fwdspecs


def sig2num_outputs(sig: inspect.Signature) -> int:
    """Inspect number of the outputs from `sig.return_annotation`"""
    ret = sig.return_annotation
    if ret is inspect.Parameter.empty:
        return 1
    origin = typing.get_origin(ret)
    args = typing.get_args(ret)
    if isinstance(ret, type) and issubclass(ret, torch.Tensor):
        return 1
    if origin is tuple and args:  # TODO: does this work as expected on Attention?
        count = sum(classified_ann[1] for a in args if (classified_ann := _classify_ann(a)))
        return max(count, 1)
    return 1


__all__ = ["FwdSpec", "apply_input_specs2fwd_specs", "fwdspecs2kwargs", "fwdspecs2args", "sig2fwdspecs", "sig2num_outputs"]
