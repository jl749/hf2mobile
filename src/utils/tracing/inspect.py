import inspect
import types
import typing
from typing import List, Any
from dataclasses import dataclass

import torch

from utils.tracing.tensor_metadata import INPUT_SPECS_TYPE


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

    def resolve_unknown(self, input_specs: INPUT_SPECS_TYPE) -> "FwdSpec":
        """Resolve unknown FwdSpec using `input_specs` (actual observed tensor)"""
        if self.kind == "unknown":
            input_spec = input_specs.get(self.name, None)
            # TODO: update FwdSpec based on input_specs
        return self

def fwdspecs2kwargs(fwdspecs: List[FwdSpec], flat: list) -> dict:
    """
    Reconstruct a kwargs dict from the flat op input list (inverse of _flatten_call_args).
    Args:
        fwdspecs: list of FwdSpec representing function input sig
        flat: function inputs in args format
    Returns:
        reconstructed kwargs in dict
    """
    kwargs = {}
    idx = 0
    for fs in fwdspecs:
        if fs.kind in ("tensor", "optional_tensor"):
            kwargs[fs.name] = flat[idx]
            idx += 1
        elif fs.kind == "tuple_tensor":
            kwargs[fs.name] = tuple(flat[idx:idx + fs.count])
            idx += fs.count
    return kwargs

# TODO: change name to fwdspecs2args
def _flatten_call_args(fwdspecs: List[FwdSpec], bound_args: dict) -> list:
    """Expand bound_args into a flat list matching the op schema."""
    flat = []
    for fs in fwdspecs:
        val = bound_args.get(fs.name)
        if fs.kind in ("tensor", "optional_tensor"):
            flat.append(val)
        elif fs.kind == "tuple_tensor":
            flat.extend(val)
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
    specs = []
    for name, param in sig.parameters.items():
        if (name == "self") or (param.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)):
            # skip self, *args, **kwargs
            continue

        ann: type | Any = param.annotation

        # NOTE: type hint does not exist
        if ann is inspect.Parameter.empty:
            specs.append(FwdSpec(name=name, kind="unknown"))
            continue

        classified_ann: tuple | None= _classify_ann(ann)

        # NOTE: type hint with elipsis e.g. Tuple[Tensor, ...]
        if classified_ann is None:
            _inner_args = typing.get_args(ann)
            if (
                typing.get_origin(ann) is tuple
                and len(_inner_args) == 2
                and _inner_args[1] is Ellipsis
                and isinstance(_inner_args[0], type)
                and issubclass(_inner_args[0], torch.Tensor)
            ):
                specs.append(FwdSpec(name=name, kind="unknown"))
            continue
        specs.append(FwdSpec(name=name, kind=classified_ann[0], count=classified_ann[1]))
    return specs


def sig2num_outputs(sig: inspect.Signature) -> int:
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


__all__ = ["FwdSpec", "fwdspecs2kwargs", "_flatten_call_args", "sig2fwdspecs", "sig2num_outputs"]
