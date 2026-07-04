import inspect
from contextlib import contextmanager
from os import PathLike
from pathlib import Path

import onnx
import torch
import transformers

from ...constant import INPUT_SPECS_TYPE, KV_CACHE_PARAM_NAME
from ...tracing import (
    apply_input_specs2fwd_specs,
    fwdspecs2args,
    fwdspecs2kwargs,
    input_specs2pseudo_inputs,
    sig2fwdspecs,
)
from ...utils.logger import logger
from ..onnx import fuse_rms_norm, fuse_rope

_ONNX_OUTPUT_NAME = "attn_output"


@contextmanager
def _adapt_module_for_case(module: torch.nn.Module, input_specs: INPUT_SPECS_TYPE):
    """Temporarily overwrite Attention.forward for ONNX tracing"""
    orig_cls = module.__class__
    layer_idx = getattr(module, "layer_idx", None)
    assert layer_idx is not None, f"Attribute `{orig_cls.__name__}.layer_idx` does not exist (id={id(module)})."

    input_dict = input_specs2pseudo_inputs(input_specs)  # build fake input
    sig = inspect.signature(orig_cls.forward)
    fwd_specs = sig2fwdspecs(sig)
    case_fwd_specs = apply_input_specs2fwd_specs(fwd_specs, input_specs)
    flat_inputs = fwdspecs2args(case_fwd_specs, input_dict)
    graph_input_names = {fs.name for fs in case_fwd_specs}

    # everything else that `forward` needs but that is *not* a graph input
    # e.g. position_ids, cache_position, attention_mask
    extra_kwargs = {k: v for k, v in input_dict.items() if k not in graph_input_names}
    for fs in fwd_specs:
        if fs.name in graph_input_names or fs.name == KV_CACHE_PARAM_NAME:
            # skip graph_input_names and KV_CACHE_PARAM_NAME
            # e.g. "position_embeddings" is already included inside `graph_input_names`
            continue
        if sig.parameters[fs.name].default is inspect.Parameter.empty:
            # forward does not specify default value -> set it None under `extra_kwargs`
            # e.g. Qwen2Attention / Qwen3Attention / Phi3Attention do not specify `attention_mask=None`
            #   1. `ModuleIOSpec.unique_ios()` drops empty `TensorSpec` - "attention_mask"
            #   2. `apply_input_specs2fwd_specs` drops "attention_mask" from `List[FwdSpec]`
            #   3. extra_kwargs["attention_mask"] = None
            extra_kwargs.setdefault(fs.name, None)

    has_kv = KV_CACHE_PARAM_NAME in graph_input_names
    output_names = (_ONNX_OUTPUT_NAME,) + (("new_key", "new_value") if has_kv else ())

    # TODO: factory function returns different forward based on attention type (cross, sliding, moe ... etc)

    def _traceable_forward(self_inner, *args):
        """ONNX takes `*args` as inputs `attn_out`, `(optional){key_out, key_in}` as outputs"""
        kwargs = fwdspecs2kwargs(case_fwd_specs, list(args))
        kwargs.update(extra_kwargs)  # re-inject non ONNX input params
        if has_kv:
            past_key, past_value = kwargs.pop(KV_CACHE_PARAM_NAME)
            _pad = [(torch.empty(0), torch.empty(0))] * layer_idx
            cache = transformers.DynamicCache(ddp_cache_data=_pad + [(past_key, past_value)])
            kwargs[KV_CACHE_PARAM_NAME] = cache

        out = orig_cls.forward(self_inner, **kwargs)

        attn_out = out[0] if isinstance(out, tuple) else out
        if has_kv:
            return attn_out, cache.layers[layer_idx].keys, cache.layers[layer_idx].values
        return attn_out

    module.__class__ = type(
        f"Traceable{orig_cls.__name__}",
        (orig_cls,),
        {"forward": _traceable_forward},
    )
    try:
        yield flat_inputs, output_names
    finally:
        module.__class__ = orig_cls


def export(
    module: torch.nn.Module,
    input_specs: INPUT_SPECS_TYPE,
    onnx_path: str | PathLike,
    opset_version: int,
) -> None:
    onnx_path: Path = Path(onnx_path)
    with _adapt_module_for_case(module, input_specs) as (flat_inputs, output_names):
        torch.onnx.export(
            module,
            args=tuple(flat_inputs),
            kwargs={},
            f=onnx_path,
            opset_version=opset_version,
            output_names=output_names,
        )
    _m = onnx.load(str(onnx_path), load_external_data=True)
    _m, _n_rms = fuse_rms_norm(_m)
    _m, _n_rope = fuse_rope(_m)
    onnx.save(_m, str(onnx_path))
    if _n_rms:
        logger.debug(f"  fused {_n_rms} RMSNorm(s) in {onnx_path.name}")
    if _n_rope:
        logger.debug(f"  fused {_n_rope} RotaryEmbedding(s) in {onnx_path.name}")


__all__ = ["export"]
