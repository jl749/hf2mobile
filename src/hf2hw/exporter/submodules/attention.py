from contextlib import contextmanager
from os import PathLike
from pathlib import Path

import onnx
import torch
import transformers

from hf2hw.constant import INPUT_KWARGS_TYPE
from hf2hw.exporter.onnx import fuse_rms_norm, fuse_rope
from hf2hw.utils import update_input_cache
from hf2hw.utils.logger import logger

_ONNX_OUTPUT_NAME = "attn_output"


@contextmanager
def _adapt_module_for_case(module: torch.nn.Module, input_dict: INPUT_KWARGS_TYPE):
    """Temporarily overwrite Attention.forward for ONNX tracing"""
    orig_cls = module.__class__
    layer_idx = getattr(module, "layer_idx", None)
    assert layer_idx is not None, f"Attribute `{orig_cls.__name__}.layer_idx` does not exist (id={id(module)})."
    cache_param_name = update_input_cache(input_dict, layer_idx)

    output_names = (_ONNX_OUTPUT_NAME,)
    if cache_param_name:
        # NOTE: KV caches are enabled (use_cache=True)
        output_names += ("new_key", "new_value")
        past_k, past_v = input_dict.pop(cache_param_name)
        input_dict["past_key"] = past_k  # local key
        input_dict["past_value"] = past_v  # local value

    # TODO: factory function returns different forward based on attention type (cross, sliding, moe ... etc)

    def _traceable_forward(self_inner, past_key, past_value, **kwargs):
        if cache_param_name:
            past_key_values = [(torch.empty(0), torch.empty(0))] * layer_idx
            past_key_values[layer_idx] = (past_key, past_value)
            cache = transformers.DynamicCache(ddp_cache_data=past_key_values)
            kwargs[cache_param_name] = cache
        out = orig_cls.forward(self_inner, **kwargs)
        attn_out = out[0] if isinstance(out, tuple) else out
        if cache_param_name:
            new_key = cache.layers[layer_idx].keys
            new_value = cache.layers[layer_idx].values
            return attn_out, new_key, new_value
        else:
            return attn_out

    module.__class__ = type(
        f"Traceable{orig_cls.__name__}",
        (orig_cls,),
        {"forward": _traceable_forward},
    )
    try:
        yield input_dict, output_names
    finally:
        module.__class__ = orig_cls


def export(
    module: torch.nn.Module,
    input_dict: INPUT_KWARGS_TYPE,
    onnx_path: PathLike,
    opset_version: int,
) -> None:
    onnx_path: Path = Path(onnx_path)
    with _adapt_module_for_case(module, input_dict) as (export_kwargs, output_names):
        torch.onnx.export(
            module,
            args=(),
            kwargs=export_kwargs,
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
