import inspect
from contextlib import contextmanager
from os import PathLike
from pathlib import Path

import torch
import transformers

from hf2mobile.constant import INPUT_SPECS_TYPE
from hf2mobile.tracing import (
    apply_input_specs2fwd_specs,
    fwdspecs2args,
    input_specs2pseudo_inputs,
    sig2fwdspecs,
)
from hf2mobile.utils.logger import logger

_ONNX_OUTPUT_NAMES = ("cos", "sin")


@contextmanager
def _adapt_module_for_case(module: torch.nn.Module, input_specs: INPUT_SPECS_TYPE):
    """
    Temporarily overwrite RotaryEmbedding.forward for ONNX tracing.

    RoPE is position-wise, so instead of tracing the "seq-length-specific" cos/sin computation
    we precompute the full `(cos, sin)` tables over `[0, max_position_embeddings)` once,
    bake them as constants, and trace a plain lookup such as `cos = table[position_ids]`.
    """
    orig_cls = module.__class__
    hf_config: transformers.PretrainedConfig = module.config

    x_spec = input_specs.pop("x")  # torch takes (x, position_ids), ONNX required (position_ids,) only
    # NOTE: ONNX RotaryEmbed must satisfy the same dtype for x and cos/sin_cache
    #   `x` is the hidden states, so its observed dtype is the model activation dtype (bf16).
    model_dtype = x_spec.torch_dtype or torch.float32

    input_dict = input_specs2pseudo_inputs(input_specs)  # build fake input
    sig = inspect.signature(orig_cls.forward)
    fwd_specs = sig2fwdspecs(sig)
    case_fwd_specs = apply_input_specs2fwd_specs(fwd_specs, input_specs)
    flat_inputs = fwdspecs2args(case_fwd_specs, input_dict)

    full_position_ids = torch.arange(
        hf_config.max_position_embeddings, dtype=input_dict["position_ids"].dtype
    ).unsqueeze(0)
    fake_x = torch.empty(0, dtype=torch.float32)
    with torch.no_grad():
        cos_table, sin_table = orig_cls.forward(module, fake_x, full_position_ids)
    cos_table = cos_table.contiguous()[0]  # (1, max_len, head_dim) -> (max_len, head_dim)
    sin_table = sin_table.contiguous()[0]

    # NOTE: onnx RotaryEmbedding op does half rotation itself
    #   torch impl takes full head_dim where onnx takes head_dim//2
    #   downcast fp32 tables to the model activation dtype (compute in fp32, store in e.g. bf16)
    head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
    cos_table = cos_table[:, : head_dim // 2].to(model_dtype)
    sin_table = sin_table[:, : head_dim // 2].to(model_dtype)

    def _traceable_forward(self_inner, position_ids: torch.Tensor):
        """ONNX takes `*args` (position_ids) as inputs and `cos_table`, `sin_table` as outputs"""
        cos = torch.nn.functional.embedding(position_ids, cos_table)
        sin = torch.nn.functional.embedding(position_ids, sin_table)

        return cos, sin

    module.__class__ = type(
        f"Traceable{orig_cls.__name__}",
        (orig_cls,),
        {"forward": _traceable_forward},
    )
    try:
        yield flat_inputs, _ONNX_OUTPUT_NAMES
    finally:
        module.__class__ = orig_cls


def export(
    module: torch.nn.Module,
    input_specs: INPUT_SPECS_TYPE,
    onnx_path: str | PathLike,
    opset_version: int,
) -> None:
    onnx_path = Path(onnx_path)
    with _adapt_module_for_case(module, input_specs) as (flat_inputs, output_names):
        torch.onnx.export(
            module,
            args=tuple(flat_inputs),
            kwargs={},
            f=onnx_path,
            opset_version=opset_version,
            output_names=list(output_names),
        )
    logger.debug(f"  exported RotaryEmbedding subgraph: {onnx_path.name}")


__all__ = ["export"]
