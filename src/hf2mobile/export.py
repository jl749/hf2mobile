import argparse
import logging
import os
from typing import Literal

import torch
import transformers

from .constant import DEBUG_CONFIG, SUPPORTED_TARGETS
from .exporter import (
    Gemma3ForCausalLMExporter,
    LlamaForCausalLMExporter,
    Qwen2ForCausalLMExporter,
    Qwen3ForCausalLMExporter,
)
from .utils.logger import logger

EXPORT_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def export(
    model_id: str,
    target: Literal["ORT", "QNN"],
    export_dtype: str | None = None,
    debug: bool = False,
):
    assert (
        target in SUPPORTED_TARGETS
    ), f"Unsupported `{target=}`. Currently supported targets are: `{SUPPORTED_TARGETS}`."

    if debug:
        logger.setLevel(logging.DEBUG)

    logger.info(f"export: loading `{model_id}` (target={target}, export_dtype={export_dtype}, debug={debug})")

    config = transformers.AutoConfig.from_pretrained(model_id)
    if debug:
        logger.debug(f"export: forcing tiny model config: {DEBUG_CONFIG}")
        for key, value in DEBUG_CONFIG.items():
            setattr(config, key, value)

    dtype = EXPORT_DTYPES[export_dtype] if export_dtype else (config.dtype or torch.get_default_dtype())
    logger.info(f"export: casting weights to {dtype}")

    model = (
        transformers.AutoModelForCausalLM.from_pretrained(model_id, config=config, ignore_mismatched_sizes=debug)
        .cpu()
        .to(dtype)
    )
    model.eval()

    architecture = model.config.architectures[0]
    logger.info(f"export: resolved architecture `{architecture}`")

    if architecture == "LlamaForCausalLM":
        exporter = LlamaForCausalLMExporter(model=model)
    elif architecture == "Qwen3ForCausalLM":
        exporter = Qwen3ForCausalLMExporter(model=model)
    elif architecture == "Qwen2ForCausalLM":
        exporter = Qwen2ForCausalLMExporter(model=model)
    elif architecture == "Gemma3ForCausalLM":
        exporter = Gemma3ForCausalLMExporter(model=model)
    else:
        raise ValueError(f"{model_id=} is not a supported architecture `{architecture}`.")

    exported_onnx_paths = exporter.export(
        path_template="case{i}.onnx",
        opset_version=25,
        target=target,
    )
    relative_onnx_paths = [os.path.relpath(p) for p in exported_onnx_paths]
    logger.info(
        f"export: done — `{model_id}` → ONNX "
        f"(target={target}, export_dtype={export_dtype}); exported_onnx_paths={relative_onnx_paths}"
    )


def main():
    parser = argparse.ArgumentParser(
        prog="python -m hf2mobile.export",
        description="Export a Hugging Face causal LM to a hardware-targeted ONNX model.",
    )
    parser.add_argument(
        "repo_id",
        help="Hugging Face repo id of the model to export (e.g. `Qwen/Qwen2.5-0.5B-Instruct`).",
    )
    parser.add_argument(
        "-t",
        "--target",
        default="ORT",
        choices=SUPPORTED_TARGETS,
        help="Export target (default: %(default)s).",
    )
    parser.add_argument(
        "--export_dtype",
        default=None,
        choices=sorted(EXPORT_DTYPES),
        help=("Cast the weights to this dtype before tracing, fixing the dtype of the exported graph."),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Force a tiny random-init model (num_hidden_layers=2) and DEBUG logging for quick debugging.",
    )
    args = parser.parse_args()

    export(args.repo_id, args.target, export_dtype=args.export_dtype, debug=args.debug)


if __name__ == "__main__":
    main()
