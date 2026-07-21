import argparse
from typing import Literal

import transformers

from .constant import SUPPORTED_TARGETS
from .exporter import (
    Gemma3ForCausalLMExporter,
    LlamaForCausalLMExporter,
    Qwen2ForCausalLMExporter,
    Qwen3ForCausalLMExporter,
)


def export(model_id: str, target: Literal["ORT", "QNN"], debug=False):
    assert (
        target in SUPPORTED_TARGETS
    ), f"Unsupported `{target=}`. Currently supported targets are: `{SUPPORTED_TARGETS}`."

    if debug:
        attribute = {
            "num_hidden_layers": 2,
            # sliding_window: 3,
            # attn_implementation: "eager",  # TODO: support eager
        }
    else:
        attribute = {}
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", **attribute).cpu()
    model.eval()

    architecture = model.config.architectures[0]

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

    exporter.export(
        path_template="case{i}.onnx",
        opset_version=25,
        target=target,
    )


def main():
    parser = argparse.ArgumentParser(
        prog="python -m hf2hw.export",
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
    parser.add_argument("--debug", action="store_true", help="Force small export.")
    args = parser.parse_args()

    export(args.repo_id, args.target)


if __name__ == "__main__":
    main()
