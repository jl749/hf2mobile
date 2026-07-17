from typing import Literal

import transformers

from .constant import SUPPORTED_TARGETS
from .exporter import LlamaForCausalLMExporter, Qwen2ForCausalLMExporter, Qwen3ForCausalLMExporter


def export(model_id: str, target: Literal["ORT", "QNN"]):
    assert (
        target in SUPPORTED_TARGETS
    ), f"Unsupported `{target=}`. Currently supported targets are: `{SUPPORTED_TARGETS}`."

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        # attn_implementation="eager",  # TODO: support eager
        # sliding_window=3,
        # num_hidden_layers=2,  # TODO: debug
    ).cpu()
    model.eval()

    architecture = model.config.architectures[0]

    if architecture == "LlamaForCausalLM":
        exporter = LlamaForCausalLMExporter(
            model=model,
            plugin_suffix=("Attention", "RotaryEmbedding"),
        )
    elif architecture == "Qwen3ForCausalLM":
        exporter = Qwen3ForCausalLMExporter(
            model=model,
            plugin_suffix=("Attention", "RotaryEmbedding"),
        )
    elif architecture == "Qwen2ForCausalLM":
        exporter = Qwen2ForCausalLMExporter(
            model=model,
            plugin_suffix=("Attention", "RotaryEmbedding"),
        )
    else:
        raise ValueError(f"{model_id=} is not a supported architecture `{architecture}`.")

    exporter.export(
        path_template="case{i}.onnx",
        opset_version=25,
        target=target,
    )


# TODO: main entry point for CLI
