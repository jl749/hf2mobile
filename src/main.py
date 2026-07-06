import transformers

from hf2hw import CausalLMExporter


def main():
    model_name = "Qwen/Qwen3-0.6B"
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").cpu()
    model.eval()

    exporter = CausalLMExporter(
        model=model,
        plugin_suffix=("Attention", "RotaryEmbedding"),
    )
    exporter.export(
        path_template="case{i}.onnx",
        opset_version=25,
    )


if __name__ == "__main__":
    main()
