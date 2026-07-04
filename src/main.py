import transformers

from hf2hw import CausalLMExporter


def main():
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").cpu()
    model.eval()

    prompt = "Where is Paris located?"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    exporter = CausalLMExporter(
        model=model,
        plugin_suffix=("Attention", "RotaryEmbedding"),
    )
    exporter.export(
        model_inputs=model_inputs,
        path_template="case{i}.onnx",
        opset_version=25,
        max_new_tokens=10,
        do_sample=False,
    )


if __name__ == "__main__":
    main()
