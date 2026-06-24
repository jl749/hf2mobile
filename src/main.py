import transformers

from wrapper import CausalLMWrapper


def main():
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
    ).cpu()

    prompt = "Give me a short introduction to large language model./think"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    model_wrapper = CausalLMWrapper(
        model,
        model_inputs,
        plugin_suffix=("Attention", "RotaryEmbedding"),
    )

    # Prefill: full prompt length
    # Decode:  single new token (last position), same attention mask length
    model_wrapper.export_graphs(
        prefill_inputs={
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
        },
        decode_inputs={
            "input_ids": model_inputs["input_ids"][:, -1:],
            "attention_mask": model_inputs["attention_mask"],
        },
        prefill_path="prefill.onnx",
        decode_path="decode.onnx",
    )


if __name__ == "__main__":
    main()
