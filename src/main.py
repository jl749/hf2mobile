import json
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
        enable_thinking=True  # by default
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    model_wrapper = CausalLMWrapper(
        model, 
        model_inputs,
        plugin_suffix=("Attention", "RotaryEmbedding")
    )
    print(model_wrapper.captured_plugin_inputs)
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(model_wrapper.captured_plugin_inputs, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
