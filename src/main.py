import torch
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
        opset_version=22,
        max_new_tokens=10,
        do_sample=False,
    )
    breakpoint()

    # Build a populated KV cache by running one prefill pass through the (now
    # patched-but-pass-through) model; we need it for case-1 (decode) export.
    with torch.no_grad():
        prefill_out = model(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            use_cache=True,
            return_dict=True,
        )

    next_token = prefill_out.logits[:, -1:].argmax(dim=-1)
    extended_mask = torch.cat(
        [
            model_inputs["attention_mask"],
            torch.ones((1, 1), dtype=model_inputs["attention_mask"].dtype),
        ],
        dim=1,
    )

    case_inputs = [
        # case 0: prefill — full prompt, no past_key_values
        {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "use_cache": False,
        },
        # case 1: decode — single token, populated KV cache
        {
            "input_ids": next_token,
            "attention_mask": extended_mask,
            "past_key_values": prefill_out.past_key_values,
            "use_cache": True,
        },
    ]

    tracer.export_graphs(case_inputs=case_inputs, path_template="case{i}.onnx")
    breakpoint()


if __name__ == "__main__":
    main()
