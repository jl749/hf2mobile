import torch
import transformers

from hf2hw import CausalLMExporter


def main():
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
    ).cpu()
    model.eval()

    prompt = "Hello"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Phase 1 + 2: observe a short generation to capture per-case IO profiles,
    # then patch the plugin modules' forward in place.
    tracer = CausalLMExporter(
        model,
        plugin_suffix=("Attention", "RotaryEmbedding"),
        # plugin_suffix=("RotaryEmbedding"),
    )
    tracer.trace_plugin_io(
        model_inputs={
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"]
        },
        max_new_tokens=2,
        do_sample=False,
    )

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
