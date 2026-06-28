import torch
import torch.utils._pytree as pytree
import transformers
from transformers.cache_utils import DynamicCache

from wrapper import CausalLMTracer


# Make DynamicCache a pytree leaf-container so torch.export accepts it as input.
def _flatten_dynamic_cache(cache):
    leaves = []
    for layer in cache.layers:
        leaves.append(layer.keys)
        leaves.append(layer.values)
    return leaves, len(cache.layers)


def _unflatten_dynamic_cache(values, num_layers):
    ddp_data = [(values[2 * i], values[2 * i + 1]) for i in range(num_layers)]
    return DynamicCache(ddp_cache_data=ddp_data)


def _flatten_with_keys_dynamic_cache(cache):
    leaves = []
    for li, layer in enumerate(cache.layers):
        leaves.append((pytree.SequenceKey(2 * li), layer.keys))
        leaves.append((pytree.SequenceKey(2 * li + 1), layer.values))
    return leaves, len(cache.layers)


pytree.register_pytree_node(
    DynamicCache,
    _flatten_dynamic_cache,
    _unflatten_dynamic_cache,
    flatten_with_keys_fn=_flatten_with_keys_dynamic_cache,
)


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
    tracer = CausalLMTracer(
        model,
        plugin_suffix=("Attention", "RotaryEmbedding"),
        # plugin_suffix=("RotaryEmbedding"),
    )
    tracer.trace_graph(
        model_inputs={
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"]
        },
        max_new_tokens=2,
        do_sample=False,
    )
    return  # TODO: remove

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


if __name__ == "__main__":
    main()
