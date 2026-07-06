"""ONNX Runtime inference for the exported two-case CausalLM.

The exporter emits two graphs:
  * case1.onnx – prefill:  input_ids[1,L], position_ids[1,L]           -> logits[1,1,V] (+ KV, if exported)
  * case2.onnx – decode:   input_ids[1,1], position_ids[1,1], past_*   -> logits[1,1,V], past_*_out

This script wires them into an autoregressive greedy loop. It introspects the
graphs, so it adapts to the number of layers / kv-heads / head-dim / dtype and
to whether case1 exports the KV cache.

NOTE (as of this writing the exported model does not yet run in onnxruntime):
  * case1 currently outputs only logits (prefill uses use_cache=False), so it
    cannot seed case2's KV — we fall back to prefilling token-by-token via case2.
  * the fused opset-23 RotaryEmbedding expects cos/sin of size head_dim/2; the
    current fusion emits full head_dim, which ORT rejects. Once that's fixed the
    loop below runs unchanged.
"""

import re

import numpy as np
import onnxruntime as ort

_ORT2NP = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(double)": np.float64}


def _np_dtype(ort_type: str):
    if ort_type in _ORT2NP:
        return _ORT2NP[ort_type]
    if ort_type == "tensor(bfloat16)":
        import ml_dtypes  # onnxruntime dependency

        return ml_dtypes.bfloat16
    raise ValueError(f"unhandled KV dtype {ort_type!r}")


def _logits_name(sess: ort.InferenceSession) -> str:
    return next(o.name for o in sess.get_outputs() if not o.name.startswith("past_"))


def _n_layers(sess: ort.InferenceSession) -> int:
    return sum(1 for i in sess.get_inputs() if i.name.startswith("past_keys_"))


def _kv_meta(sess2: ort.InferenceSession):
    """Return (n_layers, n_kv_heads, head_dim, np_dtype) from case2's past_keys_0 input."""
    pk0 = next(i for i in sess2.get_inputs() if i.name == "past_keys_0")
    _, n_kv_heads, _, head_dim = pk0.shape  # [1, H, L_prev, E]
    return _n_layers(sess2), int(n_kv_heads), int(head_dim), _np_dtype(pk0.type)


def _collect_past(sess: ort.InferenceSession, outs, n_layers: int) -> dict:
    """Map a session's `past_*_out` outputs back to case2's `past_*` input names."""
    name2out = {o.name: v for o, v in zip(sess.get_outputs(), outs)}
    past = {}
    for i in range(n_layers):
        past[f"past_keys_{i}"] = name2out[f"past_keys_{i}_out"]
        past[f"past_values_{i}"] = name2out[f"past_values_{i}_out"]
    return past


def _decode_step(sess2, token_id: int, position: int, past: dict, n_layers: int):
    feeds = {
        "input_ids": np.array([[token_id]], np.int64),
        "position_ids": np.array([[position]], np.int64),
        **past,
    }
    outs = sess2.run(None, feeds)
    logits = {o.name: v for o, v in zip(sess2.get_outputs(), outs)}[_logits_name(sess2)]
    return logits, _collect_past(sess2, outs, n_layers)


def generate(sess1, sess2, input_ids, max_new_tokens: int = 32, eos_token_id: int | None = None) -> list[int]:
    """Greedy-decode `max_new_tokens` given a prompt (1-D / [1,L] int ids)."""
    n_layers, n_kv_heads, head_dim, kv_dtype = _kv_meta(sess2)
    input_ids = np.asarray(input_ids, np.int64).reshape(1, -1)
    L = input_ids.shape[1]

    if _n_layers(sess1) > 0:
        # ---- fast prefill: case1 returns logits + KV in one shot ----
        pos = np.arange(L, dtype=np.int64)[None]
        outs = sess1.run(None, {"input_ids": input_ids, "position_ids": pos})
        logits = {o.name: v for o, v in zip(sess1.get_outputs(), outs)}[_logits_name(sess1)]
        past = _collect_past(sess1, outs, n_layers)
        cur_len = L
    else:
        # ---- case1 has no KV: prefill token-by-token through case2 from an empty cache ----
        empty = np.zeros((1, n_kv_heads, 0, head_dim), kv_dtype)
        past = {f"past_keys_{i}": empty for i in range(n_layers)}
        past.update({f"past_values_{i}": empty for i in range(n_layers)})
        logits = None
        for t in range(L):
            logits, past = _decode_step(sess2, int(input_ids[0, t]), t, past, n_layers)
        cur_len = L

    generated: list[int] = []
    next_id = int(logits[0, -1].argmax())
    for _ in range(max_new_tokens):
        generated.append(next_id)
        if eos_token_id is not None and next_id == eos_token_id:
            break
        logits, past = _decode_step(sess2, next_id, cur_len, past, n_layers)
        cur_len += 1
        next_id = int(logits[0, -1].argmax())
    return generated


def main(
    case1: str = "case1.onnx",
    case2: str = "case2.onnx",
    model_name: str = "Qwen/Qwen3-0.6B",
    prompt: str = "Where is Paris located?",
    max_new_tokens: int = 32,
):
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    sess1 = ort.InferenceSession(case1, providers=["CPUExecutionProvider"])
    sess2 = ort.InferenceSession(case2, providers=["CPUExecutionProvider"])

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(text, return_tensors="np")["input_ids"]

    out_ids = generate(sess1, sess2, input_ids, max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_token_id)
    print(tokenizer.decode(out_ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
