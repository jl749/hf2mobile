#!/usr/bin/env python3
"""ONNX Runtime inference for an exported Qwen3 CausalLM generation graph (``case2.onnx``).

The exported graph is bf16 and dynamic in the query length ``L``, so the *same* graph serves
both prefill (``L > 1`` with an empty KV cache) and decode (``L = 1`` with a growing cache).

ORT's CPU execution provider has no bf16 kernels (e.g. bf16 ``MatMul`` is not implemented), so
on CPU the model is upcast to fp32 once and cached next to the original as ``<name>.fp32.onnx``
(+ ``.data``). Upcasting bf16 -> fp32 is lossless for the stored weights. On a bf16-capable
provider (CUDA) pass ``--no-upcast`` to run the graph natively.

Usage:
    python3 ort_infer.py --onnx path/to/case2.onnx --prompt "hello"
    python3 ort_infer.py --onnx case2.onnx --prompt "Where is Paris?" --max-new-tokens 64
"""

import argparse
import os
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnx import AttributeProto, TensorProto, numpy_helper


def _np_dtype(ort_type: str):
    """Map an ORT input/output type string to the numpy dtype ORT expects to be fed."""
    simple = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
    }
    if ort_type in simple:
        return simple[ort_type]
    if ort_type == "tensor(bfloat16)":
        import ml_dtypes  # onnxruntime dependency

        return ml_dtypes.bfloat16
    raise ValueError(f"unhandled ORT tensor type {ort_type!r}")


_FLOAT_ELEM_TYPES = {TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.BFLOAT16, TensorProto.DOUBLE}


def assert_finite(model: onnx.ModelProto, stage: str = "", abs_max: float = 1e30) -> None:
    """Raise if any float weight holds non-finite or absurdly large (``|x| > abs_max``) values.

    A sanity check on a loaded export: corruption from uninitialized memory shows up as
    ``NaN``/``inf`` or bf16-range garbage (``~1e38``), which normal weights never reach. Scans
    initializers and ``Constant`` values in the main graph and every FunctionProto body; integer
    tensors (shapes/axes) are skipped.
    """
    bad: list[str] = []

    def _check(tp: onnx.TensorProto, where: str) -> None:
        if tp.data_type not in _FLOAT_ELEM_TYPES:
            return
        arr = numpy_helper.to_array(tp).astype(np.float32)
        if arr.size == 0:
            return
        finite = np.isfinite(arr)
        if not finite.all():
            bad.append(f"{tp.name}[{where}]: {int((~finite).sum())} non-finite")
        elif np.abs(arr).max() > abs_max:
            bad.append(f"{tp.name}[{where}]: max|x|={float(np.abs(arr).max()):.3e}")

    def _scan(nodes, initializers, where: str) -> None:
        for tp in initializers:
            _check(tp, where)
        for node in nodes:
            if node.op_type == "Constant" and node.output:
                for attr in node.attribute:
                    if attr.name == "value":
                        _check(attr.t, where)

    _scan(model.graph.node, model.graph.initializer, "main")
    for func in model.functions:
        _scan(func.node, (), f"fn:{func.name}")

    if bad:
        raise ValueError(f"assert_finite FAILED after {stage!r}: {len(bad)} corrupt tensor(s) -> " + "; ".join(bad[:5]))


# ── bf16 -> fp32 upcast (for CPU) ─────────────────────────────────────────────


def _upcast_model_to_fp32(model: onnx.ModelProto) -> onnx.ModelProto:
    """Rewrite every bf16 tensor / IO type in ``model`` to fp32, in place."""

    def _conv(tp: onnx.TensorProto) -> None:
        if tp.data_type == TensorProto.BFLOAT16:
            tp.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(tp).astype(np.float32), tp.name))

    for tp in model.graph.initializer:
        _conv(tp)
    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.BFLOAT16:
                    attr.i = TensorProto.FLOAT  # keep explicit dtype casts consistent with the upcast
        for attr in node.attribute:
            if attr.type == AttributeProto.TENSOR:
                _conv(attr.t)
            for t in attr.tensors:
                _conv(t)
    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        tt = vi.type.tensor_type
        if tt.elem_type == TensorProto.BFLOAT16:
            tt.elem_type = TensorProto.FLOAT
    del model.graph.value_info[:]  # drop stale (bf16) intermediates; let ORT re-infer as fp32
    return model


def _build_session(onnx_path: str, providers: list[str], upcast: bool) -> ort.InferenceSession:
    """Create an ORT session, upcasting bf16 -> fp32 (cached on disk) when requested."""
    so = ort.SessionOptions()
    so.log_severity_level = 3
    if not upcast:
        return ort.InferenceSession(onnx_path, so, providers=providers)

    fp32_path = f"{onnx_path}.fp32.onnx"
    if not os.path.exists(fp32_path):
        print(f"[ort_infer] upcasting bf16 -> fp32 (one-time) -> {fp32_path}")
        model = onnx.load(onnx_path, load_external_data=True)
        assert_finite(model, "loaded export")  # sanity-check weights before wasting time on inference
        model = _upcast_model_to_fp32(model)
        data_name = os.path.basename(fp32_path) + ".data"
        data_path = os.path.join(os.path.dirname(fp32_path), data_name)
        if os.path.exists(data_path):
            os.remove(data_path)  # onnx refuses to overwrite an existing external-data file
        onnx.save(
            model,
            fp32_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_name,
            size_threshold=1024,
        )
    return ort.InferenceSession(fp32_path, so, providers=providers)


# ── autoregressive greedy decode ──────────────────────────────────────────────


def generate(sess: ort.InferenceSession, input_ids: np.ndarray, max_new_tokens: int, eos_ids: set[int]) -> list[int]:
    """Greedy-decode from a [1, L] prompt through the single dynamic-L generation graph."""
    n_layers = sum(1 for i in sess.get_inputs() if i.name.startswith("past_keys_"))
    pk0 = next(i for i in sess.get_inputs() if i.name == "past_keys_0")
    _, n_kv_heads, _, head_dim = pk0.shape  # [1, H, L_prev, E]
    kv_dtype = _np_dtype(pk0.type)
    out_names = [o.name for o in sess.get_outputs()]

    # start from an empty KV cache (L_prev = 0)
    past = {f"past_keys_{i}": np.zeros((1, n_kv_heads, 0, head_dim), kv_dtype) for i in range(n_layers)}
    past.update({f"past_values_{i}": np.zeros((1, n_kv_heads, 0, head_dim), kv_dtype) for i in range(n_layers)})
    cur = 0  # number of tokens already in the cache

    def step(tokens: np.ndarray) -> int:
        nonlocal past, cur
        seq = tokens.shape[1]
        position_ids = np.arange(cur, cur + seq, dtype=np.int64)[None]
        outs = dict(zip(out_names, sess.run(None, {"input_ids": tokens, "position_ids": position_ids, **past})))
        past = {
            **{f"past_keys_{i}": outs[f"past_keys_{i}_out"] for i in range(n_layers)},
            **{f"past_values_{i}": outs[f"past_values_{i}_out"] for i in range(n_layers)},
        }
        cur += seq
        logits = outs["logits"].astype(np.float32)  # [1, seq, V]  (bf16/fp16 -> fp32 for argmax)
        return int(logits[0, -1].argmax())

    # the exported Attention is causal (is_causal=1), so a batched multi-token prefill is
    #   correctly lower-triangular (empty KV -> square) and decode (q_len=1) attends all past.
    next_id = step(input_ids.astype(np.int64))  # one-shot prefill

    generated: list[int] = []
    for _ in range(max_new_tokens):
        if next_id in eos_ids:
            break
        generated.append(next_id)
        next_id = step(np.array([[next_id]], np.int64))  # decode one token
    return generated


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True, help="path to the exported generation graph (case2.onnx)")
    ap.add_argument("--prompt", required=True, help="user prompt")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HF repo for the tokenizer")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--provider", default="CPUExecutionProvider")
    ap.add_argument("--no-upcast", action="store_true", help="run the bf16 graph natively (needs a bf16-capable EP)")
    args = ap.parse_args()

    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(text, return_tensors="np")["input_ids"]

    sess = _build_session(args.onnx, providers=[args.provider], upcast=not args.no_upcast)

    eos_ids = {tid for tid in (tokenizer.eos_token_id, getattr(tokenizer, "pad_token_id", None)) if tid is not None}
    print(f"[ort_infer] prompt tokens: {input_ids.shape[1]} | provider: {args.provider}")
    t0 = time.perf_counter()
    out_ids = generate(sess, input_ids, args.max_new_tokens, eos_ids)
    dt = time.perf_counter() - t0

    print("\n" + tokenizer.decode(out_ids, skip_special_tokens=True))
    print(f"\n[ort_infer] {len(out_ids)} tokens in {dt:.1f}s ({len(out_ids) / dt:.2f} tok/s)")


if __name__ == "__main__":
    main()
