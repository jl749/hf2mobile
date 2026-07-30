#!/usr/bin/env python3
"""Run an exported CausalLM graph and print what it generates.

A thin CLI over the Rust runtime (``hf2mobile._ortrs_binding``, sources in
``src/onnx_inferencer``). It takes an *export directory* — whatever ``hf2mobile.export``
wrote — and picks the pieces out of it:

    case2.onnx              the generation graph
    tokenizer.json          handed to the Rust runtime, which does all the tokenizing
    tokenizer_config.json   the chat template, and the `eos_token` name as a fallback
    generation_config.json  the `eos_token_id`, which is what actually stops decoding

Everything expensive happens on the Rust side: the KV cache never crosses into Python, and
the timings are measured around ONNXRuntime alone. dtype is fixed at export too — a bf16
graph is refused rather than rewritten (see ``hf2mobile.export --export_dtype``).

Usage:
    python -m hf2mobile.quantize path/to/2026-01-01__ORT__model --prompt "Where is Paris?"
"""

import argparse
import json
import os
from pathlib import Path

from .constant import GENERATION_CONFIG_FILE, TOKENIZER_CONFIG_FILE, TOKENIZER_FILE
from .inference import CausalLMInferencer

GENERATION_GRAPH = "case2.onnx"


def _read_json(path: Path) -> dict:
    """Parse `path`, or return `{}` if it is absent or unreadable."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def resolve_eos_tokens(export_dir: Path, tokenizer) -> list[int]:
    """The ids that should stop decoding, in order of authority.

    1. ``generation_config.json``'s ``eos_token_id``. This wins: it is what
       ``model.generate`` itself would have used, it is already an id, and it can name
       several — Llama 3 stops on both ``<|eot_id|>`` and ``<|end_of_text|>``.
    2. ``tokenizer_config.json``'s ``eos_token``. A *name*, so it has to be looked up in
       the vocabulary, and it only ever names one.

    Returns `[]` when neither says anything, leaving generation bounded only by
    ``--num-generation``.
    """
    eos_token_id = _read_json(export_dir / GENERATION_CONFIG_FILE).get("eos_token_id")
    if eos_token_id is not None:
        # An int for most models, a list for those with several terminators. Normalize so
        # nothing downstream has to care which.
        return [eos_token_id] if isinstance(eos_token_id, int) else [int(i) for i in eos_token_id]

    eos_token = _read_json(export_dir / TOKENIZER_CONFIG_FILE).get("eos_token")
    if isinstance(eos_token, dict):
        eos_token = eos_token.get("content")  # a serialized `AddedToken`
    if isinstance(eos_token, str):
        token_id = tokenizer.convert_tokens_to_ids(eos_token)
        if token_id is not None:
            return [int(token_id)]

    return []


def apply_chat_template(tokenizer, prompt: str) -> str:
    """Wrap `prompt` the way the model was instruction-tuned to expect.

    An instruct model given a bare prompt tends to continue it rather than answer it, so
    this is the default. Models with no template — base models, and the byte-level fixture
    the tests use — are passed through untouched.
    """
    if not getattr(tokenizer, "chat_template", None):
        print("[quantize] tokenizer carries no chat template; using the prompt as-is", flush=True)
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="python -m hf2mobile.quantize",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("export_dir", help="an export directory produced by `python -m hf2mobile.export`")
    ap.add_argument("--prompt", required=True, help="user prompt")
    ap.add_argument(
        "--skip_template",
        action="store_true",
        help="feed --prompt to the model verbatim, instead of through the tokenizer's chat template.",
    )
    ap.add_argument("--num-generation", type=int, default=64, help="maximum tokens to generate")
    ap.add_argument(
        "--eos-token",
        type=int,
        action="append",
        default=None,
        help=(
            "stop on this token id; repeatable. Overrides whatever the export's "
            f"{GENERATION_CONFIG_FILE}/{TOKENIZER_CONFIG_FILE} say."
        ),
    )
    ap.add_argument("--intra-threads", type=int, default=None, help="ORT intra-op thread count")
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 (default) is greedy and reproducible; above 0 samples.",
    )
    ap.add_argument("--top-k", type=int, default=0, help="keep only the k highest-scoring tokens (0 = off)")
    ap.add_argument("--top-p", type=float, default=1.0, help="nucleus sampling threshold (1.0 = off)")
    args = ap.parse_args()

    export_dir = Path(args.export_dir)
    onnx_path = export_dir / GENERATION_GRAPH
    tokenizer_path = export_dir / TOKENIZER_FILE
    for required in (onnx_path, tokenizer_path):
        if not required.exists():
            raise SystemExit(f"{str(export_dir)!r} is not a complete export: {required.name} is missing")

    # Two tokenizers, doing different jobs. The Rust runtime gets `tokenizer.json` and does
    # every encode and decode; this transformers one exists only for the chat template and
    # the EOS-name fallback, neither of which `tokenizer.json` carries.
    import transformers

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(str(export_dir))

    eos_tokens = args.eos_token or resolve_eos_tokens(export_dir, hf_tokenizer)
    prompt = args.prompt if args.skip_template else apply_chat_template(hf_tokenizer, args.prompt)

    lm = CausalLMInferencer(str(onnx_path), str(tokenizer_path), intra_threads=args.intra_threads)
    # `flush` matters here: Rust streams straight to fd 1 while Python buffers when piped,
    # so without it the generated text lands above this header in a redirected log.
    print(
        f"[quantize] {os.path.relpath(onnx_path)} | eos: {eos_tokens or 'none — bounded by --num-generation'}",
        flush=True,
    )

    # One call for the whole budget. `text` comes back None if that budget ran out before an
    # EOS token did — the turn is still open at that point, and another `generate` would
    # carry on from the same KV cache.
    text, (ttft_s, tps) = lm.generate(
        prompt,
        num_generation=args.num_generation,
        eos_tokens=eos_tokens,
        stream_output=True,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    ending = "EOS" if text is not None else f"budget of {args.num_generation} exhausted, no EOS"
    print(
        f"\n[quantize] {lm.prefill_len} prompt tokens, {len(lm.token_ids)} generated ({ending}) | "
        f"TTFT {ttft_s * 1000:.1f} ms | {tps:.2f} tok/s"
    )
    lm.reset()


if __name__ == "__main__":
    main()
