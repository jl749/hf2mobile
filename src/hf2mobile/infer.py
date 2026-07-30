#!/usr/bin/env python3
"""
# hf2mobile.infer (try exported+postprocessed ONNX inference on the host CPU)

A thin CLI over the Rust runtime (`hf2mobile._ortrs_binding`, sources in `src/onnx_inferencer`).
It takes an *export directory* — `hf2mobile.export` wrote and `hf2mobile.postprocess` then patched

*export directory* must contain ...
    inference.onnx          the generation graph, ending in a `SampleLogits` node
    tokenizer.json          handed to the Rust runtime, which does all the tokenizing
    tokenizer_config.json   the chat template

Nothing about *how* to decode is passed in here.
The sampling strategy as well as the EOS markers live under the postprocessed graph.

    >>> python -m hf2mobile.export {hf repo id} --target ORT --export_dtype {bf16/fp16/fp32}
    >>> python -m hf2mobile.postprocess {export dir} {... optional flags ...}

KV cache management as well as token encoding/decoding are handled within the rust runtime.

Usage:
    python -m hf2mobile.infer {export dir} --prompt "Where is Paris?"
"""

import argparse
import os
from pathlib import Path

from .constant import CAUSALLM_INFERENCE_GRAPH, TOKENIZER_FILE
from .inference import CausalLMInferencer
from .utils.logger import logger


def apply_chat_template(tokenizer, prompt: str) -> str:
    """Wrap `prompt` the way the model was instruction-tuned to expect.

    An instruct model given a bare prompt tends to continue it rather than answer it, so
    this is the default. Models with no template — base models, and the byte-level fixture
    the tests use — are passed through untouched.
    """
    if not getattr(tokenizer, "chat_template", None):
        logger.warning("tokenizer carries no chat template; using the prompt as-is")
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="python -m hf2mobile.infer",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "export_dir",
        help="an export directory produced by `python -m hf2mobile.export` and `python3 -m hf2mobile.postprocess`",
    )
    ap.add_argument("--prompt", required=True, help="user prompt")
    ap.add_argument(
        "--skip_template",
        action="store_true",
        help="feed --prompt to the model verbatim, instead of through the tokenizer's chat template.",
    )
    ap.add_argument("--num-generation", type=int, default=512, help="maximum tokens to generate")
    ap.add_argument("--intra-threads", type=int, default=None, help="ORT intra-op thread count")
    args = ap.parse_args()

    export_dir = Path(args.export_dir)
    onnx_path = export_dir.joinpath(CAUSALLM_INFERENCE_GRAPH)
    tokenizer_path = export_dir.joinpath(TOKENIZER_FILE)
    if not onnx_path.exists():
        raise SystemExit(
            f"{str(export_dir)!r} holds no {CAUSALLM_INFERENCE_GRAPH}. Please postprocess the graph first:\n"
            f"    python -m hf2mobile.postprocess {args.export_dir}"
        )
    if not tokenizer_path.exists():
        raise SystemExit(f"{str(export_dir)!r} is not a complete export: {tokenizer_path.name} is missing")

    import transformers

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(str(export_dir))
    prompt = args.prompt if args.skip_template else apply_chat_template(hf_tokenizer, args.prompt)

    logger.info(f"constructing CausalLMInferencer with '{os.path.relpath(onnx_path)}'")
    lm = CausalLMInferencer(str(onnx_path), str(tokenizer_path), intra_threads=args.intra_threads)

    # TODO: better rust api instead of parsing here
    outputs = set(lm.output_names)
    kv_slots = sum(1 for name in lm.input_names if f"{name}_out" in outputs)
    logger.info(
        f"loaded {len(lm.input_names)} inputs / {len(lm.output_names)} outputs "
        f"({kv_slots} KV cache slots) | eos: {lm.eos_tokens or 'none — bounded by --num-generation'}"
    )
    threads = f"{args.intra_threads} intra-op threads" if args.intra_threads else "one thread per core"
    logger.info(f"generating up to {args.num_generation} tokens on {threads} — streaming to stdout")
    text, (ttft_s, tps) = lm.generate(prompt, num_generation=args.num_generation, stream_output=True)

    if text is None:
        print(flush=True)
    ending = "EOS" if text is not None else f"budget of {args.num_generation} exhausted, EOS not reached"
    logger.info(
        f"{lm.prefill_len} prompt tokens, {len(lm.token_ids)} generated ({ending}) | TTFT {ttft_s * 1000:.1f} ms | {tps:.2f} tok/s"
    )
    lm.reset()


if __name__ == "__main__":
    main()
