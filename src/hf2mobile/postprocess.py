#!/usr/bin/env python3
"""
# postprocess (produce mobile inferencable ONNX graph)
## Attach sampling plugin into the exported graph.

Takes an *export directory* — `hf2mobile.export` wrote — and writes a second graph beside the first
which returns a token id instead of a row of logits:

    case2.onnx  ->  inference.onnx
    logits [1, L, vocab]  ->  SampleLogits(top_k, top_p, temperature)  ->  sampled_token [1, 1] int32

`SampleLogits` plugin is implemented under `src/onnx_plugins` and loaded by ONNXRuntime as a custom-op library:

    cargo build --release --manifest-path src/onnx_plugins/Cargo.toml
    >>> session_options = ort.SessionOptions()
    >>> session_options.opts.register_custom_ops_library("src/onnx_plugins/target/release/libhf2mobile_plugins.so")

## Insert `hf2mobile_EOS_tokens` Const node at index 0

A floating ``Constant`` node (no consumers) holding the EOS token ids as int32.
It exists so that the runtime can read the EOS values directly from ONNX without parsing json or input args

The sampling parameters (`--top_k`, `--top_p`, `--temp`) come from the export's `generation_config.json` or `tokenizer_config.json`
in case certain values are omited use `transformers`s own defaults — i.e. what `model.generate` would have used.

Usage:
    python -m hf2mobile.postprocess {hf2mobile.export output directory}
    python -m hf2mobile.postprocess {hf2mobile.export output directory} --top_k 64 --top_p 0.95 --temp 1.0
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import onnx

from .constant import (
    CAUSALLM_EXPORTED_GRAPH,
    CAUSALLM_INFERENCE_GRAPH,
    EOS_TOKENS_CONST_NAME,
    GENERATION_CONFIG_FILE,
    LOGITS_NAME,
    ONNX_DOMAIN_NAME,
    SAMPLE_LOGITS_OP,
    SAMPLED_TOKEN_NAME,
    TOKENIZER_CONFIG_FILE,
    TOKENIZER_FILE,
)
from .utils.logger import logger
from .utils.onnx_helper import drop_vi_by_name, save_onnx, update_opset

# transformer attr name -> postprocess flag name
SAMPLING_PARAMS = {
    "top_k": "--top_k",
    "top_p": "--top_p",
    "temperature": "--temp",
}

# Searched in this order; the first file that names a key wins.
SAMPLING_CONFIGS = (GENERATION_CONFIG_FILE, TOKENIZER_CONFIG_FILE)

# `do_sample: false` says the model's policy is greedy, and the three knobs above are then
# ignored by `generate`. The plugin spells greedy as `temperature <= 0`.
GREEDY = {"top_k": 0, "top_p": 1.0, "temperature": 0.0}


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _transformers_defaults() -> Dict[str, int | float]:
    return {"top_k": 50, "top_p": 1.0, "temperature": 1.0}


def _lookup(configs: Dict[str, dict], key: str) -> Tuple[str, Any] | None:
    """The first `(filename, value)` in `SAMPLING_CONFIGS` order that names `key`."""
    for filename, config in configs.items():
        if config.get(key) is not None:
            return filename, config[key]
    return None


def resolve_sampling_params(export_dir: Path, overrides: Dict[str, float | None]) -> Dict[str, float]:
    """Returns the dict with the final sampling params to apply."""
    configs = {filename: _read_json(export_dir / filename) for filename in SAMPLING_CONFIGS}
    if not any(configs.values()) and any(value is None for value in overrides.values()):
        flags = " ".join(f"{flag} <value>" for flag in SAMPLING_PARAMS.values())
        raise SystemExit(
            f"`{export_dir}` has neither {' nor '.join(SAMPLING_CONFIGS)}, so the model's own "
            f"sampling policy is unknown.\nPass the sampling parameters explicitly: {flags}"
        )

    do_sample = _lookup(configs, "do_sample")
    greedy = do_sample is not None and do_sample[1] is False
    if greedy:
        logger.info(f"postprocess: {do_sample[0]} sets `do_sample=false` -> baking greedy decoding")

    default_genconfig = _transformers_defaults()
    resolved: Dict[str, float] = {}
    for name, flag in SAMPLING_PARAMS.items():
        val: Tuple[str, Any] | None = _lookup(configs, name)
        if overrides[name] is not None:  # override (user input)
            resolved[name] = overrides[name]
        elif greedy:  # override (greedy setting)
            resolved[name] = GREEDY[name]
        elif val is not None:  # from the saved jsons
            logger.debug(f"postprocess: {flag} sets `{name}={val[1]}`")
            resolved[name] = val[1]
        else:
            logger.warning(
                f"postprocess: neither {' nor '.join(SAMPLING_CONFIGS)} sets `{name}`; falling back to "
                f"the transformers default {name}={default_genconfig[name]} (override with `{flag}`)"
            )
            resolved[name] = default_genconfig[name]

    resolved["top_k"] = max(int(resolved["top_k"]), 0)
    resolved["top_p"] = float(resolved["top_p"])
    resolved["temperature"] = float(resolved["temperature"])
    return resolved


def resolve_eos_tokens(export_dir: Path) -> List[int]:
    """Returns the list of the EOS token ids extracted from the exported jsons."""
    eos_token_id = _read_json(export_dir.joinpath(GENERATION_CONFIG_FILE)).get("eos_token_id", None)
    if eos_token_id is not None:
        return [int(eos_token_id)] if isinstance(eos_token_id, int) else [int(i) for i in eos_token_id]

    eos_token = _read_json(export_dir.joinpath(TOKENIZER_CONFIG_FILE)).get("eos_token", None)
    if isinstance(eos_token, dict):
        eos_token = eos_token.get("content")
    if isinstance(eos_token, str):
        tokenizer = _read_json(export_dir.joinpath(TOKENIZER_FILE))
        # search `added_tokens` then `model.vocab`
        for added in tokenizer.get("added_tokens", []):
            if added.get("content") == eos_token:
                return [int(added["id"])]
        token_id = tokenizer.get("model", {}).get("vocab", {}).get(eos_token)
        if isinstance(token_id, int):
            return [token_id]
        logger.warning(
            f"postprocess: {TOKENIZER_CONFIG_FILE} names `{eos_token}`, which {TOKENIZER_FILE} has no id for"
        )
    return []


def attach_sample_logits(
    model: onnx.ModelProto,
    sampling: Dict[str, float],
    eos_tokens: List[int],
    *,
    keep_logits: bool = False,
) -> None:
    """Append `SampleLogits` to `model`'s graph and prepend the EOS const node."""
    graph: onnx.GraphProto = model.graph
    if not any(vi.name == LOGITS_NAME for vi in graph.output):
        raise SystemExit(f"the graph declares no `{LOGITS_NAME}` output; is this a CausalLM graph?")
    if any(node.op_type == SAMPLE_LOGITS_OP for node in graph.node):
        raise SystemExit(f"the graph already carries a `{SAMPLE_LOGITS_OP}` node — nothing to do")

    # ===== `hf2mobile_EOS_tokens` (floating Constant, index 0) ===== #
    eos_tensor = onnx.numpy_helper.from_array(np.asarray(eos_tokens, dtype=np.int32), name=EOS_TOKENS_CONST_NAME)
    graph.node.insert(
        0,
        onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=[EOS_TOKENS_CONST_NAME],
            value=eos_tensor,
            name=EOS_TOKENS_CONST_NAME,
        ),
    )

    # ===== `SampleLogits` (graph tail) ===== #
    graph.node.append(
        onnx.helper.make_node(
            SAMPLE_LOGITS_OP,
            inputs=[LOGITS_NAME],
            outputs=[SAMPLED_TOKEN_NAME],
            name=f"node_{SAMPLE_LOGITS_OP}",
            domain=ONNX_DOMAIN_NAME,
            top_k=int(sampling["top_k"]),
            top_p=float(sampling["top_p"]),
            temperature=float(sampling["temperature"]),
        )
    )
    graph.output.append(onnx.helper.make_tensor_value_info(SAMPLED_TOKEN_NAME, onnx.TensorProto.INT32, [1, 1]))
    if not keep_logits:
        drop_vi_by_name(graph.output, {LOGITS_NAME})

    update_opset(model, ONNX_DOMAIN_NAME, 1)


def postprocess(
    export_dir: str | Path,
    *,
    top_k: int | None = None,
    top_p: float | None = None,
    temperature: float | None = None,
    eos_tokens: List[int] | None = None,
    keep_logits: bool = False,
    output: str | Path | None = None,
) -> Tuple[Path, Dict[str, float], List[int]]:
    export_dir = Path(export_dir)
    graph_path = export_dir.joinpath(CAUSALLM_EXPORTED_GRAPH)
    if not graph_path.is_file():
        raise SystemExit(
            f"`{export_dir}` does not exist or not `hf2mobile.export` directory: {CAUSALLM_EXPORTED_GRAPH} is missing"
        )

    sampling = resolve_sampling_params(export_dir, {"top_k": top_k, "top_p": top_p, "temperature": temperature})
    eos_tokens = eos_tokens if eos_tokens is not None else resolve_eos_tokens(export_dir)

    model = onnx.load(graph_path, load_external_data=True)  # TODO: share weight?
    attach_sample_logits(model, sampling, eos_tokens, keep_logits=keep_logits)

    out_path = Path(output) if output else export_dir.joinpath(CAUSALLM_INFERENCE_GRAPH)
    save_onnx(model, out_path)

    policy = ", ".join(f"{name}={sampling[name]}" for name in SAMPLING_PARAMS)
    logger.info(f"postprocess: {SAMPLE_LOGITS_OP}({policy}) -> `{SAMPLED_TOKEN_NAME}` (1, 1) int32")
    logger.info(f"postprocess: `{EOS_TOKENS_CONST_NAME}` = {eos_tokens or 'empty — the export names no stop token'}")
    logger.info(f"postprocess: wrote `{out_path}`")
    return out_path, sampling, eos_tokens


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m hf2mobile.postprocess",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("export_dir", help="an export directory produced by `python -m hf2mobile.export`")
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help=f"keep only the k highest-scoring tokens (0 = off). Overrides {GENERATION_CONFIG_FILE}'s `top_k`.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help=f"nucleus sampling threshold (1.0 = off). Overrides {GENERATION_CONFIG_FILE}'s `top_p`.",
    )
    parser.add_argument(
        "--temp",
        "--temperature",
        dest="temperature",
        type=float,
        default=None,
        help=(
            "logit temperature; 0 bakes in greedy decoding and ignores --top_k/--top_p. "
            f"Overrides {GENERATION_CONFIG_FILE}'s `temperature`."
        ),
    )
    parser.add_argument(
        "--eos_tokens",
        type=int,
        action="append",
        default=None,
        help=(
            f"stop token id to write into `{EOS_TOKENS_CONST_NAME}`; repeatable. "
            f"Overrides whatever the export's {GENERATION_CONFIG_FILE}/{TOKENIZER_CONFIG_FILE} say."
        ),
    )
    parser.add_argument(
        "--keep_logits",
        action="store_true",
        help="also keep `logits` as a graph output, for comparing the sampled token against the row it came from.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=f"write the patched graph here instead of `<export dir>/{CAUSALLM_INFERENCE_GRAPH}`.",
    )
    args = parser.parse_args()

    postprocess(
        args.export_dir,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        eos_tokens=args.eos_tokens,
        keep_logits=args.keep_logits,
        output=args.output,
    )


if __name__ == "__main__":
    main()
