# hf2mobile

Export HuggingFace `transformers` causalLMs into mobile-targeted ONNX graphs (ONNXRuntime CPU-EP, QNN HTP-EP).

```bash
hf2mobile-export Qwen/Qwen2.5-0.5B-Instruct --target {ORT,QNN}
```

---

## Table of Contents

- [Motivation](#motivation)
- [How hf2mobile tackles it](#how-hf2mobile-tackles-it)
- [How it works (pipeline)](#how-it-works-pipeline)
- [Tested architectures](#tested-architectures)
- [Requirements](#requirements)
- [Install](#install)
- [Build the wheel](#build-the-wheel)
- [Usage (CLI)](#usage-cli)
- [Roadmap](#roadmap)

---

## Motivation

### ONNX standardized the *operator* level tracing

ONNX defined a portable vocabulary of computation primitives — `MatMul`, `Conv`, `ReLU`, `Attention`.
For classical ML graphs, the operator level tracing was enough to cover majority of the model porting cases.


### The operator level tracing is too low to be the unit of portability today

A single ONNX graph now has to generalize across 2 independent axes at once:

- **Hardware** — NPU / CPU / GPU, each with different quantization schemes, memory layouts, and ops support.
- **Runtimes** — TRT-LLM, vLLM, llama.cpp, ORT — each expecting different graph topology, KV-cache handling, and optimization metadata.

Covering every *(hardware × runtime)* cell at the operator level is manual, per-combination work, and subtle mismatches could silently break correctness or performance.

---

ONNX's core assumption is a **sequential DAG**: a static, acyclic graph you feed *once*, execute in topological order, and read outputs from — a pure function of its inputs.
Modern LLMs and multimodal pipelines are *stateful*, and statefulness breaks nearly every clause of that assumption:

- **Persistent state / loop-carried dependency.** The KV cache is mutable state threaded from one decode step into the next. A "turn" is not a single DAG pass but a *sequence* of passes sharing memory — closer to a stateful loop than to a pure function.
- **Data-dependent routing.** In an MoE layer the experts that actually run depend on the input token, so the graph dispatches a *different* subgraph per token instead of executing one static path.
- **Data-dependent control flow.** Plugins such as gated attention let an input-conditioned gate decide, per token and per head, whether the attention branch *runs at all* — the runtime skips the computation rather than always executing it and multiplying by a near-zero weight. Which path runs depends on the data, not on a fixed forward sequence of ops.
- **Configuration-dependent weight selection.** A LoRA-adapted model keeps the frozen base weights plus one or more low-rank adapters resident, and which adapter applies is chosen per *request* (by the user/config), not per token by the data. Unless the adapter is merged into the base weights ahead of time — `W' = W + BA`, which collapses back to a static graph — the artifact has to carry *multiple* candidate weight sets and select among them at runtime, which a fixed-constant DAG cannot express.

Each of these is a place where a single static operator-DAG stops being an accurate description of what actually runs — some conditioned on the data, some on the request.

### The industry moved up a level — but there is no standard there

Attention has correspondingly shifted from the operator view toward a **module / plugin** view (fused attention, KV-cache blocks, MoE routers). This is an active, ongoing effort — every runtime already ships and maintains its *own* fused-attention module rather than sharing one:

- **ONNXRuntime** — `Attention` / `MultiHeadAttention` / `GroupQueryAttention` contrib ops: [`contrib_ops/cpu/bert`](https://github.com/microsoft/onnxruntime/tree/v1.27.1/onnxruntime/contrib_ops/cpu/bert)
- **TensorRT-LLM** — [`gptAttentionPlugin`](https://github.com/NVIDIA/TensorRT-LLM/tree/v1.2.1/cpp/tensorrt_llm/plugins/gptAttentionPlugin) (the plugin boundary), backed by kernels such as `decoderMaskedMultiheadAttention` and `contextFusedMultiHeadAttention` in [`cpp/tensorrt_llm/kernels`](https://github.com/NVIDIA/TensorRT-LLM/tree/v1.2.1/cpp/tensorrt_llm/kernels)

But no standard covers that middle ground, so model publishers, agent frameworks, and distribution hubs each re-implement the same per-target conversion glue:

| Path            | Converter                                                                                               |
| --------------- | ------------------------------------------------------------------------------------------------------- |
| ONNX → QNN      | [ONNX2DLC](https://docs.qualcomm.com/doc/80-63442-10/topic/converters.html#onnx-conversion)             |
| ONNX → TensorRT | [ONNX2TRT](https://github.com/onnx/onnx-tensorrt)                                                       |
| ONNX → OpenVINO | [ONNX2OVIR](https://docs.openvino.ai/2026/openvino-workflow/model-preparation/convert-model-onnx.html)  |
| ONNX → IREE     | [ONNX2MLIR](https://iree.dev/guides/ml-frameworks/onnx/)                                                |

**`hf2mobile` targets exactly that middle ground.**

---

## How hf2mobile tackles it

`hf2mobile` is a framework built **on top of the HuggingFace `transformers` library**. Instead of lowering a model to a flat operator graph and hoping each backend copes, it works one level up:

1. **Trace at the *module* level, not the operator level.** The model is traced as its semantic building blocks — attention, RoPE, RMSNorm, the causal LM head — rather than as an undifferentiated soup of `MatMul` / `Mul` / `Softmax`.
2. **Expand each module differently per target.** A traced module knows how to emit the right subgraph for its `--target`: a fused GQA/attention plugin for a runtime that supports it, a sliding-window mask template for another, or a single-head decomposition for an NPU that lacks fused attention.
3. **Generalize by adding a module exporter, not by rewriting the graph.** Supporting a new architecture or a new hardware/runtime target means contributing a small, self-contained exporter — the framework handles the rest.

The result is a single, extensible pipeline that produces correctly-specialized graphs per target while keeping the shared, model-semantic structure in one place.

---

## How it works (pipeline)

Every supported model inherits `CausalLMExporter` (`src/hf2mobile/exporter/causallm.py`), which drives a five-stage export. A single run produces **two graphs** — a **prefill** case (processes the full prompt) and a **generation** case (single-token decode with KV cache in/out) — because those are the two distinct shapes a decoder actually runs at inference time.

| Stage | What happens | Where it lives |
| ----- | ------------ | -------------- |
| **1. Trace module I/O** | Run `model.generate(...)` once and record the inputs/outputs of each semantic module (the "plugin" boundaries). | `src/hf2mobile/tracing/` |
| **2. Export subgraphs** | Emit each traced module as a standalone ONNX subgraph, specialized for the target. | `src/hf2mobile/exporter/submodules/` |
| **3. Register plugin ops** | Register the custom ops so the main-graph export can reference them by name instead of inlining operators. | `src/hf2mobile/tracing/register.py` |
| **4. Export model cases** | `torch.onnx.export` the model for each unique case (prefill + generation), with a KV-cache-aware forward so the cache I/O survives dead-code elimination. | `src/hf2mobile/exporter/causallm.py` |
| **5. Merge + postprocess** | Inline the subgraphs into each main graph, then run target-specific fusions and postprocessing. | `src/hf2mobile/exporter/onnx/` |

Stage 5 is where the target specialization becomes concrete — e.g. for `ORT`:

- **Fusion** (`onnx/fusion/`): fuse RMSNorm and Group-Query-Attention into ORT contrib ops (`fuse_rms_norm`, `fuse_group_query_attention`).
- **Postprocess** (`onnx/postprocess/`): make I/O shapes dynamic and attach the sliding-window mask (`attach_sliding_window_mask_onnx`).

Adding a new architecture is usually a thin subclass of `CausalLMExporter` (see `llama.py`, `qwen2.py`, `qwen3.py`, `gemma3.py`); adding a new target is mostly new branches under `onnx/fusion` and `onnx/postprocess`.

The graph progresses through the export like this:

| <img src="docs/1_module_level_graph.svg" width="260"> | <img src="docs/2_module_level_postprocessed_graph.svg" width="260"> | <img src="docs/3_final_graph.svg" width="260"> |
| :---: | :---: | :---: |
| Initial module-level graph | Postprocessed module-level graph | Flattened operator-level graph (ORT: CPU-EP target) |

---

## Tested architectures

| Architecture          | Example model                     |
| --------------------- | --------------------------------- |
| `LlamaForCausalLM`    | `meta-llama/Llama-3.2-1B-Instruct` |
| `Qwen2ForCausalLM`    | `Qwen/Qwen2.5-0.5B-Instruct`      |
| `Qwen3ForCausalLM`    | `Qwen/Qwen3-0.6B`                 |
| `Gemma3ForCausalLM`   | `google/gemma-3-270m-it`          |

---

## Requirements

- Python **>= 3.12**
- [`uv`](https://docs.astral.sh/uv/) for dependency management and builds

The package pins CPU-only PyTorch wheels by default (see the `[tool.uv.index]`
block in `pyproject.toml`). To use CUDA wheels instead, remove that block.

---

## Install

**From a built wheel (production):**

```bash
uv pip install dist/hf2mobile-0.1.0-py3-none-any.whl
```

**From source (development):**

```bash
uv sync
source .venv/bin/activate
```

---

## Build the wheel

```bash
uv build
```

This produces both a wheel and an sdist in `dist/`:

```
dist/
├── hf2mobile-0.1.0-py3-none-any.whl
└── hf2mobile-0.1.0.tar.gz
```

---

## Usage (CLI)

```bash
python3 -m hf2mobile.export {HF repo id} --target {ORT|QNN}
# OR, via the installed entry point:
hf2mobile-export {HF repo id} --target {ORT|QNN}
```

### Options

| Argument         | Description                                                              | Default |
| ---------------- | ----------------------------------------------------------------------- | ------- |
| `repo_id`        | Hugging Face repo id of the model to export.                            | —       |
| `-t`, `--target` | Export target: `ORT` or `QNN`. Graph topology may change based on it.   | `ORT`   |
| `--debug`        | Force a tiny random-init model (2 layers) + DEBUG logging for a fast smoke test. | off |

Output lands in a timestamped directory named `{date}__{target}__{model}`, containing `case1.onnx` (prefill) and `case2.onnx` (generation).

### Examples

```bash
# Gemma 3 (270M) for ONNXRuntime
python3 -m hf2mobile.export google/gemma-3-270m-it --target ORT

# Qwen2.5 (0.5B) for ONNXRuntime
python3 -m hf2mobile.export Qwen/Qwen2.5-0.5B-Instruct --target ORT

# Fast smoke test with a tiny random-init model (no real weights downloaded)
python3 -m hf2mobile.export meta-llama/Llama-3.2-1B-Instruct --target ORT --debug
```

---

## Roadmap

Grouped by the axis each item unblocks. Checked items ship in the current export path; the rest are ordered roughly by priority within each group.

### QNN target (Qualcomm HTP)

- [ ] **ONNX → DLC conversion & compilation.** CLI to lower the exported ONNX to a Qualcomm DLC and compile per HTP target (v79, v81, …), fetching the resulting `EPContext` `.so`. This is what makes a QNN export actually loadable on-device.
- [ ] **QNN-scheme quantization.** Quantize following the QNN quantization scheme (HTP prefers `u16` activations over `u8` — see the Quantization group below).

### ORT target (ONNXRuntime)

- [ ] **More contrib-op fusions.** Extend beyond RMSNorm/GQA to the remaining layer-norm family (`SkipLayerNormalization`, `SkipSimplifiedLayerNormalization`, `SimplifiedLayerNormalization`) so the graph maps onto ORT's optimized kernels.
- [ ] **MoE and LoRA plugins.** Add module exporters for data-dependent expert routing (MoE) and adapter weights (LoRA) — the two module types the current static per-case export does not yet cover.
- [ ] **ORT-scheme quantization.** Quantize following the ORT quantization scheme.

### Quantization

Targeting `u16` activations for QNN instead of `u8`, to preserve accuracy on HTP.

- [ ] **SmoothQuant (`u8s8`).** Migrate activation outliers into weights (fold diagonal matrix) so both sides quantize cleanly.
- [ ] **SpinQuant / QuaRot (`f16s8`).** Rotation-based outlier suppression (hadamard matrix), starting with the `R1` rotation only(WoQ) before adding the rest.
- [ ] **AWQ / GPTQ (`f16s8`).** Weight-only PTQ methods for the weight-quantized path.

### Runtime & benchmarking

- [ ] **On-device Android execution.** Cross-compile the runtime so an exported model runs end-to-end on Android.
- [ ] **Benchmark harness.** Report TTFT (time-to-first-token), TPS (tokens/second), and peak memory across targets.
- [ ] **Regression testing (pytest).** Extend the per-architecture suite under `tests/`.
