# hf2mobile

Export HuggingFace torch models to mobile targeted (ONNXRuntime - CPUEP, QNNEP) ONNX graphs.

---

## Motivation

**ONNX standardized the *operator* level.** It defined a portable vocabulary of computation primitives — `MatMul`, `Conv`, `ReLU`, `Attention` — and that has been the right abstraction for the classical CV applications.

**But the operator level is too low to be the unit of portability today.** A single ONNX graph still has to generalize across two axes at once:

- **Hardware** — NPU / CPU / GPU, each with different quantization schemes, memory layouts, and supported ops.
- **Runtimes** — TRT-LLM, vLLM, llama.cpp, ORT — each expecting different graph topology, KV-cache handling, and optimization metadata.

Expressing this at the operator level means a lot of manual, per-combination work, and subtle differences silently break correctness or performance — it does not scale.

**Meanwhile the models became stateful.** LLMs and multimodal pipelines carry KV cache across turns, chain encoders into decoders, and expose capabilities (speculative decoding, structured output) that live *above* individual
operators but *below* the application.

Statefulness pulls the model away from the very assumption ONNX was built on: a **classical sequential DAG** — a static, acyclic graph you feed *once*, execute in topological order, and read outputs from, as a pure function of its inputs. A stateful decoder breaks nearly every clause of that sentence:

- **Persistent state / loop-carried dependency.** The KV cache is mutable state threaded from one decode step into the next. A "turn" is not a single DAG pass but a *sequence* of passes sharing memory — closer to a stateful loop than to a pure function.
- **Data-dependent, non-linear control flow.** Plugins like gated attention bypass the normal sequential flow: an input-conditioned gate decides, per token and per head, whether the attention branch *runs at all* — letting the runtime skip the computation rather than always executing it and multiplying by a near-zero weight. Which path actually runs is conditioned on the data, not a fixed forward sequence of ops.
- **Data-dependent routing.** In an MoE layer the experts that actually run depend on the input token, so the computation dynamically dispatches a *different* subgraph per token instead of executing one static path.

Each of these is a place where a single static operator-DAG stops being an accurate description of what actually runs — which is exactly why the portable unit has to move up to the module level.

The industry's attention has correspondingly shifted from the operator view toward a **module / plugin** view — but there is no standard covering that middle ground, so model publishers, agent frameworks, and distribution hubs each re-implement the same per-target conversion glue. (e.g. https://github.com/onnx/onnx-tensorrt, https://docs.qualcomm.com/doc/80-63442-10/topic/converters.html#onnx-conversion, https://iree.dev/guides/ml-frameworks/onnx/)

## Proposed Approach

`hf2mobile` is a framework built **on top of the HuggingFace `transformers` library** that targets exactly that middle ground and is designed to be easy to extend to new architectures and new targets.
Instead of lowering a model to a flat operator graph and hoping each backend copes, it works one level up:

1. **Trace at the *module* level, not the operator level.** The model is traced as its semantic building blocks (attention, RoPE, RMSNorm, the causal LM head) rather than as an undifferentiated soup of `MatMul`/`Mul`/`Softmax`.
2. **Expand each module differently based on the target.** A traced module knows how to emit the right subgraph for its `--target` — e.g. a fused GQA/attention plugin for a runtime that supports it, a sliding-window mask template for another, or a single-head-attention decomposition for an NPU that lacks fused attention.
3. **Generalize by adding a module exporter, not by rewriting the graph.** Supporting a new architecture or a new hardware/runtime target means contributing a small, self-contained exporter — the framework handles the rest.

The result is a single, extensible pipeline that produces correctly-specialized graphs per target while keeping the shared, model-semantic structure in one place.

---

## Tested Architectures
- `LlamaForCausalLM`
- `Qwen2ForCausalLM`
- `Qwen3ForCausalLM`
- `Gemma3ForCausalLM`

---

## Requirements

- Python **>= 3.12**
- [`uv`](https://docs.astral.sh/uv/) for dependency management and builds

The package pins CPU-only PyTorch wheels by default (see the `[tool.uv.index]` block in `pyproject.toml`).
If you would like to use CUDA wheels instead remove that block.

---

## Install

### From a built wheel (production)

```bash
uv pip install dist/hf2mobile-0.1.0-py3-none-any.whl
```

### From source (development)

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

Export a mobile CPU/NPU targeted ONNX graph.

```bash
python3 -m hf2mobile.export {HF repo id} --target {ORT/QNN}
# OR
hf2mobile-export {HF repo id} --target {ORT/QNN}
```

### Options

| Argument            | Description                                                           | Default |
| ------------------- | --------------------------------------------------------------------- | ------- |
| `repo_id`           | Hugging Face repo id of the model to export.                          | —       |
| `-t`, `--target`    | Export target: `ORT` or `QNN`. Graph topology may change base on it   | `ORT`   |
| `--debug`           | Force a tiny random-init model (2 layers) + DEBUG logging for a fast smoke test. | off |

### Examples

```bash
# Gemma 3 (270M) for ONNXRuntime
python3 -m hf2mobile.export google/gemma-3-270m-it --target ORT

# Qwen2.5 (0.5B) for ONNXRuntime
python3 -m hf2mobile.export Qwen/Qwen2.5-0.5B-Instruct --target ORT

# Fast smoke test with a tiny random-init model
python3 -m hf2mobile.export meta-llama/Llama-3.2-1B-Instruct --target ORT --debug
```

---

## Roadmap / TODO

**QNN**
- [ ] ONNX → DLC conversion and compilation(EPContext fetching .so) CLI, per HTP target (e.g. v79, v81, …).
- [ ] ONNX quantization support (must follow the QNN quantization scheme).

**ORT**
- [ ] More contrib-op plugin fusions (e.g. `SimplifiedLayerNormalization`).
- [ ] Support MoE and LoRA plugins.
- [ ] ONNX quantization support (must follow the ORT quantization scheme).

**Quantization(u16 instead of u8 act for QNN)**
- [ ] SmoothQuant (u8s8).
- [ ] SpinQuant / QuaRot (R1 only to begin with) (f16s8).
- [ ] AWQ / GPTQ (f16s8).

**Runtime**
- [ ] Model executable on Android (cross-compile).
- [ ] Benchmark support (TTFT, TPS, peak Mem .. etc)
