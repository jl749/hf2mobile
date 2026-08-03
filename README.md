# hf2mobile

**Export HuggingFace `transformers` LMs into mobile-targeted ONNX graphs (ONNXRuntime CPU-EP, QNN-EP).**

![Python](https://img.shields.io/badge/python-%3E%3D3.12-3776AB?logo=python&logoColor=white)
![Rust](https://img.shields.io/badge/rust-extension-000000?logo=rust&logoColor=white)
![ONNX Runtime](https://img.shields.io/badge/onnxruntime-%3E%3D1.22-005CED?logo=onnx&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

```bash
hf2mobile-export    Qwen/Qwen3-0.6B --target ORT --export_dtype float32           # trace + ORT fusions, then bake in sampling + EOS ids -> inference.onnx
hf2mobile-inference 2026-08-01__ORT__Qwen-Qwen3-0.6B --prompt "Where is Paris?"   # rust runtime: graph + tokenizer, nothing else
```

---

## 📖 Table of Contents

- [🎯 Motivation](#-motivation) — why operator-level ONNX stopped being enough
- [🧩 Approach](#-approach) — the module boundary as the unit of export
- [🔧 How it works](#-how-it-works) — the export stages, and the Rust runtime that runs the result on a phone
- [📋 Requirements](#-requirements)
- [📦 Install](#-install)
- [🚀 Usage (CLI)](#-usage-cli) — export → postprocess → infer, on the host or over `adb`
- [🧭 Roadmap](#-roadmap)

---

## 🎯 Motivation

<details>
<summary><b>Click to expand</b> — why operator-level ONNX stopped being enough</summary>

### ONNX standardized the *operator* level tracing

ONNX defined a portable vocabulary of computation primitives — `MatMul`, `Conv`, `ReLU`, `Attention`.
For classical ML graphs, the operator level tracing was enough to cover majority of the model porting cases.

It worked because the vocabulary was **small and universal**. A ResNet is a fixed sequence of convolutions: recording the operators that one execution touched described the model *completely*, and any backend implementing the same standard operator set could run any graph anyone exported. Portability was a consequence of the contract being narrow — nothing in the file required knowledge that lived outside the standard.

### The operator level tracing is too low to be the unit of portability today

A single ONNX graph now has to generalize across 2 independent axes at once:

- **Hardware** — NPU / CPU / GPU, each with different quantization schemes, memory layouts, and ops support.
- **Runtimes** — TRT-LLM, vLLM, llama.cpp, ORT — each expecting different graph topology, KV-cache handling, and optimization metadata.

Covering every *(hardware × runtime)* cell at the operator level is manual, per-combination work, and subtle mismatches could silently break correctness or performance.

In practice nobody covers that matrix cell by cell. The ecosystem went **plugin-centric** instead: each runtime grows its **own** extensions — fused kernels, side-car configs, session-level switches — to express the parts of a modern model the standard vocabulary cannot. A plugin is the escape hatch from a fixed operator set, and every runtime reached for it independently. That buys back performance and expressiveness on that one runtime, but it costs precisely the property ONNX existed to provide: **a universal IR that any runtime can interpret.**

### The DAG assumption, and where modern LLMs break it

ONNX is an **interchange format**, and every mobile and edge target consumes it — directly (ONNXRuntime) or through a converter (QNN, TensorRT, OpenVINO, IREE). What it assumes in exchange is a **sequential DAG**: a static, acyclic graph you feed *once*, execute in topological order, and read outputs from — a pure function of its inputs. `If` / `Loop` / `Scan` do exist, but as second-class citizens: their bodies are *attributes* rather than values, so a branch cannot be split across backends the way a straight-line graph can, and an accelerator that compiles its partition ahead of time cannot claim a region whose trip count is unknown.

Modern LLMs and multimodal pipelines are *stateful* and autoregressive — far more involved than the traditional CV graphs the **DAG** assumption was shaped around.

That makes them hard to *export*, not merely hard to represent. `torch.onnx.export` works by **tracing**: it runs the model once on example inputs and records the operators that actually executed. Python-level control flow is evaluated during that run and then disappears — an `if` on a tensor value leaves behind only the branch it happened to take, a `for` leaves its body unrolled to the trip count it happened to see, and every shape observed becomes a constant unless explicitly marked dynamic. The export therefore captures *what the model did that one time*, and it does so without complaining: the resulting graph is perfectly valid, it just describes a narrower model than the one you started with.

Four axes make this bite. Each varies at runtime along a dimension the graph has no way to vary over, and each is resolved the same way: by moving the decision out of the graph and into the runtime. We cite **ONNXRuntime**'s answer for each — as the reference implementation of ONNX, maintained by the organization that authored the format, it is the project best positioned to have solved these *within* the graph.

| Axis | What varies, and per what | ONNXRuntime's answer | In the graph? |
| ---- | ------------------------- | -------------------- | ------------- |
| **Persistent state** | The KV cache, threaded from one decode step into the next — per **step**. A "turn" is not a single DAG pass but a *sequence* of passes sharing memory. | `com.microsoft.GroupQueryAttention` hides the cache append in-kernel, past and present sharing one preallocated buffer so it grows in place instead of being concatenated and copied. The loop *around* it moves out to a separate library, `onnxruntime-genai`, with its own `genai_config.json`. | ❌ a contrib op no converter reads, plus host code every non-ORT runtime rewrites |
| **Data-dependent routing** | An MoE router picks k of n experts — per **token**. A static graph must either evaluate every expert and mask (dense cost for sparse compute) or gather into a compact batch (data-dependent shapes). | `com.microsoft.MoE` takes `router_probs` as an ordinary input and does top-k selection inside the kernel, so the node stays static while the data-dependence happens where ONNX cannot see it. | ❌ contrib op — one opaque node with a fixed idea of what an expert is |
| **Data-dependent control flow** | Mixture-of-Depths skips a block entirely; an early-exit LM stops descending the stack — per **token**. | None. Neither a contrib op nor a runtime mechanism (`onnxruntime-genai` stops early per *sequence*, not per token), so these architectures export dense. | ❌ the saving does not survive the export at all |
| **Config-dependent weights** | Which LoRA adapter applies — per **request**, by config rather than by data. Merging (`W' = W + BA`) collapses to a static graph at the cost of a full weight set per adapter. | Adapters demoted from initializers to graph *inputs* — giving up constant folding and weight pre-packing — selected via `RunOptions.add_active_adapter` from a separate `.onnx_adapter` file. | ❌ not even an op, a session API |

`If` does not close rows 2 and 3: its predicate is one decision per graph execution, and routing needs one per token. Row 3 is the sharpest — no plugin, so no portability cost, and no capability either: the graph translates cleanly precisely *because* the thing worth exporting did not survive the export.

**Row 1 is the one everything below turns on.** It is the axis every decoder hits on every token, and the one `hf2mobile` has to take a position on.

### EXAMPLE: attention plugin

Attention has correspondingly shifted from the operator view toward a **module / plugin** view (fused attention, KV-cache blocks, MoE routers). This is the **plugin-centric** turn in its clearest form: every runtime ships and maintains its *own* fused-attention module rather than sharing one, and while everyone agrees on the mathematics, no two agree on the boundary — specifically, on where the KV cache lives relative to the op.

| Runtime | Fused attention | Where the KV cache lives |
| ------- | --------------- | ------------------------ |
| **ONNXRuntime** | `Attention` / `MultiHeadAttention` / `GroupQueryAttention` contrib ops — [`contrib_ops/cpu/bert`](https://github.com/microsoft/onnxruntime/tree/v1.27.1/onnxruntime/contrib_ops/cpu/bert) | **Inside the op.** Graph tensors — `past_key`/`past_value` in, `present_key`/`present_value` out — but when past and present are *the same* tensor it is sized to `max_sequence_length` and the kernel appends in place. With `seqlens_k`, `total_sequence_length` and `do_rotary` it applies RoPE in the same kernel. The host allocates the buffer; the op decides what happens to it, and how far into it "now" is. |
| **TensorRT-LLM** | [`gptAttentionPlugin`](https://github.com/NVIDIA/TensorRT-LLM/tree/v1.2.1/cpp/tensorrt_llm/plugins/gptAttentionPlugin), over [`cpp/tensorrt_llm/kernels`](https://github.com/NVIDIA/TensorRT-LLM/tree/v1.2.1/cpp/tensorrt_llm/kernels) | **Beside the op.** No KV tensors in that sense: with paged KV the cache is a pool of blocks handed out per request by a cache manager, and the plugin is passed the block offsets and host-side metadata needed to find them. Shape and behaviour are fixed in plugin *fields* at build time rather than expressed in a portable signature. |
| **OpenVINO** | [`ScaledDotProductAttention`](https://docs.openvino.ai/2026/documentation/openvino-ir-format/operation-sets/operation-specs/sequence/scaled-dot-product-attention.html) | **Outside the graph's I/O.** The mathematics alone — `query`/`key`/`value`, optional mask and scale, a `causal` flag. The cache is *state*: `ReadValue`/`Assign` pairs on a `Variable`, carried between `infer()` calls, reachable only through `query_state()`. Serving stacks then rewrite that again — `ov::pass::SDPAToPagedAttention` trades the state for a 28-input `PagedAttentionExtension` and block tables. One vendor, two incompatible KV contracts. |

The incompatibility is not naming, it is **who owns what**: who allocates the cache, who advances the position, who tracks sequence length. A converter cannot mechanically rewrite one into another, because what it would be rewriting is not the same node; it is a different answer to where the runtime ends and the graph begins.

So the module level is where the performance lives, and — being plugin-centric — where portability stops. A graph containing `GroupQueryAttention` is an *ONNXRuntime* graph, not an ONNX one. Decompose it back to `MatMul`/`Softmax`/`Concat` and the file translates again, at the price of the concatenate-and-copy KV growth, the fused kernel, and often the accelerator partition with it.

With no standard at that level, model publishers, agent frameworks, and distribution hubs each re-implement the same per-target conversion glue. Every converter below reads standard ONNX and nothing else, so an export falls off this table the moment it uses the plugin that made the model fast:

| Path            | Converter                                                                                               |
| --------------- | ------------------------------------------------------------------------------------------------------- |
| ONNX → QNN      | [ONNX2DLC](https://docs.qualcomm.com/doc/80-63442-10/topic/converters.html#onnx-conversion)             |
| ONNX → TensorRT | [ONNX2TRT](https://github.com/onnx/onnx-tensorrt)                                                       |
| ONNX → OpenVINO | [ONNX2OVIR](https://docs.openvino.ai/2026/openvino-workflow/model-preparation/convert-model-onnx.html)  |
| ONNX → IREE     | [ONNX2MLIR](https://iree.dev/guides/ml-frameworks/onnx/)                                                |

> **The choice as it stands:** a **fast** export bound to one runtime, or a **portable** export that gave up the reason you exported at all.
> `hf2mobile` targets exactly that middle ground.

</details>

---

## 🧩 Approach

<details>
<summary><b>Click to expand</b> — the module boundary as the unit of export</summary>

`hf2mobile` is a framework built **on top of the HuggingFace `transformers` library**. The motivation ends on a choice between a fast export bound to one runtime and a portable export that gave up the reason you exported at all. The way out has a concrete location — the **module boundary** — and everything below follows from keeping it. Instead of lowering a model to a flat operator graph and hoping each backend copes, it works one level up:

1. **Trace at the *module* level, not the operator level.** The model is traced as its semantic building blocks — attention, RoPE, RMSNorm, the causal LM head — rather than as an undifferentiated soup of `MatMul` / `Mul` / `Softmax`. `transformers` is what makes that practical: every architecture is defined against the same template — `Qwen3RMSNorm`, `LlamaAttention`, `Gemma3RotaryEmbedding` — so the semantic boundaries are already drawn, consistently, by the library the models are published with. An `Attention` `torch.nn.Module` leaves the trace as **one node**, keeping its full identity (`Qwen3Attention`, `model.layers.0.self_attn`) all the way into the graph.
2. **Expand the module level nodes — per target, and per strategy.** A traced module knows how to emit the right subgraph for its `--target`: a fused GQA/attention plugin for a runtime that supports it, a sliding-window mask template for another, or a single-head decomposition for an NPU that lacks fused attention. Same node, different expansions, decided at export time.
3. **Generalize by extending the existing exporter api, not by rewriting the graph lazily.** A plugin is selected by class-name *suffix*, so one `Attention` exporter covers every architecture that follows the convention instead of one exporter per model. Supporting a new architecture or a new hardware/runtime target means contributing a small, self-contained exporter — the framework handles the rest.

The rest is extension rather than replacement:

- **Reused** — `transformers` module definitions as the semantic baseline; ORT contrib ops (`GroupQueryAttention`, RMSNorm / RoPE fusions); ORT execution providers and `EPContext` for placement.
- **Added** — the module-level tracer and plugin registry, the per-target expanders, `SampleLogits` and the in-graph EOS constant, and the Rust runtime that drives them.

### The plugin node is the unit of control

Holding `Attention` as a single node until the last step is what makes the divergence *yours*. The HuggingFace module is the baseline, and every expansion is a deliberate, legible departure from it — chosen per target and per strategy, rather than being whatever a converter's pattern matcher happened to recognize in a graph that had already been flattened. The user decides how far the ported graph moves from the baseline, and where.

It is also what survives the trade the motivation ends on:

> The portable artifact is **not the exported file**. A graph carrying `GroupQueryAttention` is an ONNXRuntime graph, and no amount of care changes that — what stays portable is the **module-level description one step upstream of it**, which every target's file is generated *from*.

Being plugin-centric stops being a property you are stuck with, because what you keep is the description rather than any one expansion of it.

### Placing the seam: one graph, more than one backend

The goal is not merely to emit a graph a mobile runtime will accept — it is to emit one that lands on the right silicon. A phone has both an NPU and a CPU, so the interesting question is never which of the two to pick; it is **where to put the seam between them**:

| Side | Takes | Because |
| ---- | ----- | ------- |
| **NPU** | compute-heavy, statically-shaped blocks — the MLP stack, the projections | that is where the TOPS are |
| **CPU** | control flow, sampling, dynamic-shaped glue, the KV cache and the attention that reads it, and any operator the accelerator has no kernel for | that is where the flexibility is |

Left alone, that boundary is drawn by whatever the converter happens to claim, and a single unsupported op in the wrong place can strand an entire block off the accelerator. `hf2mobile` treats the split as an **export-time** decision instead: shape the graph so the NPU can take the parts that pay for themselves, and let the CPU take the rest by design rather than by accident.

The decision that sets everything else is what to do about the KV cache — the hardest piece, and the axis the motivation's first row turns on. The choice here is to **take an ORT-executable graph as the baseline** and let ORT own the cache, rather than adopt whichever cache mechanism a given backend prefers. The attention node expands into `GroupQueryAttention`, whose `past_key` / `present_key` buffers are ordinary graph tensors: the state stays visible in the IR instead of disappearing into a side-car library, and the same handling holds whether the node runs on the CPU EP or the GPU EP. We start with the **CPU EP**, being the more generic of the two.

There is a second reason attention has to sit on that side of the seam: the cache is what makes its shapes move. `total_sequence_length` grows by one on every decode step, so the tensors attention reads are a different size each time it runs. An NPU gets its efficiency from compiling a partition **ahead of time** against fixed shapes — a dimension that only becomes known per step is exactly what it cannot plan for. CPU and GPU execution providers resolve shapes at run time and simply absorb the growth. So the dynamic half of the model belongs where dynamism is free, and it belongs there for the same reason it belongs to ORT: it is the same half.

The line is drawn between *kinds* of attention, not across attention as a whole. Encoder and vision attention carry no KV cache and run at a sequence length fixed by the input, so they are statically shaped like the projections around them and belong on the NPU — expanded there into whatever form the accelerator has kernels for, a single-head decomposition included. It is **decoder** attention, the one threading a growing cache from step to step, that stays CPU-side. Same traced node, two expansions, chosen by what the module actually does.

Everything else follows from that. With the baseline runtime holding the cache, what is left is statically shaped — the QKV projections, the MLP stack, whose dimensions come from the config and do not move as the cache grows — which is precisely what an ahead-of-time compiling accelerator is built for. And ORT can carry both in one file: an `EPContext` node embeds an ahead-of-time compiled partition for another backend inside the same ONNX graph, so a single `.onnx` covers multi-device deployment. One graph, one session, mixed execution — the shape a mobile platform actually needs.

### A descriptive graph, and a generic runtime to read it

The exporter and the runtime are one deliverable, designed against each other. The graph carries the **description** — module identity survives the trace, cache slots are named, the sampling policy and the EOS ids are [baked in](#2-hf2mobilepostprocess--bake-in-the-decode-policy) — and the Rust runtime beside it ([CausalLM inferencer](#example-causallm-inferencer)) stays small and model-agnostic precisely because it can read all of that out of the file. The model's semantics stay in **one artifact, in the IR**, where the next tool can see them — instead of spread across a contrib op, a side-car config and a session API, each holding a piece of the model on its own terms.

Attention shows the division. `GroupQueryAttention` owns the in-kernel cache append, so the export targets it directly; the loop around it — prefill, decode, stop — is host work by nature, and the runtime owns that. Driving it takes no per-model knowledge, because everything the loop needs is already stated in the graph. One loop runs every exported model, and it is small enough to cross-compile for a phone.

</details>

---

## 🔧 How it works

<details>
<summary><b>Click to expand</b> — export stages, the graph at each step, and the Rust runtime</summary>

### EXAMPLE: CausalLM exporter

Every supported model inherits `CausalLMExporter` (`src/hf2mobile/exporter/causallm.py`), which drives a five-stage export. A single run traces **two cases** — **prefill** (the full prompt) and **generation** (single-token decode with KV cache in/out) — because those are the two distinct shapes a decoder actually runs at inference time. For `ORT` the deliverable is the generation graph alone, whose dynamic `L` covers both.

| Stage | What happens | Where it lives |
| ----- | ------------ | -------------- |
| **1. Trace module I/O** | Run `model.generate(...)` once and record the inputs/outputs of each semantic module (the "plugin" boundaries). | `src/hf2mobile/tracing/` |
| **2. Export subgraphs** | Emit each traced module as a standalone ONNX subgraph, specialized for the target. | `src/hf2mobile/exporter/submodules/` |
| **3. Register plugin ops** | Register the custom ops so the main-graph export can reference them by name instead of inlining operators. | `src/hf2mobile/tracing/register.py` |
| **4. Export model cases** | `torch.onnx.export` the model for each unique case (prefill + generation), with a KV-cache-aware forward so the cache I/O survives dead-code elimination. | `src/hf2mobile/exporter/causallm.py` |
| **5. Merge + postprocess** | Inline the subgraphs into each main graph, then run target-specific fusions and postprocessing. | `src/hf2mobile/exporter/onnx/` |

Stage 5 is where the target specialization becomes concrete — e.g. for `ORT`:

- **Fusion** (`onnx/fusion/`): fuse RMSNorm, RoPE and Group-Query-Attention into ORT contrib ops (`fuse_rms_norm`, `fuse_rope`, `fuse_group_query_attention`).
- **Postprocess** (`onnx/postprocess/`): make I/O shapes dynamic and attach the sliding-window mask (`attach_sliding_window_mask_onnx`).

Adding a new architecture is usually a thin subclass of `CausalLMExporter` (see `llama.py`, `qwen2.py`, `qwen3.py`, `gemma3.py`); adding a new target is mostly new branches under `onnx/fusion` and `onnx/postprocess`.

> **On memory.** Stages 4–5 hold the graph as a single `onnx_ir.Model` from load to save. `ir.load` mmaps external tensors — the weights are page cache the kernel can evict, not heap — and every fusion and postprocess mutates that one live model in place, so there is no save/reload round-trip between stages. Exporting a multi-GB model no longer means materializing its weights once per stage.

Then, past the exporter:

| Stage | What happens | Where it lives |
| ----- | ------------ | -------------- |
| **6. Bake the decode policy** | Append a `SampleLogits` node so the graph returns a token id, and park the EOS ids in the graph as a `Constant`. | `src/hf2mobile/postprocess.py` |
| **7. Run it** | Tokenize, prefill, decode, detokenize — in Rust, over the postprocessed graph. | `src/onnx_inferencer/` |

The graph progresses through the export like this:

| <img src="docs/1_module_level_graph.svg" width="200"> | <img src="docs/2_module_level_postprocessed_graph.svg" width="200"> | <img src="docs/3_final_graph.svg" width="200"> | <img src="docs/4_final_graph_postprocessed.svg" width="200"> |
| :---: | :---: | :---: | :---: |
| Initial module-level graph | Postprocessed module-level graph | Flattened operator-level graph (ORT: CPU-EP target) | Decode policy baked in (`SampleLogits` + EOS ids) |

---

### EXAMPLE: CausalLM inferencer

The exported graph is only half the deliverable; a runtime has to load it. `hf2mobile` ships two Rust crates that share one source file:

| Crate | Artifact | Role |
| ----- | -------- | ---- |
| `src/onnx_inferencer/` | `hf2mobile._ortrs_binding` (a Python extension module, built by maturin) | *Drives* ONNXRuntime — session setup, KV cache, prefill/decode loop, tokenizer, timing. Backs `python -m hf2mobile.infer`. |
| `src/onnx_inferencer/` | `hf2mobile-infer` (a standalone executable, built by cargo) | The same engine with no interpreter, so it cross-compiles: `adb push` it to a phone with an export directory and run the graph on the device it was built for. |
| `src/onnx_plugins/` | `libhf2mobile_plugins.so` | *Driven by* ONNXRuntime — a custom-op library exporting the C `RegisterCustomOps` entry point, loadable from Python, C++ or an Android app. |

They are separate crates (and separate cargo workspaces) because they need opposite `ort` configurations: the runtime dlopens onnxruntime, the plugin is already running inside it. But `sample_logits.rs` is compiled into **both**, so the token a mobile runtime picks and the token the dev runtime picks come from one definition.

Nothing tells the engine *how* to decode. The policy is a `SampleLogits` node and the stop ids are a `hf2mobile_EOS_tokens` constant, both baked into the graph by postprocess:

```text
logits [1, L, vocab]  --SampleLogits(top_k, top_p, temperature)-->  sampled_token [1, 1] int32
```

A 262k-wide fp32 logits row is 1 MB per token, so reducing it to one integer *inside* the graph is the copy the decode loop most wants back. What the engine adds around that: a zero-copy KV cache (tensors stay ORT-side between steps), one dynamic-`L` graph serving both prefill and decode, and TTFT/tok-s per run. `DEBUG=1` also dumps the Level3-optimized graph and a `chrome://tracing` profile.

Two front ends, one engine — a cargo feature picks which is compiled:

#### HostPC

```bash
hf2mobile-inference 2026-08-01__ORT__Qwen-Qwen3-0.6B --prompt "Where is Paris?"
```

```python
from hf2mobile.inference import CausalLMInferencer

lm = CausalLMInferencer("inference.onnx", "tokenizer.json")
text, (ttft_s, tps) = lm.generate("Where is Paris?", num_generation=64)
```

#### Android

`hf2mobile-infer` is the same engine with the pyo3 layer swapped for a CLI, so it cross-compiles — no interpreter, no app, no JNI, and no `libhf2mobile_plugins.so` (the operator is compiled in). Three files go to the phone: the binary, an `arm64-v8a` `libonnxruntime.so` (dlopened, so it is found beside the binary at runtime), and the export directory.

```bash
nix develop .#android -c cargo build --release --target aarch64-linux-android --bin hf2mobile-infer

D=/data/local/tmp/hf2mobile && adb shell mkdir -p $D
adb push target/aarch64-linux-android/release/hf2mobile-infer libonnxruntime.so $D/
adb push 2026-08-01__ORT__google-gemma-3-270m-it $D/
adb shell chmod +x $D/hf2mobile-infer

adb shell "$D/hf2mobile-infer $D/2026-08-01__ORT__google-gemma-3-270m-it --prompt 'Where is Paris?'"
```

```text
[hf2mobile] INFO     | loaded 39 inputs / 37 outputs (36 KV cache slots) | eos: [1, 106]
Paris is a French city, which is known for its iconic landmarks and rich history.
[hf2mobile] INFO     | 14 prompt tokens, 18 generated (EOS) | TTFT 113.9 ms | 13.38 tok/s
```

*(that run is the same binary on the host — a device's numbers will differ, the point is that the two are directly comparable)*

The binary applies the model's chat template itself — the job `transformers` does on the host — rendering the export's own Jinja, so the prompt reaching the model is token-for-token what the Python path produces.

</details>

---

## 📋 Requirements

- Python **>= 3.12**
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- A **Rust toolchain** + [`maturin`](https://www.maturin.rs/) — only to build from source; `hf2mobile` is a mixed Rust/Python project, and a build compiles the `_ortrs_binding` extension module

Everything above comes from `flake.nix`; nothing needs to be installed globally:

```bash
nix develop      # rustc, cargo, rustfmt, rust-analyzer, maturin, python312, uv — and activates .venv
```

**To run on an Android device**, additionally:

- The **Android NDK** (its clang links the binary and builds the C sources inside `tokenizers`) and **`adb`** — both in a second shell, `nix develop .#android`, so the multi-GB NDK stays out of the day-to-day one. The `aarch64-linux-android` standard library is already on the toolchain in both shells.
- An **ONNX Runtime built for `arm64-v8a`** — `jni/arm64-v8a/libonnxruntime.so` out of the [`onnxruntime-android` AAR](https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/). The copy in your `.venv` is an x86-64 host build and will not load on a phone.
- A device with USB debugging on. Nothing needs root: `/data/local/tmp` is writable *and* executable by `adb shell`.

The package pins CPU-only PyTorch wheels by default (see the `[tool.uv.index]`
block in `pyproject.toml`). To use CUDA wheels instead, remove that block.

---

## 📦 Install

### HostPC

```bash
uv pip install dist/hf2mobile-0.1.0-cp312-abi3-linux_x86_64.whl   # a built wheel: no Rust toolchain needed
```

From source, for development:

```bash
uv sync                     # Python dependencies
maturin develop --release   # compile the Rust extension into the venv  (~45s cold, ~15s warm)
```

Re-run `maturin develop --release` after any change under `src/onnx_inferencer/` — Python imports the *installed* `.so`, so a bare `cargo build` will not be picked up.

To build the artifacts themselves:

```bash
maturin build --release -o dist/                                    # the wheel (abi3-py312, one per OS/arch)
cargo build --release --manifest-path src/onnx_plugins/Cargo.toml   # libhf2mobile_plugins.so
```

The plugin `.so` is only needed to load an `hf2mobile` graph from a runtime *other* than `hf2mobile.infer`, which registers the operator in-process:

```python
import onnxruntime as ort

opts = ort.SessionOptions()
opts.register_custom_ops_library("src/onnx_plugins/target/release/libhf2mobile_plugins.so")
session = ort.InferenceSession("inference.onnx", opts)
```

### Android

Nothing is *installed* on the device — one binary is cross-compiled and pushed (see [Usage](#-usage-cli)):

```bash
nix develop .#android -c cargo build --release --target aarch64-linux-android --bin hf2mobile-infer

file target/aarch64-linux-android/release/hf2mobile-infer
# ELF 64-bit LSB pie executable, ARM aarch64, interpreter /system/bin/linker64, for Android 24
```

`.cargo/config.toml` points cargo and the `cc` crate at the NDK's API-24 clang — API 24 (Android 7.0) is the floor ONNX Runtime's own Android builds target. `llvm-strip` roughly halves the 9 MB if the push is slow. The same source built for the host is just `cargo build --release`.

---

## 🚀 Usage (CLI)

<details>
<summary><b>Click to expand</b> — export, postprocess, infer, run on device</summary>

Two entry points: `hf2mobile-export` builds the shippable graph (export **and** postprocess), `hf2mobile-inference` runs it. The two stages are also runnable on their own — `python3 -m hf2mobile.export --skip_postprocess` then `python3 -m hf2mobile.postprocess {export dir}` — which is what you want when re-baking the decode policy without re-exporting.

### HostPC

#### 1. `hf2mobile.export` — HF → ONNX

```bash
hf2mobile-export {HF repo id} --target {ORT|QNN}
# OR, as a module:
python3 -m hf2mobile.export {HF repo id} --target {ORT|QNN}
```

Supported architectures: `LlamaForCausalLM`, `Qwen2ForCausalLM`, `Qwen3ForCausalLM`, `Gemma3ForCausalLM`.

| Argument         | Description                                                              | Default |
| ---------------- | ----------------------------------------------------------------------- | ------- |
| `repo_id`        | Hugging Face repo id of the model to export.                            | —       |
| `-t`, `--target` | Export target: `ORT` or `QNN`. Graph topology may change based on it.   | `ORT`   |
| `--export_dtype` | `float32` / `float16` / `bfloat16`. Casts the weights before tracing, fixing the exported graph's dtype. | the model's own dtype |
| `--debug`        | Force a tiny random-init model (2 layers) + DEBUG logging for a fast smoke test. Also keeps the intermediate graphs. | off |
| `--skip_postprocess` | Stop after the ONNX export, leaving `inference.onnx` to a separate `hf2mobile.postprocess` run. | off |
| *(postprocess flags)* | `--top_k`, `--top_p`, `--temp`, `--eos_tokens`, `--keep_logits`, `-o` — the same parser as [2.](#2-hf2mobilepostprocess--bake-in-the-decode-policy), reused here. | from `generation_config.json` |

Output lands in a timestamped directory named `{date}__{target}__{model}`:

```
2026-08-01__ORT__Qwen-Qwen3-0.6B/
├── case2.onnx + case2.onnx.data           the dynamic-L generation graph (serves prefill and decode both)
├── inference.onnx + inference.onnx.data   case2 + the baked-in decode policy — what you ship (see 2.)
├── tokenizer.json                         handed to the runtime, which does all the tokenizing
├── tokenizer_config.json                  the chat template
└── generation_config.json                 the sampling policy + EOS ids that `postprocess` reads
```

Two cases are traced — prefill (`case1`) and generation (`case2`) — but the ORT deliverable is the single dynamic-`L` generation graph that serves both, so `case1.onnx` is deleted at the end of the export. `--debug` keeps a `debug__case1.onnx` / `debug__case2.onnx` snapshot of each, plus the standalone module subgraphs.

#### 2. `hf2mobile.postprocess` — bake in the decode policy

Run for you by `hf2mobile-export` unless `--skip_postprocess` is passed; run it directly to re-bake the policy of an existing export without re-exporting.

```bash
python3 -m hf2mobile.postprocess {export dir}
python3 -m hf2mobile.postprocess {export dir} --top_k 64 --top_p 0.95 --temp 1.0
```

| Argument        | Description                                                                       | Default |
| --------------- | --------------------------------------------------------------------------------- | ------- |
| `export_dir`    | An export directory produced by `hf2mobile.export`.                               | —       |
| `--top_k`       | Keep only the k highest-scoring tokens (`0` = off).                               | from `generation_config.json` |
| `--top_p`       | Nucleus sampling threshold (`1.0` = off).                                         | from `generation_config.json` |
| `--temp`        | Logit temperature; `0` bakes in greedy and ignores `--top_k`/`--top_p`.           | from `generation_config.json` |
| `--eos_tokens`  | Stop token id to write into `hf2mobile_EOS_tokens`; repeatable.                    | from `generation_config.json` / `tokenizer_config.json` |
| `--keep_logits` | Also keep `logits` as a graph output, for checking a sampled token against its row. | off |
| `-o`, `--output`| Write the patched graph somewhere other than `<export dir>/inference.onnx`.        | — |

Writes `inference.onnx` beside `case2.onnx`. Anything omitted is resolved from the export's own configs, so the plain form reproduces what `model.generate` would have done.

#### 3. `hf2mobile.infer` — run it

```bash
hf2mobile-inference {export dir} --prompt "Where is Paris?"
# OR, as a module:
python3 -m hf2mobile.infer {export dir} --prompt "Where is Paris?"
```

| Argument           | Description                                                                  | Default |
| ------------------ | ---------------------------------------------------------------------------- | ------- |
| `export_dir`       | A **postprocessed** export directory (must hold `inference.onnx`).           | —       |
| `--prompt`         | User prompt.                                                                  | required |
| `--skip_template`  | Feed `--prompt` verbatim instead of through the tokenizer's chat template.    | off |
| `--num-generation` | Maximum tokens to generate.                                                   | `512` |
| `--intra-threads`  | ORT intra-op thread count.                                                    | one per core |

Streams to stdout, then reports prompt length, tokens generated, TTFT and tok/s. No sampling flags — the policy is in the graph.

#### Examples

```bash
# Gemma 3 (270M), end to end
hf2mobile-export google/gemma-3-270m-it --target ORT --export_dtype float32
hf2mobile-inference 2026-08-01__ORT__google-gemma-3-270m-it --prompt "Where is Paris?"

# Force greedy decoding regardless of what the model's config says
hf2mobile-export google/gemma-3-270m-it --target ORT --temp 0
# ... or re-bake an export you already have
python3 -m hf2mobile.postprocess 2026-08-01__ORT__google-gemma-3-270m-it --temp 0

# Fast smoke test with a tiny random-init model (no real weights downloaded)
hf2mobile-export meta-llama/Llama-3.2-1B-Instruct --target ORT --debug
```

### Android

Step 3 without the interpreter: one executable, the same flags, the same output. Build it as shown in [Install](#-install), then push three files and run:

```bash
D=/data/local/tmp/hf2mobile
adb shell mkdir -p $D
adb push target/aarch64-linux-android/release/hf2mobile-infer $D/
adb push libonnxruntime.so $D/                            # jni/arm64-v8a/ out of the AAR
adb push 2026-08-01__ORT__google-gemma-3-270m-it $D/      # the whole export directory
adb shell chmod +x $D/hf2mobile-infer

adb shell "$D/hf2mobile-infer $D/2026-08-01__ORT__google-gemma-3-270m-it \
           --prompt 'Where is Paris?' --num-generation 64"
```

```text
[hf2mobile] INFO     | loaded 39 inputs / 37 outputs (36 KV cache slots) | eos: [1, 106]
Paris is a French city, which is known for its iconic landmarks and rich history.
[hf2mobile] INFO     | 14 prompt tokens, 18 generated (EOS) | TTFT 113.9 ms | 13.38 tok/s
```

*(above is the host build of the same binary; a phone's numbers will differ)*

Flags are step 3's, plus `--ort-dylib` — where to dlopen ONNX Runtime from. It defaults to `$ORT_DYLIB_PATH`, then next to the binary, then the export directory, which is why the push above needs nothing set in the environment.

- **`/data/local/tmp`, not `/sdcard`** — the latter is mounted `noexec`.
- **Sweep `--intra-threads`.** On big.LITTLE the default (one thread per core) puts work on little cores that then hold the fast ones up; matching the big cluster is often quicker.
- **Compare a second run against the first.** Thermal throttling shows up as tok/s falling across a run, so one number on a warm phone is not a measurement.

</details>

---

## 🧭 Roadmap

Grouped by the axis each item unblocks. Checked items ship in the current export path; the rest are ordered roughly by priority within each group.

### 📱 QNN target (Qualcomm HTP)

- [ ] **ONNX → DLC conversion & compilation.** CLI to lower the exported ONNX to a Qualcomm DLC and compile per HTP target (v79, v81, …), fetching the resulting `EPContext` `.so`. This is what makes a QNN export actually loadable on-device.
- [ ] **QNN-scheme quantization.** Quantize following the QNN quantization scheme (HTP prefers `u16` activations over `u8` — see the Quantization group below).

### 🧠 ORT target (ONNXRuntime)

- [ ] **More contrib-op fusions.** Extend beyond RMSNorm/GQA to the remaining layer-norm family (`SkipLayerNormalization`, `SkipSimplifiedLayerNormalization`, `SimplifiedLayerNormalization`) so the graph maps onto ORT's optimized kernels.
- [ ] **MoE and LoRA plugins.** Add module exporters for data-dependent expert routing (MoE) and adapter weights (LoRA) — two of the module types the current static per-case export does not yet cover.
- [ ] **Per-token control flow.** Module exporters for Mixture-of-Depths and early-exit decoders, where whether a block runs at all is decided per token — the one axis in the motivation with no de facto answer in any runtime today.
- [ ] **NPU/CPU EP partition.** Assign the fused attention node to the CPU EP and the statically-shaped blocks (QKV projections, MLP) to the NPU EP within a single session, so one graph covers both backends.
- [ ] **Multimodal support.** Extend beyond text-only causal LMs to vision-language models (`Qwen2.5-VL`, `Phi-4-multimodal`, …): a separately traced vision encoder feeding a decoder whose sequence length is set by the input image.
- [ ] **ORT-scheme quantization.** Quantize following the ORT quantization scheme.

### 🎚 Quantization

Targeting `u16` activations for QNN instead of `u8`, to preserve accuracy on HTP.

- [ ] **SmoothQuant (`u8s8`).** Migrate activation outliers into weights (fold diagonal matrix) so both sides quantize cleanly.
- [ ] **SpinQuant / QuaRot (`f16s8`).** Rotation-based outlier suppression (hadamard matrix), starting with the `R1` rotation only(WoQ) before adding the rest.
- [ ] **AWQ / GPTQ (`f16s8`).** Weight-only PTQ methods for the weight-quantized path.

### ⏱ Runtime & benchmarking

- [x] **Host-CPU runtime.** Rust ONNXRuntime engine (`hf2mobile._ortrs_binding`) with a zero-copy KV cache and an in-graph `SampleLogits` operator; reports TTFT and TPS per run.
- [x] **Loadable custom-op library.** `libhf2mobile_plugins.so` — the same operator, as a `.so` any ONNXRuntime binding can register.
- [x] **Android CLI.** `hf2mobile-infer` — the engine as a standalone `aarch64-linux-android` executable (no Python, no app), which renders the model's chat template itself and finds ONNX Runtime beside itself. `adb push` binary + `libonnxruntime.so` + export directory, and an exported model decodes on the device it was exported for.
- [ ] **Android CI / device verification.** The cross build and the produced ELF are checked; running it against a physical device is still manual. Also cross-compile `src/onnx_plugins` for `aarch64-linux-android`, which an app driving ORT through its Java API needs.
- [ ] **Benchmark harness.** Extend the per-run TTFT/TPS numbers into peak memory and a comparison across targets.
- [ ] **More logits dtypes.** `SampleLogits` has an fp32 kernel only; fp16 needs one registration each side.
- [ ] **Regression testing (pytest).** Extend the per-architecture suite under `tests/`, including numerical parity of the exported graph against the `transformers` baseline it was traced from.
