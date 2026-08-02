# hf2mobile

Export HuggingFace `transformers` LMs into mobile-targeted ONNX graphs (ONNXRuntime CPU-EP, QNN-EP).

```bash
python3 -m hf2mobile.export      Qwen/Qwen3-0.6B --target ORT --export_dtype float32           # module-level trace + ORT fusions -> case2.onnx
python3 -m hf2mobile.postprocess 2026-08-01__ORT__Qwen-Qwen3-0.6B                              # bake in sampling + EOS ids      -> inference.onnx
python3 -m hf2mobile.infer       2026-08-01__ORT__Qwen-Qwen3-0.6B --prompt "Where is Paris?"   # rust runtime: graph + tokenizer, nothing else
```

---

## Table of Contents

- [Motivation](#motivation)
- [How hf2mobile tackles it](#how-hf2mobile-tackles-it)
- [How it works](#how-it-works)
- [Tested architectures](#tested-architectures)
- [Requirements](#requirements)
- [Install](#install)
- [Usage (CLI)](#usage-cli)
- [Roadmap](#roadmap)

---

## Motivation

<details>
<summary>Click to expand</summary>

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

That trade — *expressiveness bought with portability* — is the thread running through the rest of this section, and the thing `hf2mobile` is built to put back under your control.

### The DAG assumption, and where modern LLMs break it

ONNX is an **interchange format**: a protobuf file holding a graph of standardized operators plus the weights they consume, so a model trained in PyTorch can be executed by a runtime that has never heard of PyTorch. That promise is what the whole deployment stack is built on — every mobile and edge target consumes ONNX, either directly (ONNXRuntime) or through a converter (QNN, TensorRT, OpenVINO, IREE).

What it assumes in exchange is a **sequential DAG**: a static, acyclic graph you feed *once*, execute in topological order, and read outputs from — a pure function of its inputs. `If` / `Loop` / `Scan` do exist, but as second-class citizens: their bodies are *attributes* rather than values, so a branch cannot be split across backends the way a straight-line graph can, and an accelerator that compiles its partition ahead of time cannot claim a region whose trip count is unknown.

Modern LLMs and multimodal pipelines are *stateful* and autoregressive — far more involved than the traditional CV graphs the **DAG** assumption was shaped around.

That makes them hard to *export*, not merely hard to represent. `torch.onnx.export` works by **tracing**: it runs the model once on example inputs and records the operators that actually executed. Python-level control flow is evaluated during that run and then disappears — an `if` on a tensor value leaves behind only the branch it happened to take, a `for` leaves its body unrolled to the trip count it happened to see, and every shape observed becomes a constant unless explicitly marked dynamic. The export therefore captures *what the model did that one time*, and it does so without complaining: the resulting graph is perfectly valid, it just describes a narrower model than the one you started with.

Four axes make this bite. Each varies at runtime along a dimension the graph has no way to vary over (per step, per token, or per request), and each is resolved the same way: by moving the decision out of the graph and into the runtime.

For each axis below we cite **how ONNXRuntime handles it**. ONNXRuntime is the reference implementation of ONNX — maintained by the same organization that authored the format, and the most mature consumer of it — so if a limitation could be solved *within* the graph, that is the project best positioned to have done it. Every one of the four is instead answered plugin-centrically: a `com.microsoft` contrib op, a separate library (`onnxruntime-genai`), a session-level `RunOptions` switch, or nothing at all. That is the evidence these are not gaps a better exporter would close — and since every answer is a runtime specific plugin, every answer is illegible to the converters in the table below.

- **Persistent state / loop-carried dependency.** The KV cache is mutable state threaded from one decode step into the next. A "turn" is not a single DAG pass but a *sequence* of passes sharing memory — closer to a stateful loop than to a pure function. ONNXRuntime splits this in half: `com.microsoft.GroupQueryAttention` hides the per-step cache append inside the kernel, with past and present sharing one preallocated buffer so the cache grows in place instead of being concatenated and copied — while the loop *around* it moves out into a separate library, `onnxruntime-genai`, carrying its own `genai_config.json`. Neither half is in the ONNX graph: one is a plugin no converter below can read, the other is host code every non-ORT runtime rewrites.
- **Data-dependent routing.** In an MoE layer the router picks k of n experts *per token*, so which FLOPs actually run depends on the input. ONNX has no way to dispatch per token, so a static graph has to fake it — evaluate every expert and mask the unused ones (paying dense cost for sparse compute), or gather the routed tokens into a compact batch (data-dependent shapes, which defeat the static shape inference that ahead-of-time memory planning is built on). Either way the sparsity that MoE exists for does not survive the export. `If` does not close the gap: its predicate is one decision per graph execution, and routing needs one per token. ONNXRuntime's answer is to push the routing *below* the graph — `com.microsoft.MoE` takes `router_probs` as an ordinary input and performs the top-k expert selection inside the kernel, so the node stays static while the data-dependence happens where ONNX cannot see it. It genuinely works (the CPU kernel runs as of ORT 1.27), but it is a single opaque op with a fixed idea of what an expert is, and it is a `com.microsoft` contrib op — none of the converters in the table below can read it.
- **Data-dependent control flow.** Mixture-of-Depths puts a router in front of each block so a token can skip the layer *entirely*; an early-exit LM stops descending the stack once a token's prediction is confident enough. In both, whether a computation runs at all is decided by the data, per token, at runtime. A graph fixed at export time has one answer for every token — so the saving, which is the whole point of the technique, is exactly what is lost. Here there is no de facto answer at all: ONNXRuntime has neither a contrib op nor a runtime mechanism for skipping a layer per token (`onnxruntime-genai` stops early per *sequence*, not per token), so these architectures export dense and hand back exactly the compute they were designed to save. No plugin, so no portability cost — and no capability either: the graph translates cleanly precisely *because* the thing worth exporting did not survive the export.
- **Configuration-dependent weight selection.** A LoRA-adapted model keeps the frozen base weights plus one or more low-rank adapters resident, and which adapter applies is chosen per *request* (by the user/config), not per token by the data. Merging ahead of time — `W' = W + BA` — collapses back to a static graph, at the cost of a full weight set per adapter. ONNXRuntime instead demotes the adapters from initializers to graph *inputs* — giving up the constant folding and weight pre-packing the base weights keep — and selects among them through `RunOptions.add_active_adapter`, fed from a separate `.onnx_adapter` file. That works, and it is exactly the shape of the problem: the graph could not express the choice, so the runtime grew a plugin beside it — here not even an op, but a session API. The adapter switch cannot be converted because it was never *in* the file that gets converted.

Each of these is a place where a single static operator-DAG stops being an accurate description of what actually runs — some conditioned on the data, some on the request. And in every case the working answer, ONNXRuntime's included, is plugin-centric — which runs, and does not translate.

### EXAMPLE: attention plugin

Attention has correspondingly shifted from the operator view toward a **module / plugin** view (fused attention, KV-cache blocks, MoE routers). This is the **plugin-centric** turn in its clearest form, and it is an active, ongoing effort — every runtime already ships and maintains its *own* fused-attention module rather than sharing one:

- **ONNXRuntime** — `Attention` / `MultiHeadAttention` / `GroupQueryAttention` contrib ops: [`contrib_ops/cpu/bert`](https://github.com/microsoft/onnxruntime/tree/v1.27.1/onnxruntime/contrib_ops/cpu/bert)
- **TensorRT-LLM** — [`gptAttentionPlugin`](https://github.com/NVIDIA/TensorRT-LLM/tree/v1.2.1/cpp/tensorrt_llm/plugins/gptAttentionPlugin), over [`cpp/tensorrt_llm/kernels`](https://github.com/NVIDIA/TensorRT-LLM/tree/v1.2.1/cpp/tensorrt_llm/kernels)
- **OpenVINO** — [`ScaledDotProductAttention`](https://docs.openvino.ai/2026/documentation/openvino-ir-format/operation-sets/operation-specs/sequence/scaled-dot-product-attention.html)

Attention is the sharpest illustration, because everyone agrees on the mathematics and no two runtimes agree on the boundary — specifically, on where the KV cache lives relative to the op:

- **Inside the op.** `com.microsoft.GroupQueryAttention` keeps the cache as graph tensors — `past_key`/`past_value` in, `present_key`/`present_value` out — but when past and present are *the same* tensor it is sized to `max_sequence_length` and the kernel appends in place. It also takes `seqlens_k` and `total_sequence_length`, and with `do_rotary` plus `cos_cache`/`sin_cache` applies RoPE in the same kernel. The host allocates the buffer; the op decides what happens to it, and how far into it "now" is.
- **Beside the op.** TensorRT-LLM's `gptAttentionPlugin` has no KV tensors in that sense: with paged KV the cache is a pool of blocks handed out per request by a cache manager, and the plugin is passed the block offsets and host-side metadata needed to find them. Shape and behaviour are fixed in plugin *fields* at build time rather than expressed in a portable signature. The runtime owns the memory and the bookkeeping.
- **Outside the graph's I/O.** OpenVINO's `ScaledDotProductAttention` is the mathematics alone — `query`/`key`/`value`, optional mask and scale, a `causal` flag. The cache is neither an input nor an output: it is *state*, declared as `ReadValue`/`Assign` pairs on a `Variable` and carried between `infer()` calls by the `InferRequest`, reachable only through `query_state()`. That is what `optimum-intel` exports by default; serving stacks then rewrite it into something else again, `ov::pass::SDPAToPagedAttention` trading the state for a 28-input `PagedAttentionExtension` and block tables. One vendor, two incompatible KV contracts, chosen by the export pipeline rather than by the model.

Same mathematics, three different divisions of labour — ORT hands the op one contiguous buffer per sequence, TensorRT-LLM hands it nothing and lets a cache manager own the memory, OpenVINO hands it nothing either and keeps the cache as model state the application never passes. The incompatibility is not naming, it is **who owns what**: who allocates the cache, who advances the position, who tracks sequence length. A converter cannot mechanically rewrite one into another, because what it would be rewriting is not the same node; it is a different answer to where the runtime ends and the graph begins.

So the module level is where the performance lives, and — being plugin-centric — where portability stops. A graph containing `GroupQueryAttention` is an *ONNXRuntime* graph, not an ONNX one. Decompose it back to `MatMul`/`Softmax`/`Concat` and the file translates again, at the price of the concatenate-and-copy KV growth, the fused kernel, and often the accelerator partition with it.

With no standard at that level, model publishers, agent frameworks, and distribution hubs each re-implement the same per-target conversion glue. Every converter below reads standard ONNX and nothing else, so an export falls off this table the moment it uses the plugin that made the model fast:

| Path            | Converter                                                                                               |
| --------------- | ------------------------------------------------------------------------------------------------------- |
| ONNX → QNN      | [ONNX2DLC](https://docs.qualcomm.com/doc/80-63442-10/topic/converters.html#onnx-conversion)             |
| ONNX → TensorRT | [ONNX2TRT](https://github.com/onnx/onnx-tensorrt)                                                       |
| ONNX → OpenVINO | [ONNX2OVIR](https://docs.openvino.ai/2026/openvino-workflow/model-preparation/convert-model-onnx.html)  |
| ONNX → IREE     | [ONNX2MLIR](https://iree.dev/guides/ml-frameworks/onnx/)                                                |

The choice as it stands is therefore between a fast export bound to one runtime and a portable export that gave up the reason you exported at all. **`hf2mobile` targets exactly that middle ground.**

### Our goal: place the seam deliberately

The goal is not merely to emit a graph a mobile runtime will accept — it is to emit one that lands on the right silicon. A phone has both an NPU and a CPU, so the interesting question is never which of the two to pick; it is **where to put the seam between them**. Compute-heavy, statically-shaped blocks — the MLP stack, the projections — belong on the NPU, because that is where the TOPS are. Control flow, sampling, dynamic-shaped glue, the KV cache and the attention that reads it, and any operator the accelerator has no kernel for belong on the CPU, because that is where the flexibility is.

Left alone, that boundary is drawn by whatever the converter happens to claim, and a single unsupported op in the wrong place can strand an entire block off the accelerator. `hf2mobile` treats the split as an **export-time** decision: shape the graph so the NPU can take the parts that pay for themselves, and let the CPU take the rest by design rather than by accident.

Placing the seam is also how the trade above stops being one. It only bites when the module-level knowledge — *this is attention, this is the KV cache, this is the decode loop* — is thrown away at export and has to be recovered by pattern-matching flat operators. `hf2mobile` keeps that knowledge until the last moment and spends it per target: the same traced module becomes a fused plugin where one exists, a decomposition where the accelerator needs plain operators, and the parts that genuinely cannot live in a DAG — sampling, stop conditions, the loop — get [baked into the graph](#comhf2mobilesamplelogits) instead of deferred to a side-car config the next toolchain will not read. Plugin-centric becomes a per-export choice, not a property of the file you are stuck with.

</details>

---

## How hf2mobile tackles it

<details>
<summary>Click to expand</summary>

`hf2mobile` is a framework built **on top of the HuggingFace `transformers` library**. The middle ground the motivation ends on has a concrete location — the **module boundary** — and everything below follows from keeping it. Instead of lowering a model to a flat operator graph and hoping each backend copes, it works one level up:

1. **Trace at the *module* level, not the operator level.** The model is traced as its semantic building blocks — attention, RoPE, RMSNorm, the causal LM head — rather than as an undifferentiated soup of `MatMul` / `Mul` / `Softmax`. An `Attention` `torch.nn.Module` leaves the trace as **one node**, with the module boundary still intact.
2. **Expand the module level nodes — per target, and per strategy.** A traced module knows how to emit the right subgraph for its `--target`: a fused GQA/attention plugin for a runtime that supports it, a sliding-window mask template for another, or a single-head decomposition for an NPU that lacks fused attention. Same node, different expansions, decided at export time.
3. **Generalize by extending the existing exporter api, not by rewriting the graph lazily.** Supporting a new architecture or a new hardware/runtime target means contributing a small, self-contained exporter — the framework handles the rest.

`transformers` is what makes that practical. Every architecture is defined against the same template — `Qwen3RMSNorm`, `LlamaAttention`, `Gemma3RotaryEmbedding` — so the semantic boundaries are already drawn, consistently, by the library the models are published with. `hf2mobile` keys on exactly that: a plugin is selected by class-name *suffix*, so one `Attention` exporter covers every architecture that follows the convention instead of one exporter per model, and the traced node keeps the module's full identity (`Qwen3Attention`, `model.layers.0.self_attn`) all the way into the graph.

The rest is extension rather than replacement:

- **Reused** — `transformers` module definitions as the semantic baseline; ORT contrib ops (`GroupQueryAttention`, RMSNorm / RoPE fusions); ORT execution providers and `EPContext` for placement.
- **Added** — the module-level tracer and plugin registry, the per-target expanders, `SampleLogits` and the in-graph EOS constant, and the Rust runtime that drives them.

### The plugin node is the unit of control

Holding `Attention` as a single node until the last step is what makes the divergence *yours*. The HuggingFace module is the baseline, and every expansion is a deliberate, legible departure from it — chosen per target and per strategy, rather than being whatever a converter's pattern matcher happened to recognize in a graph that had already been flattened. The user decides how far the ported graph moves from the baseline, and where.

It is also what survives the trade the motivation describes. The portable artifact is not the exported file — a graph carrying `GroupQueryAttention` is an ONNXRuntime graph, and no amount of care changes that. The portable artifact is the module-level description one step upstream of it, which every target's file is generated *from*. Being plugin-centric stops being a property you are stuck with, because what you keep is the description rather than any one expansion of it.

### One graph, more than one backend

That control is what makes the NPU/CPU seam expressible in a single file, and the decision that sets everything else is what to do about the KV cache — the hardest piece, and the one the motivation above spends three bullets on.

The choice here is to **take an ORT-executable graph as the baseline** and let ORT own the cache, rather than adopt whichever cache mechanism a given backend prefers. The attention node expands into `GroupQueryAttention`, whose `past_key` / `present_key` buffers are ordinary graph tensors: the state stays visible in the IR instead of disappearing into a side-car library, and the same handling holds whether the node runs on the CPU EP or the GPU EP. We start with the **CPU EP**, being the more generic of the two.

There is a second reason attention has to sit on that side of the seam: the cache is what makes its shapes move. `total_sequence_length` grows by one on every decode step, so the tensors attention reads are a different size each time it runs. An NPU gets its efficiency from compiling a partition **ahead of time** against fixed shapes — a dimension that only becomes known per step is exactly what it cannot plan for. CPU and GPU execution providers resolve shapes at run time and simply absorb the growth. So the dynamic half of the model belongs where dynamism is free, and it belongs there for the same reason it belongs to ORT: it is the same half.

The line is drawn between *kinds* of attention, not across attention as a whole. Encoder and vision attention carry no KV cache and run at a sequence length fixed by the input, so they are statically shaped like the projections around them and belong on the NPU — expanded there into whatever form the accelerator has kernels for, a single-head decomposition included. It is **decoder** attention, the one threading a growing cache from step to step, that stays CPU-side. Same traced node, two expansions, chosen by what the module actually does.

Everything else follows from that. With the baseline runtime holding the cache, what is left is compute-heavy and **statically shaped** — the QKV projections, the MLP stack, whose dimensions come from the config and do not move as the cache grows — which is precisely what an ahead-of-time compiling accelerator is built for, so that is what goes to the **NPU**. And ORT can carry both in one file: an `EPContext` node embeds an ahead-of-time compiled partition for another backend inside the same ONNX graph, so a single `.onnx` covers multi-device deployment. One graph, one session, mixed execution — the shape a mobile platform actually needs.

### A descriptive graph, and a generic runtime to read it

The exporter and the runtime are one deliverable, designed against each other. The graph carries the **description** — module identity survives the trace, cache slots are named, the sampling policy and the EOS ids are baked in — and the Rust runtime beside it ([CausalLM inference](#example-causallm-inference)) stays small and model-agnostic precisely because it can read all of that out of the file.

Attention shows the division. `GroupQueryAttention` owns the in-kernel cache append, so the export targets it directly; the loop around it — prefill, decode, stop — is host work by nature, and the runtime owns that. Driving it takes no per-model knowledge: cache slots are discovered by convention (`x` / `x_out`), stop tokens are read straight from the graph, and `SampleLogits` returns a token id rather than a megabyte-wide logits row. One loop runs every exported model, and it is small enough to cross-compile for a phone.

That is what keeping the description in the file buys. The model's semantics stay in one artifact, in the IR, where the next tool can see them — instead of spread across a contrib op, a side-car config and a session API, each holding a piece of the model on its own terms.

The result is a single, extensible pipeline that produces correctly-specialized graphs per target — **ORT** today — while keeping the shared, model-semantic structure in one place.

</details>

---

## How it works

<details>
<summary>Click to expand</summary>

### EXAMPLE: CausalLM exporter
Every supported model inherits `CausalLMExporter` (`src/hf2mobile/exporter/causallm.py`), which drives a five-stage export. A single run produces **two graphs** — a **prefill** case (processes the full prompt) and a **generation** case (single-token decode with KV cache in/out) — because those are the two distinct shapes a decoder actually runs at inference time.

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

### EXAMPLE: CausalLM inference

The exported graph is only half the deliverable; a runtime has to load it. `hf2mobile` ships two Rust crates that share one source file:

| Crate | Artifact | Role |
| ----- | -------- | ---- |
| `src/onnx_inferencer/` | `hf2mobile._ortrs_binding` (a Python extension module, built by maturin) | *Drives* ONNXRuntime — session setup, KV cache, prefill/decode loop, tokenizer, timing. Backs `python -m hf2mobile.infer`. |
| `src/onnx_plugins/` | `libhf2mobile_plugins.so` | *Driven by* ONNXRuntime — a custom-op library exporting the C `RegisterCustomOps` entry point, loadable from Python, C++ or an Android app. |

They are separate crates (and separate cargo workspaces) because they need opposite `ort` configurations: the runtime dlopens onnxruntime, the plugin is already running inside it. But `sample_logits.rs` is compiled into **both**, so the token a mobile runtime picks and the token the dev runtime picks come from one definition.

#### `com.hf2mobile:SampleLogits`

```text
logits [1, L, vocab]  --SampleLogits(top_k, top_p, temperature)-->  sampled_token [1, 1] int32
```

The sampling policy is a set of node attributes baked in at postprocess time, not an argument passed per call. That is the whole point: a 262k-wide fp32 logits row is 1 MB per token, and copying it out of ONNXRuntime only to reduce it to one integer is the most expensive thing a decode step does that isn't arithmetic. `--temp 0` bakes in greedy.

`python -m hf2mobile.postprocess` reads the policy from the export's `generation_config.json` / `tokenizer_config.json` (`do_sample: false` ⇒ greedy), so by default the graph decodes the way `model.generate` would have.

#### Stop tokens travel with the graph

The EOS ids go into the graph as a floating `Constant` named `hf2mobile_EOS_tokens`. It is the one thing a decode loop needs that is neither an input nor an output, and putting it in the model means a runtime reads it from the file it already has to open — no json to parse, no argument nobody remembers to pass. ONNXRuntime prunes the node before a session exists (it has no consumers), so the runtime walks the protobuf wire format directly to read it.

#### What the runtime does

- **Zero-copy KV cache** — cache tensors stay ORT-side across decode steps, bound by borrowed handle and taken back by handle. Cost per token is a few pointer writes, not tens of MB of `memcpy`. Cache slots are *discovered* (any input `x` with a matching output `x_out`), so layer count and naming can change without touching Rust.
- **Turn-based generation** — a turn spans calls, so `generate(..., num_generation=1)` polls one token at a time and still pays for prefill exactly once.
- **One graph, two jobs** — the dynamic-`L` graph serves both prefill (`L = prompt length`, empty cache) and decode (`L = 1`).
- **`DEBUG=1`** — also writes the Level3-optimized graph as `<model>.ort` and a `chrome://tracing` profile.

```python
from hf2mobile.inference import CausalLMInferencer

lm = CausalLMInferencer("inference.onnx", "tokenizer.json")
text, (ttft_s, tps) = lm.generate("Where is Paris?", num_generation=64)
```

</details>

---


## Requirements

- Python **>= 3.12**
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- A **Rust toolchain** + [`maturin`](https://www.maturin.rs/) — `hf2mobile` is a mixed Rust/Python project, so a build compiles the `_ortrs_binding` extension module

Everything above comes from `flake.nix`; nothing needs to be installed globally:

```bash
nix develop      # rustc, cargo, rustfmt, rust-analyzer, maturin, python312, uv — and activates .venv
```

The package pins CPU-only PyTorch wheels by default (see the `[tool.uv.index]`
block in `pyproject.toml`). To use CUDA wheels instead, remove that block.

---

## Install

### Default

**From a built wheel** — the wheel carries the compiled extension, so no Rust toolchain is needed:

```bash
uv pip install dist/hf2mobile-0.1.0-cp312-abi3-linux_x86_64.whl
```

**From source (development):**

```bash
uv sync                     # Python dependencies
maturin develop --release   # compile the Rust extension into the venv  (~45s cold, ~15s warm)
```

Re-run `maturin develop --release` after any change under `src/onnx_inferencer/` — Python imports the *installed* `.so`, so a bare `cargo build` will not be picked up.

### Build from scratch

**The wheel** (contains the compiled extension, so it is platform-specific — `abi3-py312`, one wheel per OS/arch):

```bash
maturin build --release -o dist/
```

**The ONNXRuntime plugin library** (only needed to load an `hf2mobile` graph from a runtime *other* than `hf2mobile.infer`, which registers the operator in-process):

```bash
cargo build --release --manifest-path src/onnx_plugins/Cargo.toml
# -> src/onnx_plugins/target/release/libhf2mobile_plugins.so
```

```python
import onnxruntime as ort

opts = ort.SessionOptions()
opts.register_custom_ops_library("src/onnx_plugins/target/release/libhf2mobile_plugins.so")
session = ort.InferenceSession("inference.onnx", opts)
```

---

## Usage (CLI)

<details>
<summary>Click to expand</summary>

Three commands, run in order. Each takes the *export directory* the previous one produced.

### 1. `hf2mobile.export` — HF → ONNX

```bash
python3 -m hf2mobile.export {HF repo id} --target {ORT|QNN}
# OR, via the installed entry point:
hf2mobile-export {HF repo id} --target {ORT|QNN}
```

| Argument         | Description                                                              | Default |
| ---------------- | ----------------------------------------------------------------------- | ------- |
| `repo_id`        | Hugging Face repo id of the model to export.                            | —       |
| `-t`, `--target` | Export target: `ORT` or `QNN`. Graph topology may change based on it.   | `ORT`   |
| `--export_dtype` | `float32` / `float16` / `bfloat16`. Casts the weights before tracing, fixing the exported graph's dtype. | the model's own dtype |
| `--debug`        | Force a tiny random-init model (2 layers) + DEBUG logging for a fast smoke test. Also keeps the intermediate graphs. | off |

Output lands in a timestamped directory named `{date}__{target}__{model}`:

```
2026-08-01__ORT__Qwen-Qwen3-0.6B/
├── case2.onnx + case2.onnx.data   the dynamic-L generation graph (serves prefill and decode both)
├── tokenizer.json                 handed to the runtime, which does all the tokenizing
├── tokenizer_config.json          the chat template
└── generation_config.json         the sampling policy + EOS ids that stage 2 reads
```

Two cases are traced — prefill (`case1`) and generation (`case2`) — but the ORT deliverable is the single dynamic-`L` generation graph that serves both, so `case1.onnx` is deleted at the end of the export. `--debug` keeps a `debug__case1.onnx` / `debug__case2.onnx` snapshot of each, plus the standalone module subgraphs.

### 2. `hf2mobile.postprocess` — bake in the decode policy

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

### 3. `hf2mobile.infer` — run it

```bash
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

### Examples

Supported architectures
- `LlamaForCausalLM`
- `Qwen2ForCausalLM`
- `Qwen3ForCausalLM`
- `Gemma3ForCausalLM`

---
```bash
# Gemma 3 (270M), end to end
python3 -m hf2mobile.export google/gemma-3-270m-it --target ORT --export_dtype float32
python3 -m hf2mobile.postprocess 2026-08-01__ORT__google-gemma-3-270m-it
python3 -m hf2mobile.infer 2026-08-01__ORT__google-gemma-3-270m-it --prompt "Where is Paris?"

# Force greedy decoding regardless of what the model's config says
python3 -m hf2mobile.postprocess 2026-08-01__ORT__google-gemma-3-270m-it --temp 0

# Fast smoke test with a tiny random-init model (no real weights downloaded)
python3 -m hf2mobile.export meta-llama/Llama-3.2-1B-Instruct --target ORT --debug
```

</details>

---

## Roadmap

Grouped by the axis each item unblocks. Checked items ship in the current export path; the rest are ordered roughly by priority within each group.

### QNN target (Qualcomm HTP)

- [ ] **ONNX → DLC conversion & compilation.** CLI to lower the exported ONNX to a Qualcomm DLC and compile per HTP target (v79, v81, …), fetching the resulting `EPContext` `.so`. This is what makes a QNN export actually loadable on-device.
- [ ] **QNN-scheme quantization.** Quantize following the QNN quantization scheme (HTP prefers `u16` activations over `u8` — see the Quantization group below).

### ORT target (ONNXRuntime)

- [ ] **More contrib-op fusions.** Extend beyond RMSNorm/GQA to the remaining layer-norm family (`SkipLayerNormalization`, `SkipSimplifiedLayerNormalization`, `SimplifiedLayerNormalization`) so the graph maps onto ORT's optimized kernels.
- [ ] **MoE and LoRA plugins.** Add module exporters for data-dependent expert routing (MoE) and adapter weights (LoRA) — two of the module types the current static per-case export does not yet cover.
- [ ] **Per-token control flow.** Module exporters for Mixture-of-Depths and early-exit decoders, where whether a block runs at all is decided per token — the one axis in the motivation with no de facto answer in any runtime today.
- [ ] **NPU/CPU EP partition.** Assign the fused attention node to the CPU EP and the statically-shaped blocks (QKV projections, MLP) to the NPU EP within a single session, so one graph covers both backends.
- [ ] **Multimodal support.** Extend beyond text-only causal LMs to vision-language models (`Qwen2.5-VL`, `Phi-4-multimodal`, …): a separately traced vision encoder feeding a decoder whose sequence length is set by the input image.
- [ ] **ORT-scheme quantization.** Quantize following the ORT quantization scheme.

### Quantization

Targeting `u16` activations for QNN instead of `u8`, to preserve accuracy on HTP.

- [ ] **SmoothQuant (`u8s8`).** Migrate activation outliers into weights (fold diagonal matrix) so both sides quantize cleanly.
- [ ] **SpinQuant / QuaRot (`f16s8`).** Rotation-based outlier suppression (hadamard matrix), starting with the `R1` rotation only(WoQ) before adding the rest.
- [ ] **AWQ / GPTQ (`f16s8`).** Weight-only PTQ methods for the weight-quantized path.

### Runtime & benchmarking

- [x] **Host-CPU runtime.** Rust ONNXRuntime engine (`hf2mobile._ortrs_binding`) with a zero-copy KV cache and an in-graph `SampleLogits` operator; reports TTFT and TPS per run.
- [x] **Loadable custom-op library.** `libhf2mobile_plugins.so` — the same operator, as a `.so` any ONNXRuntime binding can register.
- [ ] **On-device Android execution.** Cross-compile both crates for `aarch64-linux-android` so an exported model runs end-to-end on device.
- [ ] **Benchmark harness.** Extend the per-run TTFT/TPS numbers into peak memory and a comparison across targets.
- [ ] **More logits dtypes.** `SampleLogits` has an fp32 kernel only; fp16 needs one registration each side.
- [ ] **Regression testing (pytest).** Extend the per-architecture suite under `tests/`, including numerical parity of the exported graph against the `transformers` baseline it was traced from.
