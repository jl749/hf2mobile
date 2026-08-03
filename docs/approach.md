# 🧩 Approach

*The module boundary as the unit of export.* — [back to README](../README.md)

---

`hf2mobile` is built **on top of HuggingFace `transformers`**. The [motivation](motivation.md) ends in a dilemma: a fast export bound to one runtime, or a portable export that gave up the reason you exported at all. The way out is to work one level up from the flat operator graph — at the **module boundary**:

1. **Trace at the *module* level.** The model is recorded as its semantic building blocks — attention, RoPE, RMSNorm, the LM head — rather than a soup of `MatMul` / `Mul` / `Softmax`. `transformers` makes this practical: every architecture follows the same template (`Qwen3RMSNorm`, `LlamaAttention`, `Gemma3RotaryEmbedding`), so the boundaries are already drawn by the library the models ship with. An `Attention` module enters the trace as **one node**, identity intact (`Qwen3Attention`, `model.layers.0.self_attn`). A module held whole this way — one node in the graph — is what this documentation calls a **plugin**.
2. **Expand those nodes per target, per strategy.** The same node emits a fused GQA plugin for one runtime, a sliding-window mask template for another, or a single-head decomposition for an NPU with no fused attention — decided at export time.
3. **Extend the exporter API, don't pattern-match the graph.** Plugins are selected by class-name *suffix*, so one `Attention` exporter covers every architecture following the convention. New architecture or target = one small exporter in `src/hf2mobile/exporter/`.

Here is what that trace actually is, on a two-layer Gemma 3 export:

<p align="center">
  <img src="2_module_level_postprocessed_graph.svg" width="280" alt="Module-level trace of a two-layer Gemma 3 decoder: Gemma3Attention and Gemma3RotaryEmbedding survive as single nodes, KV cache exposed as named graph I/O">
</p>

One `Gemma3Attention` and one `Gemma3RotaryEmbedding` node per layer — class name carried into the graph — with the KV cache exposed as named I/O (`past_keys_0` in, `past_keys_0_out` out) rather than hidden inside a kernel. Nothing about *how* attention runs has been decided at this point: the node is a name and a signature, and that is the baseline every target's file is expanded from. The one exception is `SlidingWindowMask` — already a target-specific expansion, the kind step 2 is about.

## EXAMPLE: placing the seam — one graph, more than one backend

A phone has both an NPU and a CPU, so the question is never which to pick — it is **where to put the seam**. **Shape decides**: an NPU EP earns its efficiency by compiling **ahead of time**, so every dimension must be known at build time, while a CPU EP resolves shapes at runtime.

Statically-shaped compute goes to the NPU — QKV projections, the MLP stack, where the TOPS are. Everything shape-dependent stays on the CPU: sampling, control flow, any op the accelerator lacks a kernel for. That puts the line between *kinds* of attention, not across attention as a whole:

| Attention | Shape | Goes to | Expanded into |
| --------- | ----- | ------- | ------------- |
| **Encoder / vision** | no cache, fixed sequence length — as static as the projections around it | **NPU** | whatever kernels the accelerator has |
| **Decoder** | threads a KV cache whose `total_sequence_length` grows by one per step — changes with the input data, so an AOT compiler cannot plan for it | **CPU** | `GroupQueryAttention`, so ORT owns the in-kernel cache append and the state stays visible in the IR as ordinary `past_key` / `present_key` tensors |

Same traced node, two expansions, chosen by what the module does. Left to the converter — onnx2trt, onnx2dlc, and the rest — the boundary lands wherever it happens to claim, and one unsupported op can strand a whole block off the accelerator. `hf2mobile` makes the split an **export-time** decision instead.

Both halves still ship as one file: an `EPContext` node embeds the compiled NPU partition inside the same ONNX graph. One graph, one session, mixed execution.

## The plugin node is the unit of control

Nothing above required a new file format — both expansions came out of one traced `Attention` node, and neither is privileged. That is why the node is held whole until the last step: the HuggingFace module stays the baseline, and every expansion is a deliberate departure from it, not whatever a converter's pattern matcher recognized in an already-flattened graph.

> The portable artifact is the module-level trace, not any file it produces. A graph carrying a runtime-specific plugin or fusion pattern is bound to that runtime — but every target's file can be translated from the same module-level expansion.

The trace is `hf2mobile`'s own representation — no other tool reads it — so this collapses the per-*(hardware × runtime)* work into one place rather than standardizing it away. The description is what gets versioned; every runtime-bound file is regenerated from it rather than maintained.

Being plugin-centric stops being a trap once what you keep is the description, not any one expansion of it.

## A descriptive graph, and a generic runtime to read it

The exporter and the runtime are one deliverable, designed against each other. Module identity is the *exporter's* lever. What the **runtime** needs is narrower but just as explicit: cache slots are named, and the sampling policy and EOS ids are [baked in](../README.md#2-hf2mobilepostprocess--bake-in-the-decode-policy). The Rust runtime beside it ([CausalLM inferencer](how-it-works.md#example-causallm-inferencer)) reads all of that out of the file, which is what keeps it small and model-agnostic. The semantics live in **one artifact, in the IR**, instead of spread across a contrib op, a side-car config and a session API.

Attention shows the division: `GroupQueryAttention` owns the in-kernel cache append, so the export targets it directly; the loop around it — prefill, decode, stop — is host work, and the runtime owns that. One loop runs every exported model, small enough to cross-compile for a phone.

## What is reused, what is added

**Reused** — `transformers` module definitions, ORT contrib ops (`GroupQueryAttention`, RMSNorm/RoPE fusions), ORT execution providers and `EPContext`.
**Added** — the module-level tracer and plugin registry, the per-target expanders, `SampleLogits` and the in-graph EOS constant, and the Rust runtime.

---

**Next:** [🔧 How it works](how-it-works.md) — the export stages, and the Rust runtime that runs the result on a phone.
