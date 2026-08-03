# 🧩 Approach

*The module boundary as the unit of export.* — [back to README](../README.md)

---

`hf2mobile` is built **on top of HuggingFace `transformers`**. The [motivation](motivation.md) leaves a choice between a fast export bound to one runtime and a portable export that gave up the reason you exported at all. The way out is to work one level up from the flat operator graph — at the **module boundary**:

1. **Trace at the *module* level.** The model is recorded as its semantic building blocks — attention, RoPE, RMSNorm, the LM head — rather than a soup of `MatMul` / `Mul` / `Softmax`. `transformers` makes this practical: every architecture follows the same template (`Qwen3RMSNorm`, `LlamaAttention`, `Gemma3RotaryEmbedding`), so the boundaries are already drawn by the library the models ship with. An `Attention` module appears in the trace as **one node** — a **plugin**, the name used throughout for a module held whole this way, since a single node is what it becomes in the graph — identity intact (`Qwen3Attention`, `model.layers.0.self_attn`).
2. **Expand those nodes per target, per strategy.** The same node emits a fused GQA plugin for one runtime, a sliding-window mask template for another, or a single-head decomposition for an NPU with no fused attention — decided at export time.
3. **Extend the exporter API, don't pattern-match the graph.** Plugins are selected by class-name *suffix*, so one `Attention` exporter covers every architecture that follows the convention. New architecture or new target = one small, self-contained exporter.

Here is what that trace actually is, on a two-layer Gemma 3 export:

<p align="center">
  <img src="2_module_level_postprocessed_graph.svg" width="280" alt="Module-level trace of a two-layer Gemma 3 decoder: Gemma3Attention and Gemma3RotaryEmbedding survive as single nodes, KV cache exposed as named graph I/O">
</p>

One `Gemma3Attention` and one `Gemma3RotaryEmbedding` node per layer — class name carried into the graph — with the KV cache exposed as named I/O (`past_keys_0` in, `past_keys_0_out` out) rather than hidden inside a kernel. Nothing about *how* attention runs has been decided at this point: the node is a name and a signature, and that is the baseline every target's file is expanded from. `SlidingWindowMask` is the one exception, and the preview — the first per-target expansion, already attached by postprocess.

Step 2 is the part worth seeing before arguing about, so take it concretely first.

## EXAMPLE: placing the seam - one graph, more than one backend

A phone has both an NPU and a CPU, so the question is never which to pick — it is **where to put the seam**. **Shape decides**: an NPU earns its efficiency by compiling **ahead of time**, so every dimension must be known at build time, while a CPU EP resolves shapes at run time.

| Side | Takes | Because |
| ---- | ----- | ------- |
| **NPU** | statically-shaped compute — QKV projections, the MLP stack, **encoder / vision attention** | fixed shapes are what an AOT compiler can plan for, and that is where the TOPS are |
| **CPU** | everything shape-dependent — **decoder attention** and the KV cache it reads, sampling, control flow, and any op the accelerator lacks a kernel for | shapes resolve at run time, so dynamism is free |

The line therefore falls between *kinds* of attention, not across attention as a whole:

- **Encoder / vision attention** carries no cache and runs at a fixed sequence length, so it is as statically shaped as the projections around it → **NPU**, decomposed into whatever kernels the accelerator has.
- **Decoder attention** threads a KV cache whose `total_sequence_length` grows by one per step — a shape that changes with the input data, exactly what an AOT compiler cannot plan for → **CPU**, expanded into `GroupQueryAttention` so ORT owns the in-kernel cache append and the state stays visible in the IR as ordinary `past_key` / `present_key` tensors.

Same traced node, two expansions, chosen by what the module does. Left alone, that boundary is drawn by whatever the converter happens to claim, and one unsupported op can strand a whole block off the accelerator; `hf2mobile` makes the split an **export-time** decision instead. Today's exports target the **CPU EP** alone, the more generic side.

Both halves still ship as one file: an `EPContext` node embeds the compiled NPU partition inside the same ONNX graph. One graph, one session, mixed execution.

## Why the node is held to the last step

Nothing above required a new file format — both expansions came out of one traced `Attention` node, and neither is privileged. That is the whole reason the node is held whole until the last step: the HuggingFace module stays the baseline, and every expansion is a deliberate departure from it — not whatever a converter's pattern matcher recognized in an already-flattened graph.

> The portable artifact is the module-level trace, not any file it produces. A graph carrying a runtime-specific plugin or fusion pattern is bound to that runtime — but every target's file can be translated from the same module-level expansion.

This is not a standard restored. The trace is `hf2mobile`'s own representation and no other tool consumes it; nothing here hands the ecosystem an interchange format it was missing. What it does is collapse the per-*(hardware × runtime)* work into one place and make each cell an explicit choice — the description is what gets kept and versioned, and every runtime-bound file is downstream of it, regenerated rather than maintained.

Being plugin-centric stops being a trap once what you keep is the description rather than any one expansion of it.

## A descriptive graph, and a generic runtime to read it

The exporter and the runtime are one deliverable, designed against each other. Module identity is the *exporter's* lever; what the **runtime** needs is narrower and just as explicit — cache slots are named, the sampling policy and EOS ids are [baked in](../README.md#2-hf2mobilepostprocess--bake-in-the-decode-policy) — so the Rust runtime beside it ([CausalLM inferencer](how-it-works.md#example-causallm-inferencer)) stays small and model-agnostic by reading those out of the file. The semantics live in **one artifact, in the IR**, instead of spread across a contrib op, a side-car config and a session API.

Attention shows the division: `GroupQueryAttention` owns the in-kernel cache append, so the export targets it directly; the loop around it — prefill, decode, stop — is host work, and the runtime owns that. One loop runs every exported model, small enough to cross-compile for a phone.

## What is reused, what is added

**Reused** — `transformers` module definitions, ORT contrib ops (`GroupQueryAttention`, RMSNorm/RoPE fusions), ORT execution providers and `EPContext`.
**Added** — the module-level tracer and plugin registry, the per-target expanders, `SampleLogits` and the in-graph EOS constant, and the Rust runtime.

---

**Next:** [🔧 How it works](how-it-works.md) — the export stages, and the Rust runtime that runs the result on a phone.
