# 🧩 Approach

*The module boundary as the unit of export.* — [back to README](../README.md)

---

`hf2mobile` is built **on top of HuggingFace `transformers`**. The [motivation](motivation.md) leaves a choice between a fast export bound to one runtime and a portable export that gave up the reason you exported at all. The way out is to work one level up from the flat operator graph — at the **module boundary**:

1. **Trace at the *module* level.** The model is recorded as its semantic building blocks — attention, RoPE, RMSNorm, the LM head — rather than a soup of `MatMul` / `Mul` / `Softmax`. `transformers` makes this practical: every architecture follows the same template (`Qwen3RMSNorm`, `LlamaAttention`, `Gemma3RotaryEmbedding`), so the boundaries are already drawn by the library the models ship with. An `Attention` module leaves the trace as **one node**, identity intact (`Qwen3Attention`, `model.layers.0.self_attn`).
2. **Expand those nodes per target, per strategy.** The same node emits a fused GQA plugin for one runtime, a sliding-window mask template for another, or a single-head decomposition for an NPU with no fused attention — decided at export time.
3. **Extend the exporter API, don't pattern-match the graph.** Plugins are selected by class-name *suffix*, so one `Attention` exporter covers every architecture that follows the convention. New architecture or new target = one small, self-contained exporter.

**Reused** — `transformers` module definitions, ORT contrib ops (`GroupQueryAttention`, RMSNorm/RoPE fusions), ORT execution providers and `EPContext`.
**Added** — the module-level tracer and plugin registry, the per-target expanders, `SampleLogits` and the in-graph EOS constant, and the Rust runtime.

## The plugin node is the unit of control

Holding plugins such as `Attention` and `RotaryEmbedding` as a single node until the last step is what makes the divergence *yours*: the HuggingFace module is the baseline, and every expansion is a deliberate departure from it — not whatever a converter's pattern matcher recognized in an already-flattened graph.

> The portable artifact is the module-level trace, not any file it produces. A graph carrying a runtime-specific plugin or fusion pattern is bound to that runtime — but every target's file can be translated from the same module-level expansion.

Being plugin-centric stops being a trap once what you keep is the description rather than any one expansion of it.

## EXAMPLE: placing the seam - one graph, more than one backend

A phone has both an NPU and a CPU, so the question is never which to pick — it is **where to put the seam**. **Shape decides**: an NPU earns its efficiency by compiling **ahead of time**, so every dimension must be known at build time, while a CPU EP resolves shapes at run time.

| Side | Takes | Because |
| ---- | ----- | ------- |
| **NPU** | statically-shaped compute — QKV projections, the MLP stack, **encoder / vision attention** | fixed shapes are what an AOT compiler can plan for, and that is where the TOPS are |
| **CPU** | everything shape-dependent — **decoder attention** and the KV cache it reads, sampling, control flow, and any op the accelerator lacks a kernel for | shapes resolve at run time, so dynamism is free |

The line therefore falls between *kinds* of attention, not across attention as a whole:

- **Encoder / vision attention** carries no cache and runs at a fixed sequence length, so it is as statically shaped as the projections around it → **NPU**, decomposed into whatever kernels the accelerator has.
- **Decoder attention** threads a KV cache whose `total_sequence_length` grows by one per step — a shape that changes with the input data, exactly what an AOT compiler cannot plan for → **CPU**, expanded into `GroupQueryAttention` so ORT owns the in-kernel cache append and the state stays visible in the IR as ordinary `past_key` / `present_key` tensors.

Same traced node, two expansions, chosen by what the module does. Left alone that boundary is drawn by whatever the converter happens to claim, and one unsupported op can strand a whole block off the accelerator; `hf2mobile` makes the split an **export-time** decision instead. Today's exports target the **CPU EP** alone, the more generic side.

Both halves still ship as one file: an `EPContext` node embeds the compiled NPU partition inside the same ONNX graph. One graph, one session, mixed execution.

## A descriptive graph, and a generic runtime to read it

The exporter and the runtime are one deliverable, designed against each other. The graph carries the **description** — module identity survives the trace, cache slots are named, the sampling policy and EOS ids are [baked in](../README.md#2-hf2mobilepostprocess--bake-in-the-decode-policy) — so the Rust runtime beside it ([CausalLM inferencer](how-it-works.md#example-causallm-inferencer)) stays small and model-agnostic by reading it out of the file. The semantics live in **one artifact, in the IR**, instead of spread across a contrib op, a side-car config and a session API.

Attention shows the division: `GroupQueryAttention` owns the in-kernel cache append, so the export targets it directly; the loop around it — prefill, decode, stop — is host work, and the runtime owns that. One loop runs every exported model, small enough to cross-compile for a phone.

---

**Next:** [🔧 How it works](how-it-works.md) — the export stages, and the Rust runtime that runs the result on a phone.
