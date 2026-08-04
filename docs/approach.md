# 🧩 Approach

*The module boundary as the unit of export.* — [back to README](../README.md)

---

`hf2mobile` is built **on top of HuggingFace `transformers`**. The [motivation](motivation.md) ends in a dilemma: a fast export bound to one runtime, or a portable export that gave up the reason you exported at all. The way out is to work one level up from the flat operator graph — at the **module boundary**:

1. **Trace at the *module* level.** The model is recorded as its semantic building blocks — attention, RoPE, RMSNorm, the LM head — rather than a soup of `MatMul` / `Mul` / `Softmax`. `transformers` makes this practical: every architecture follows the same template (`Qwen3RMSNorm`, `LlamaAttention`, `Gemma3RotaryEmbedding`), so the boundaries are already drawn by the library the models ship with. An `Attention` module enters the trace as **one node**, identity intact (`Qwen3Attention`, `model.layers.0.self_attn`). A module held whole this way — one node in the graph — is what this documentation calls a **plugin**: the same object [motivation](motivation.md) called a trap, under a different owner.
2. **Expand those nodes per target, per strategy.** The same node emits a fused GQA plugin for one runtime, a sliding-window mask template for another, or a single-head decomposition for an NPU with no fused attention — decided at export time.
3. **Extend the exporter API, don't pattern-match the graph.** Plugins are selected by class-name *suffix*, so one `Attention` exporter covers every architecture following the convention. New architecture or target = one small exporter in `src/hf2mobile/exporter/`.

Here is what that trace actually is, on a two-layer Gemma 3 export:

<p align="center">
  <img src="2_module_level_postprocessed_graph.svg" width="280" alt="Module-level trace of a two-layer Gemma 3 decoder: Gemma3Attention and Gemma3RotaryEmbedding survive as single nodes, KV cache exposed as named graph I/O">
</p>

One `Gemma3Attention` and one `Gemma3RotaryEmbedding` node per layer — class name carried into the graph — with the KV cache exposed as named I/O (`past_keys_0` in, `past_keys_0_out` out) rather than hidden inside a kernel. Nothing about *how* attention runs is decided yet: the node is a name and a signature, and that is the baseline every target's file expands from. `SlidingWindowMask` is the exception — the HuggingFace model computes it, so it belongs to step 1, but its shape depends on `Lq` and `Lkv` at decode time, so it is emitted with the step 2 expansions rather than frozen into the trace.

## EXAMPLE: placing the seam — one graph, more than one backend

A phone has both an NPU and a CPU, so the question is never which to pick — it is **where to put the seam**. Left to the converter — onnx2trt, onnx2dlc, and the rest — the boundary lands wherever the pattern matcher happens to claim, and one unsupported op can strand a whole block off the accelerator. Held as plugin nodes, the split falls on boundaries you named, at **export time**.

**Shape decides where it falls.** An NPU EP earns its efficiency by compiling **ahead of time**, so every dimension must be known at build time; a CPU EP resolves shapes at runtime. Statically-shaped compute goes to the NPU — QKV projections, the MLP stack, where the TOPS are. Everything shape-dependent stays on the CPU, along with sampling, control flow, and any op the accelerator has no kernel for. That puts the line between *kinds* of attention, not across attention as a whole:

| Attention | Shape | Goes to | Expanded into |
| --------- | ----- | ------- | ------------- |
| **Encoder / vision** | no cache, fixed sequence length — as static as the projections around it | **NPU** | whatever kernels the accelerator has |
| **Decoder** | threads a KV cache whose `total_sequence_length` grows by one per step | **CPU** | `GroupQueryAttention`, so ORT owns the in-kernel cache append and the state stays visible in the IR as ordinary `past_key` / `present_key` tensors |

Two nodes of the same kind, expanded differently because of what each one does. The decoder row is this target's call, not a law — pad the cache to a fixed length and it compiles AOT too (what Qualcomm's GENIE SDK does), buying static shape with wasted compute and a capped context. Either way the decision is made in the exporter, against a node you can name.

Both halves still ship as one file: an `EPContext` node embeds the compiled NPU partition inside the same ONNX graph. One graph, one session, mixed execution.

## The plugin node is the unit of control

Nothing above required a new file format — both expansions came out of the same kind of traced `Attention` node, and neither is privileged. That is why the node is held whole until the last step: the HuggingFace module stays the baseline, and every expansion is a deliberate departure from it, not whatever a converter's pattern matcher recognized in an already-flattened graph.

Attention is one of the four axes [motivation](motivation.md) opened; the same lever reaches the others. An MoE block held whole is one description that expands into `com.microsoft.MoE`, a dense-masked fallback, or a gather into a compact batch — three targets, not three rewrites. A LoRA adapter attaches at a module boundary by definition: merged into the weights for one target, left as a graph input for another. Mixture-of-Depths is the honest limit — a trace records the execution it saw, so per-token skipping survives no export at any boundary — but the node still names the block a custom kernel would have to claim, which a flattened graph does not.

> The portable artifact is the module-level trace, not any file it produces. A graph carrying a runtime-specific plugin or fusion pattern is bound to that runtime — but every target's file can be translated from the same module-level expansion.

That resolves the opening dilemma — not by making the fast export portable, but by keeping the two apart. Every emitted file stays bound to its runtime; the trace is what moves. The cost: the trace is `hf2mobile`'s own representation, which no other tool reads, so the per-*(hardware × runtime)* work is collapsed into one place rather than standardized away. The description is what gets versioned; every runtime-bound file is regenerated from it rather than maintained.

Being plugin-centric stops being a trap once what you keep is the description, not any one expansion of it.

## A descriptive graph, and a generic runtime to read it

The exporter and the runtime are one deliverable, designed against each other. Module identity is the *exporter's* lever; what the **runtime** needs is narrower but just as explicit — cache slots are named, and the sampling policy and EOS ids are [baked in](../README.md#2-hf2mobilepostprocess--bake-in-the-decode-policy). The Rust runtime ([CausalLM inferencer](how-it-works.md#example-causallm-inferencer)) reads all of that out of the file, which is what keeps it small and model-agnostic. The semantics live in **one artifact, in the IR**, instead of spread across a contrib op, a side-car config and a session API.

Attention shows the division: `GroupQueryAttention` owns the in-kernel cache append, so the export targets it directly; the loop around it — prefill, decode, stop — is host work, and the runtime owns that. One loop runs every exported model, small enough to cross-compile for a phone.

## What is reused, what is added

**Reused** — `transformers` module definitions, ORT contrib ops (`GroupQueryAttention`, RMSNorm/RoPE fusions), ORT execution providers and `EPContext`.
**Added** — the module-level tracer and plugin registry, the per-target expanders, `SampleLogits` and the in-graph EOS constant, and the Rust runtime.

---

**Next:** [🔧 How it works](how-it-works.md) — the export stages, and the Rust runtime that runs the result on a phone.
