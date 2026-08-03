# 🧩 Approach

*The module boundary as the unit of export.* — [back to README](../README.md)

---

`hf2mobile` is built **on top of HuggingFace `transformers`**. The [motivation](motivation.md) ends in a dilemma: a fast export bound to one runtime, or a portable export that gives up the reason you exported at all. The way out is to work one level above the flat operator graph — at the **module boundary**:

1. **Trace at the *module* level.** The model is recorded as its semantic building blocks — attention, RoPE, RMSNorm, the LM head — rather than a soup of `MatMul`, `Mul`, and `Softmax`. `transformers` makes this practical: every architecture follows the same naming convention (`Qwen3RMSNorm`, `LlamaAttention`, `Gemma3RotaryEmbedding`), so the boundaries are already drawn. An `Attention` module enters the trace as a single node with its identity intact (`Qwen3Attention`, `model.layers.0.self_attn`) — a **plugin**, in this documentation's terms: the same object [motivation](motivation.md) called a trap, now under a different owner.
2. **Expand those nodes per target and strategy.** The same node emits a fused GQA plugin for one runtime, a sliding-window mask template for another, or a single-head decomposition for an NPU lacking fused attention — decided entirely at export time.
3. **Extend the exporter API, don't pattern-match the graph.** Plugins are selected by class-name *suffix*, so one `Attention` exporter covers every architecture that follows the convention. A new architecture or target needs only one small exporter in `src/hf2mobile/exporter/`.

Here is what that trace actually looks like for a two-layer Gemma 3 export:

<p align="center">
  <img src="2_module_level_postprocessed_graph.svg" width="280" alt="Module-level trace of a two-layer Gemma 3 decoder: Gemma3Attention and Gemma3RotaryEmbedding survive as single nodes, KV cache exposed as named graph I/O">
</p>

One `Gemma3Attention` and one `Gemma3RotaryEmbedding` node per layer — class names carried directly into the graph — and the KV cache exposed as named I/O (`past_keys_0` in, `past_keys_0_out` out) rather than buried inside a black-box kernel. Nothing about *how* attention runs is decided yet: the node is a name and a signature, the baseline every target's file expands from. `SlidingWindowMask` is the sole exception — HuggingFace computes it natively, so it belongs to step 1, but its shape depends on `Lq` and `Lkv` at decode time, so it is emitted with the step 2 expansions instead of frozen into the trace.

## EXAMPLE: placing the seam — one graph, multiple backends

A phone pairs a CPU with an NPU, so the question isn't which processor to use but **where to draw the seam**. Leave it to a converter like onnx2trt or onnx2dlc and the boundary falls wherever its pattern matcher lands — one unsupported op can strand an entire block off the accelerator. Named plugin nodes move that decision to **export time**, onto boundaries you choose.

**Shape decides where it falls.** An NPU Execution Provider (EP) compiles **ahead of time**, so every dimension must be known at build time; a CPU EP resolves shapes at runtime. Statically-shaped compute goes to the NPU — QKV projections and the MLP stack, where the TOPS live. Everything shape-dependent stays on the CPU, alongside sampling, control flow, and any operator the accelerator has no kernel for. The line falls between *types* of attention rather than through attention as a whole:

| Attention | Shape | Goes to | Expanded into |
| --------- | ----- | ------- | ------------- |
| **Encoder / vision** | no cache, fixed sequence length — as static as the surrounding projections | **NPU** | whatever native kernels the accelerator provides |
| **Decoder** | threads a KV cache whose `total_sequence_length` grows by one per step | **CPU** | `GroupQueryAttention`, ensuring ORT owns the in-kernel cache append while the state remains visible in the IR as ordinary `past_key` / `present_key` tensors |

Two nodes of the same family, expanded differently by how they behave. The decoder row is this target's choice, not a law — pad the cache to a fixed length and it compiles AOT too (much as Qualcomm's GENIE SDK does), buying static shape at the cost of wasted compute and a capped context window. Either way, the decision is made deterministically in the exporter, against a named node.

Both halves still ship as a single file: an `EPContext` node embeds the compiled NPU partition directly inside the parent ONNX graph. One graph, one session, mixed execution.

## The plugin node as the unit of control

Neither expansion required a new file format, and neither is privileged — both come from the same traced `Attention` node. That's why the node is held intact until the last step: the HuggingFace module remains the single source of truth, and every expansion is a deliberate departure from it rather than a guess by a pattern matcher over an already-flattened graph.

Attention was one of four axes in [motivation](motivation.md); the same mechanism reaches the rest. An MoE block held whole fans out to three targets from one trace — `com.microsoft.MoE`, a dense-masked fallback, or a gather into a compact batch. A LoRA adapter attaches at the module boundary: merged into the weights for one target, left as an explicit graph input for another. Mixture-of-Depths marks the limit — a trace records only the path it actually saw, so per-token skipping survives no static export, at any boundary. Even then the node preserves the block's identity, so a custom kernel can claim it long after a flattened graph loses the plot.

> The portable artifact is the module-level trace, not any file it produces. A graph carrying a runtime-specific plugin or fusion pattern is bound to that runtime — but every target's file can be generated from the same module-level expansion.

This resolves the opening dilemma — not by forcing a fast export to be portable, but by keeping the two domains separate. The trade-off is that the trace is `hf2mobile`'s own representation, which no other tool reads: per-*(hardware × runtime)* work is centralized rather than standardized away. The description is what gets versioned; every runtime-bound file is regenerated from it rather than maintained by hand.

Being plugin-centric stops being a trap once what you preserve is the description, not any single expansion of it.

## A descriptive graph, and a generic runtime to read it

Exporter and runtime are a single deliverable, designed hand-in-hand. Module identity is the *exporter's* lever; the **runtime** needs less, but just as explicitly — named cache slots, and the sampling policy and EOS ids [baked in](../README.md#2-hf2mobilepostprocess--bake-in-the-decode-policy). The Rust runtime ([CausalLM inferencer](how-it-works.md#example-causallm-inferencer)) reads all of it straight out of the file, which keeps that codebase small and completely model-agnostic. The core semantics live in **one artifact, cleanly represented in the IR**, instead of being fragmented across a contrib op, a side-car config file, and a sprawling session API.

Attention shows the division: `GroupQueryAttention` owns the in-kernel cache append, a clean target for export; the loop around it — prefill, decode, stop conditions — belongs entirely to the host. That loop is minimal and uniform across models, which makes it exceptionally easy to cross-compile for mobile devices.

## What is reused, what is added

**Reused** — `transformers` module definitions, ORT contrib ops (`GroupQueryAttention`, RMSNorm/RoPE fusions), ORT execution providers, and `EPContext`.
**Added** — the module-level tracer and plugin registry, per-target expanders, `SampleLogits` alongside the in-graph EOS constant, and the custom Rust runtime.

---

**Next:** [🔧 How it works](how-it-works.md) — the export stages, and the Rust runtime that runs the result on a phone.
