# 🧩 Approach

*The module boundary as the unit of export.* — [back to README](../README.md)

---

`hf2mobile` is built **on top of HuggingFace `transformers`**. The [motivation](motivation.md) ends in a dilemma: a fast export bound to a single runtime, or a portable export that abandons the very reason you exported in the first place. The way out is to operate one level above the flat operator graph — at the **module boundary**:

1. **Trace at the *module* level.** The model is recorded as its semantic building blocks — attention, RoPE, RMSNorm, the LM head — rather than an unstructured soup of `MatMul`, `Mul`, and `Softmax`. `transformers` makes this practical because every architecture follows consistent naming conventions (`Qwen3RMSNorm`, `LlamaAttention`, `Gemma3RotaryEmbedding`), so the boundaries are already drawn. An `Attention` module enters the trace as a single node with its identity intact (`Qwen3Attention`, `model.layers.0.self_attn`). A module held whole this way — one node in the graph — is what this documentation calls a **plugin**: the same object [motivation](motivation.md) called a trap, but under a different owner.
2. **Expand those nodes per target and strategy.** The same node emits a fused GQA plugin for one runtime, a sliding-window mask template for another, or a single-head decomposition for an NPU lacking fused attention — decided entirely at export time.
3. **Extend the exporter API, don't pattern-match the graph.** Plugins are selected by class-name *suffix*, allowing a single `Attention` exporter to cover every architecture following the convention. A new architecture or target requires only one small exporter in `src/hf2mobile/exporter/`.

Here is what that trace actually looks like for a two-layer Gemma 3 export:

<p align="center">
  <img src="2_module_level_postprocessed_graph.svg" width="280" alt="Module-level trace of a two-layer Gemma 3 decoder: Gemma3Attention and Gemma3RotaryEmbedding survive as single nodes, KV cache exposed as named graph I/O">
</p>

One `Gemma3Attention` and one `Gemma3RotaryEmbedding` node per layer — with class names carried directly into the graph — and the KV cache exposed as named I/O (`past_keys_0` in, `past_keys_0_out` out) rather than buried inside a black-box kernel. Nothing about *how* attention runs is decided yet: the node is simply a name and a signature, forming the baseline from which every target's file expands. `SlidingWindowMask` is the sole exception; the HuggingFace model computes it natively, meaning it belongs to step 1, but its shape depends on `Lq` and `Lkv` at decode time, so it is emitted with the step 2 expansions rather than frozen directly into the trace.

## EXAMPLE: placing the seam — one graph, multiple backends

A phone pairs a CPU with an NPU, so the architectural question isn't which processor to use, but **where to draw the seam**. Leave that choice to automated converters like onnx2trt or onnx2dlc, and the boundary is dictated by rigid pattern matchers — and a single unsupported op can strand an entire block off the accelerator. By defining plugin nodes explicitly at **export time**, the split happens on boundaries you choose.

**Shape decides where it falls.** An NPU Execution Provider (EP) earns its efficiency by compiling **ahead of time**, meaning every dimension must be known at build time, whereas a CPU EP resolves shapes dynamically at runtime. Statically-shaped compute goes to the NPU — such as QKV projections and the MLP stack, where the TOPS live. Everything shape-dependent stays on the CPU, alongside sampling, control flow, and any operator the accelerator lacks a kernel for. This draws the line between *types* of attention rather than cutting through attention as a whole:

| Attention | Shape | Goes to | Expanded into |
| --------- | ----- | ------- | ------------- |
| **Encoder / vision** | no cache, fixed sequence length — as static as the surrounding projections | **NPU** | whatever native kernels the accelerator provides |
| **Decoder** | threads a KV cache whose `total_sequence_length` grows by one per step | **CPU** | `GroupQueryAttention`, ensuring ORT owns the in-kernel cache append while the state remains visible in the IR as ordinary `past_key` / `present_key` tensors |

Two nodes of the same general family, expanded differently based on their operational behavior. The decoder row is this target's specific choice rather than an immutable law — you can pad the cache to a fixed length and compile it AOT as well (much like Qualcomm's GENIE SDK does), buying static shape at the cost of wasted compute and a capped context window. Either way, the decision is made deterministically in the exporter against a named node.

Both halves still ship as a single file: an `EPContext` node embeds the compiled NPU partition directly inside the parent ONNX graph. One graph, one session, mixed execution.

## The plugin node as the unit of control

Nothing above required a new file format — both expansions stem from the same traced `Attention` node, and neither is privileged. That's why the node is held intact until the final step: the HuggingFace module remains the single source of truth, and every expansion is a deliberate departure from it, rather than whatever a converter's pattern matcher happens to guess in an already-flattened graph.

Attention was just one of four axes introduced in [motivation](motivation.md); the same mechanism reaches the rest. Hold an MoE block whole, and it fans out into three distinct targets without requiring three separate rewrites: `com.microsoft.MoE`, a dense-masked fallback, or a gather into a compact batch. A LoRA adapter attaches naturally at the module boundary — merged directly into the weights for one target, or left as an explicit graph input for another. Mixture-of-Depths hits the hard limit: a trace only records the execution path it actually saw, meaning per-token skipping cannot survive static export at any boundary. Even so, the high-level node still preserves the block's identity so a custom kernel can claim it, long after a flattened graph loses the plot.

> The portable artifact is the module-level trace, not any file it produces. A graph carrying a runtime-specific plugin or fusion pattern is bound to that runtime — but every target's file can be generated from the same module-level expansion.

This resolves the opening dilemma — not by forcing a fast export to be portable, but by keeping the two domains strictly separated. Every emitted file stays bound to its runtime target; the trace is what moves. The trade-off is that the trace is `hf2mobile`'s custom representation, which no other tool reads, collapsing per-*(hardware × runtime)* work into a single centralized place rather than standardizing it away. The description is what gets versioned; every runtime-bound file is regenerated from it rather than maintained by hand.

Being plugin-centric stops being a trap once what you preserve is the description, not any single downstream expansion of it.

## A descriptive graph, and a generic runtime to read it

The exporter and the runtime are built as a single unified deliverable, designed hand-in-hand. Module identity serves as the *exporter's* lever; what the **runtime** requires is narrower but equally explicit — cache slots are explicitly named, and the sampling policy and EOS ids are [baked in](../README.md#2-hf2mobilepostprocess--bake-in-the-decode-policy). The Rust runtime ([CausalLM inferencer](how-it-works.md#example-causallm-inferencer)) reads all of this configuration directly out of the file, keeping the codebase small and completely model-agnostic. The core semantics live inside **one artifact, cleanly represented in the IR**, rather than being fragmented across a contrib op, a side-car config file, and a sprawling session API.

Attention highlights this structural division: `GroupQueryAttention` owns the in-kernel cache append, making it a clean target for export. The surrounding loop — prefill, decode, and stop conditions — belongs entirely to the host runtime. Because that control loop is minimal and uniform across models, it is exceptionally easy to cross-compile for mobile devices.

## What is reused, what is added

**Reused** — `transformers` module definitions, ORT contrib ops (`GroupQueryAttention`, RMSNorm/RoPE fusions), ORT execution providers, and `EPContext`.
**Added** — the module-level tracer and plugin registry, per-target expanders, `SampleLogits` alongside the in-graph EOS constant, and the custom Rust runtime.

---

**Next:** [🔧 How it works](how-it-works.md) — the export stages, and the Rust runtime that runs the result on a phone.
