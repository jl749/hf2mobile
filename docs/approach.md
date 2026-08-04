# Approach

*The module boundary as the unit of export.* — [back to README](../README.md)

---

## 1 Tracing one level above the operator graph

`hf2mobile` is built on top of HuggingFace `transformers`. The [motivation](motivation.md) closes on a dilemma: a fast export bound to one runtime, or a portable export that has given up the reason for exporting. Our resolution is to work one level above the flat operator graph, at the **module boundary**. The design has three parts.

1. **Trace at the module level.** The model is recorded as its semantic building blocks, that is, attention, RoPE, RMSNorm and the LM head, rather than as an undifferentiated sequence of `MatMul`, `Mul` and `Softmax`. `transformers` makes this practical, because every architecture follows the same template (`Qwen3RMSNorm`, `LlamaAttention`, `Gemma3RotaryEmbedding`) and the boundaries are therefore already drawn by the library the models ship with. An attention module enters the trace as **one node** with its identity intact (`Qwen3Attention`, `model.layers.0.self_attn`). A module held whole in this way is what we call a **plugin**: the same object the motivation described as a trap, under different ownership.
2. **Expand those nodes per target and per strategy.** One node emits a fused GQA plugin for one runtime, a sliding-window mask template for another, or a single-head decomposition for an NPU with no fused attention. The choice is made at export time.
3. **Extend the exporter API rather than pattern-match the graph.** Plugins are selected by class-name *suffix*, so a single `Attention` exporter covers every architecture that follows the convention. A new architecture or target requires one small exporter in `src/hf2mobile/exporter/`.

Figure 1 shows what the trace is in practice, for a two-layer Gemma 3 export.

<p align="center">
  <img src="2_module_level_postprocessed_graph.svg" width="280" alt="Module-level trace of a two-layer Gemma 3 decoder: Gemma3Attention and Gemma3RotaryEmbedding survive as single nodes, KV cache exposed as named graph I/O">
</p>

**Figure 1: Module-level trace of a two-layer Gemma 3 decoder.** One `Gemma3Attention` and one `Gemma3RotaryEmbedding` node per layer, with the class name carried into the graph, and the KV cache exposed as named I/O (`past_keys_0` in, `past_keys_0_out` out) rather than concealed inside a kernel.

Nothing about *how* attention runs has been decided at this point. The node is a name and a signature, and it is the baseline from which every target's file is expanded. `SlidingWindowMask` is the one exception: the HuggingFace model computes it, so it belongs to step 1, but its shape depends on `Lq` and `Lkv` at decode time, so it is emitted with the step 2 expansions rather than frozen into the trace.

## 2 Case study: placing the seam across two backends

A phone provides both an NPU and a CPU, so the question is not which of the two to select but **where to place the seam** between them. Left to a converter (onnx2trt, onnx2dlc, and the rest), the boundary falls wherever the pattern matcher happens to claim, and a single unsupported operator can strand an entire block off the accelerator. Held as plugin nodes, the split falls on boundaries that were named explicitly, at export time.

Shape determines where the seam falls. An NPU execution provider derives its efficiency from compiling ahead of time, so every dimension must be known at build time, whereas a CPU execution provider resolves shapes at runtime. Statically shaped compute is assigned to the NPU, including the QKV projections and the MLP stack, where the arithmetic density is highest. Everything shape-dependent remains on the CPU, together with sampling, control flow, and any operator for which the accelerator has no kernel. The line therefore falls between *kinds* of attention rather than across attention as a whole, as shown in Table 1.

**Table 1: Attention modules of the same class, expanded differently according to shape.**

| Attention | Shape | Assigned to | Expanded into |
| --------- | ----- | ----------- | ------------- |
| **Encoder / vision** | No cache and a fixed sequence length, hence as static as the projections around it | **NPU** | Whatever kernels the accelerator provides |
| **Decoder** | Threads a KV cache whose `total_sequence_length` grows by one per step | **CPU** | `GroupQueryAttention`, so that ORT owns the in-kernel cache append while the state remains visible in the IR as ordinary `past_key` / `present_key` tensors |

Two nodes of the same class are expanded differently because of what each one does. The decoder row is this target's decision rather than a general rule: padding the cache to a fixed length allows it to compile ahead of time as well, which is the approach taken by Qualcomm's GENIE SDK, purchasing static shape at the cost of wasted compute and a capped context. In either case the decision is made in the exporter, against a node that can be named.

Both halves still ship as a single file, since an `EPContext` node embeds the compiled NPU partition inside the same ONNX graph: one graph, one session, mixed execution.

## 3 The plugin node as the unit of control

Nothing above required a new file format. Both expansions were produced from the same kind of traced attention node, and neither is privileged. This is why the node is held whole until the final step: the HuggingFace module remains the baseline, and every expansion is a deliberate departure from it rather than whatever a converter's pattern matcher recognized in an already-flattened graph.

Attention is one of the four axes opened in the [motivation](motivation.md), and the same mechanism reaches the others. An MoE block held whole is one description that expands into `com.microsoft.MoE`, into a dense-masked fallback, or into a gather over a compact batch: three targets rather than three rewrites. A LoRA adapter attaches at a module boundary by construction, and is merged into the weights for one target and left as a graph input for another. Mixture-of-Depths is the honest limit, since a trace records the execution it observed and per-token skipping therefore survives no export at any boundary; the node nonetheless names the block that a custom kernel would have to claim, which a flattened graph does not.

The same boundary is what makes module-level quantization expressible. Rotation-based schemes such as QuaRot [2] fuse Hadamard transformations into the weight matrices on either side of a block, exploiting computational invariance to remove outlier features without altering the model's output, and apply online transformations inside the attention module to make the KV cache quantizable. The transformation is defined by which weight matrices sit at a block boundary, which is information a flattened operator graph no longer carries. Rotation-based quantization is on the roadmap rather than in the current export path, but it is representative of the class of transformations the module boundary is intended to keep available.

The portable artifact is therefore the module-level trace rather than any file produced from it. A graph carrying a runtime-specific plugin or fusion pattern is bound to that runtime, but every target's file can be derived from the same module-level expansion.

This resolves the opening dilemma, not by making the fast export portable but by keeping the two apart. Every emitted file remains bound to its runtime, and the trace is what moves between them. The cost is that the trace is `hf2mobile`'s own representation, which no other tool reads, so the per-*(hardware × runtime)* work is collapsed into one place rather than standardized away. The description is what is versioned, and every runtime-bound file is regenerated from it rather than maintained. Being plugin-centric ceases to be a trap once what is retained is the description rather than any one expansion of it.

## 4 A descriptive graph and a generic runtime

The exporter and the runtime are one deliverable, designed against each other. Module identity is the *exporter's* lever. What the **runtime** requires is narrower but equally explicit: cache slots are named, and the sampling policy and EOS ids are [baked into the graph](../README.md#-usage-cli). The Rust runtime ([§2 of how it works](how-it-works.md#2-case-study-the-causallm-runtime)) reads all of this out of the file, which is what keeps it small and model-agnostic. The semantics reside in one artifact, in the IR, instead of being distributed across a contrib op, a side-car configuration file and a session API.

Attention illustrates the division of labor. `GroupQueryAttention` owns the in-kernel cache append, so the export targets it directly; the loop around it, comprising prefill, decode and stop, is host work, and the runtime owns that. One loop runs every exported model, and is small enough to cross-compile for a phone.

## 5 What is reused and what is added

**Reused.** `transformers` module definitions, ORT contrib ops (`GroupQueryAttention`, RMSNorm and RoPE fusions), ORT execution providers, and `EPContext`.

**Added.** The module-level tracer and plugin registry, the per-target expanders, `SampleLogits` together with the in-graph EOS constant, and the Rust runtime.

## References

[2] S. Ashkboos et al. *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs*. NeurIPS, 2024. [arXiv:2404.00456](https://arxiv.org/abs/2404.00456)

---

**Next:** [How it works](how-it-works.md) — the export stages, and the Rust runtime that runs the result on a phone.
