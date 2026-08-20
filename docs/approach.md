# Approach

*The module boundary as the unit of export.* ([back to README](../README.md))

---

The [motivation](motivation.md) ends on a dilemma: the plugin that makes a graph fast is also what stops portability.

Existing porting pipelines (Optimum, Olive, Qualcomm AI Hub) answer that fragmentation by deciding the expansion for you. Being community-driven, each settles on a strategy broad enough for the common cases. That strategy covers only what its authors anticipated, and their design makes extending it difficult. How a module becomes a graph is fixed by the pipeline, precisely where a mobile target needs more flexibility:

- **KV caching strategy.** Dynamic or padded, and in what tensor layout.
- **dtype management.** Where mixed precision is allowed and where it is not.
- **Execution-provider partitioning.** Which subgraphs run on which backend.

`hf2mobile` hands those decisions back to the user. Expansion becomes something you write rather than something you select, which changes what the dilemma applies to: performance and portability stop competing once the fast, target-specific export is no longer the artifact you keep. What you keep instead is the module-level trace described below, and every target's file is generated from it.

This is possible because `hf2mobile` is built **on top of HuggingFace `transformers`**, where every architecture follows the same prebuilt template (e.g. `Qwen3RMSNorm`, `LlamaAttention`, `Gemma3RotaryEmbedding`), so the module-level boundaries are already drawn by the library. Working one level up from the flat operator graph, at the **module boundary**, it supplies both a working mobile deployment pipeline and a baseline language for extending support to custom models and backends:

1. **Trace at the *module* level.** Rather than the conventional operator-level trace, the model is recorded as its semantic building blocks: Attention, RoPE, RMSNorm, the LM head. An `Attention` module enters the trace as a **single node**, with its identity intact (type: `Qwen3Attention`, name: `model.layers.0.self_attn`).
2. **Expand those nodes per target, per strategy.** Attention is the clearest case: MHA, MQA, GQA and sliding-window all enter the trace as one discrete node, and are expanded per target from the node's metadata into either a fused ORT contrib op or a plain single-head attention for QNN.
3. **To support a new model or target, extend the existing exporter API.** Custom-op registration is done through a generic register API (`hf2mobile/tracing/register.py`). Per-target (`-t ORT`, `-t QNN`) and per-model expansion rules can be specified by one small module in `hf2mobile/exporter/` that extends the existing APIs (e.g. `hf2mobile/exporter/llama.py`).

Here is what that looks like on a two-layer Gemma 3 export:

<p align="center">
  <img src="2_module_level_postprocessed_graph.svg" width="280" alt="Module-level trace of a two-layer Gemma 3 decoder: Gemma3Attention and Gemma3RotaryEmbedding survive as single nodes, KV cache exposed as named graph I/O">
  <br>
  <sub>every decoder layer exposes one <code>Gemma3Attention</code> node and one <code>Gemma3RotaryEmbedding</code> node, each carrying its class name and metadata into the graph, and the KV cache is threaded through named graph I/O (<code>past_keys_0</code> in, <code>past_keys_0_out</code> out). <code>SlidingWindowMask</code> appears only on the layers whose metadata (module attributes and <code>config.json</code>) calls for it.</sub>
</p>

At this stage, nothing about *how* attention runs has been decided. Each node carries a name, a signature, and its metadata: named I/Os, `head_size`, `hidden_dim`. This is the information the later exporter reads when expanding.

## Example: one graph, more than one backend

A mobile SoC ships an NPU and a CPU on the same die, and the best performance comes from utilizing both at full capability. The question is therefore not which one to pick, but where to place the boundary between them.

Leave it to the runtime and that boundary is chosen for you. Hand an ONNX file to the [QNN execution provider](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html) and it partitions the graph on [its own](https://github.com/microsoft/onnxruntime/blob/v1.27.0/onnxruntime/core/providers/qnn/qnn_execution_provider.cc#L948): whatever the pattern matcher can claim is fused into `EPContext` nodes, and the rest falls back to CPU (`disable_cpu_ep_fallback = 0`, by default). Either way the placement is not yours: one unsupported op in the middle of a block can push the whole block off the accelerator, and tweaking the details of the IR is an unreliable way to steer it.

Hold the model as *module-level* nodes and the boundary falls where you place it, at **export time**. The node is the unit of control, and it does not constrain where the cut lands. Since you decide how each node expands, the module boundary is the coarsest available cut rather than the only one.

**Shape is the criterion, and expansion is the lever.** An NPU EP earns its efficiency by compiling **ahead of time**, so every dimension must be known at build time, whereas a CPU EP resolves shapes at runtime. A node expanded into a fused contrib op therefore pins that region to the CPU, while one expanded into statically-shaped ops can be compiled for the NPU. Statically-shaped compute goes to the NPU, where the TOPS are (e.g. QKVO projections and the MLP nodes in the case of `EncoderLayer`/`DecoderLayer`). Everything shape-dependent stays on the CPU, along with sampling, control flow, and any op the accelerator has no kernel for.

A single layer, for example, does not land on one side or the other. Both layer types project statically, and they differ only in the attention op between those projections:

| Layer | QKV / O projections | Attention (`scaled_dot_product_attention`) | Expanded into |
| ----- | ------------------- | ------------------------------------------ | ------------- |
| **EncoderLayer** | statically shaped → **NPU** | no cache, fixed sequence length → **NPU** | single-head subgraph, as the NPU lacks 5d support |
| **DecoderLayer** | statically shaped → **NPU** | threads a KV cache whose `total_sequence_length` grows by one per step → **CPU** | `GroupQueryAttention`, so ORT owns the in-kernel cache append and the state stays visible in the IR as ordinary `past_key` / `present_key` I/O buffers |

In both cases the expansion is decided in the exporter. The same trace that generalizes across targets also localizes partitioning: each module exporter uses the node's metadata (type: `Qwen3Attention`, name: `model.layers.0.self_attn`) to declare which of the ops it emits belong to an NPU partition. Each declared partition is compiled ahead of time and replaced by an `EPContext` node, and adjacent `EPContext` nodes merge as the subgraphs are flattened in, so no pattern matcher has to recover that structure from a flattened graph.

Below is the same graph one step later, with `Gemma3Attention` expanded into `GroupQueryAttention` on the CPU side. Both NPU and CPU backends still ship as one file, since an `EPContext` node embeds the compiled NPU partition inside the same ONNX graph. One graph, one session, mixed execution.

<p align="center">
  <img src="epcontext_partitioning_example.svg" width="280" alt="The same two-layer Gemma 3 graph reduced to nine nodes: three EPContext partitions alternating with the two GroupQueryAttention nodes, which keep their past_keys and past_values I/O on the CPU side">
  <br>
  <sub>the same export, with every statically-shaped run collapsed into an <code>EPContext</code> node: 79 nodes become 9, and the two <code>GroupQueryAttention</code> nodes stay on the CPU side with their <code>past_keys_0</code> / <code>past_keys_0_out</code> I/O intact.</sub>
</p>

## What you keep is the trace

Nothing above required a new file format. The expansions above came out of the same kind of traced `Attention` node, and neither is privileged. This is why the node is held whole until the last step: the HuggingFace module stays the baseline, and every expansion is a deliberate departure from it.

<!-- TODO: cover the remaining axes from motivation.md (MoE, LoRA, Mixture-of-Depths) once they are implemented -->


> [!IMPORTANT]
> **The module-level trace is the portable artifact, and the files produced from it are not.** A graph carrying a runtime-specific plugin or fusion pattern is bound to that runtime, whereas every target's file can be generated from the same module-level trace.

That resolves the opening dilemma by keeping speed and portability in separate artifacts rather than by making the fast export portable. In practice this means versioning the trace and regenerating every runtime-bound file from it, instead of maintaining any of them. The cost is that the trace is `hf2mobile`'s own representation, which no other tool reads, so the per-*(hardware x runtime)* work is collapsed into one place rather than standardized away.

## A descriptive graph, and a generic runtime to read it

The exporter and the runtime are designed against each other. The **exporter** works off module identity, meaning which `transformers` class a node came from. The **runtime** never sees that and does not need to; what it needs is a much shorter list, stated plainly enough in the graph that the [Rust runtime](how-it-works.md#example-causallm-inferencer) can read it straight out of the file. The semantics live in **one artifact, in the IR**, instead of being spread across a contrib op, a runtime config and a session API.

Attention is a good place to see where the line falls. Growing the KV cache happens inside the graph, since `GroupQueryAttention` appends in-kernel, so the exporter's job is to emit that node. What wraps around it is plain host code: feed the prompt, call the session once per token, stop on an EOS id.

That host code can stay generic because the graph already answers everything it would otherwise have to be told. Cache slots are discovered by convention (any input `x` with a matching `x_out` output), so no layer names are hardcoded and the loop survives a change in layer count. The EOS ids are a `Constant` in the graph, and the sampling strategy is a custom node. Nothing is read from a runtime config, so there is no per-model branch anywhere in the runtime.

This is also why the runtime ships *with* the exporter. An export that picks its own partition points and its own cache layout is not what AI Hub or Olive expects to receive, so a matching runtime is part of the deliverable rather than something one hopes to find. There is not much to implement, since the whole inference path (reading the graph, the session, the KV cache, prefill and decode) is under 900 lines of Rust. It also never copies the cache, because ORT hands tensors out as reference-counted handles, so feeding the last step's keys back in is a pointer write and a refcount bump rather than a `memcpy` of tens of megabytes per token.

---

> [!IMPORTANT]
> **The module boundary is the unit of export because it is the last place where a model is still described rather than committed.** Above it, `transformers` has already drawn the boundaries. Below it, every choice (which kernel, which cache layout, which processor) has been made and can no longer be revisited. Holding the model at that boundary until the last step is what lets one trace serve targets that share no vocabulary, and what turns each new *(hardware x runtime)* cell into one exporter rather than one rewrite.

**Next:** [How it works](how-it-works.md), the export stages and the Rust runtime that runs the result on a phone.
