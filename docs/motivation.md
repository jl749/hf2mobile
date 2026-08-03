# 🎯 Motivation

*Why operator-level ONNX stopped being enough.* — [back to README](../README.md)

---

## ONNX standardized the *operator* level tracing

ONNX defined a portable vocabulary of computation primitives — `MatMul`, `Conv`, `ReLU`, `Attention`.
For classical ML graphs, the operator level tracing was enough to cover the majority of the model porting cases.

It worked because the vocabulary was **small and universal**. A ResNet is a fixed sequence of convolutions: recording the operators that one execution touched described the model *completely*, and any backend implementing the same standard operator set could run any graph anyone exported. Portability was a consequence of the contract being narrow — nothing in the file required knowledge that lived outside the standard.

## The operator level tracing is too low to be the unit of portability today

A single ONNX graph now has to generalize across two independent axes at once:

- **Hardware** — NPU / CPU / GPU, each with different quantization schemes, memory layouts, and operator coverage.
- **Runtimes** — TRT-LLM, vLLM, llama.cpp, ORT — each expecting different graph topology, KV-cache handling, and optimization metadata.

Covering every *(hardware × runtime)* cell at the operator level is manual, per-combination work, and subtle mismatches can silently break correctness or performance.

In practice nobody covers that matrix cell by cell. The ecosystem went **plugin-centric** instead: each runtime grows its **own** extensions — fused kernels, side-car configs, session-level switches — to express the parts of a modern model the standard vocabulary cannot. Every runtime arrived at that answer independently. That buys back performance and expressiveness on that one runtime, but it costs precisely the property ONNX existed to provide: **a universal IR that any runtime can interpret.**

## The DAG assumption, and where modern LLMs break it

ONNX is an **interchange format**, and every mobile and edge target consumes it — directly (ONNXRuntime) or through a converter (QNN, TensorRT, OpenVINO, IREE). What it assumes in exchange is a **sequential DAG**: a static, acyclic graph you feed *once*, execute in topological order, and read outputs from — a pure function of its inputs. `If` / `Loop` / `Scan` do exist, but as second-class citizens: their bodies are *attributes* rather than values, so a branch cannot be split across backends the way a straight-line graph can, and an accelerator that compiles its partition ahead of time cannot claim a region whose trip count is unknown.

Modern LLMs and multimodal pipelines are *stateful* and autoregressive — far more involved than the traditional CV graphs the **DAG** assumption was shaped around.

That makes them hard to *export*, not merely hard to represent. `torch.export` traces under symbolic shapes, and control flow survives only where it was written as `torch.cond` or `torch.while_loop` — which almost nobody does: neither appears anywhere in `transformers`' modeling code. So a Python `if` still specializes to the branch it took, `for` unrolls, shapes stay dynamic only where declared, and a data-dependent size fails the export. The graph is valid but its scope is narrower than the original.

Four axes make this bite. Each varies at runtime along a dimension the graph has no way to vary over, and each is resolved the same way: by moving the decision out of the graph and into the runtime. We cite **ONNXRuntime**'s answer for each — as the reference implementation of ONNX, maintained by the organization that authored the format, it is the project best positioned to have solved these *within* the graph.

| Axis | What varies, and per what | ONNXRuntime's answer | Where it actually lives |
| ---- | ------------------------- | -------------------- | ----------------------- |
| **Persistent state** | The KV cache, threaded from one decode step into the next — per **step**. A "turn" is not a single DAG pass but a *sequence* of passes sharing memory. | `com.microsoft.GroupQueryAttention` hides the cache append in-kernel, past and present sharing one preallocated buffer so it grows in place instead of being concatenated and copied. The loop *around* it moves out to a separate library, `onnxruntime-genai`, with its own `genai_config.json`. | ❌ a contrib op no converter reads, plus host code every non-ORT runtime rewrites |
| **Data-dependent routing** | An MoE router picks k of n experts — per **token**. A static graph must either evaluate every expert and mask (dense cost for sparse compute) or gather into a compact batch (data-dependent shapes). | `com.microsoft.MoE` takes `router_probs` as an ordinary input and does top-k selection inside the kernel, so the node stays static while the data-dependence happens where ONNX cannot see it. | ❌ contrib op — one opaque node with a fixed idea of what an expert is |
| **Data-dependent control flow** | Mixture-of-Depths skips a block entirely; an early-exit LM stops descending the stack — per **token**. | None. Neither a contrib op nor a runtime mechanism (`onnxruntime-genai` stops early per *sequence*, not per token), so these architectures export dense. | ❌ the saving does not survive the export at all |
| **Config-dependent weights** | Which LoRA adapter applies — per **request**, by config rather than by data. Merging (`W' = W + BA`) collapses to a static graph at the cost of a full weight set per adapter. | Adapters demoted from initializers to graph *inputs* — giving up constant folding and weight pre-packing — selected via `RunOptions.add_active_adapter` from a separate `.onnx_adapter` file. | ❌ not even an op, a session API |

`If` does not close rows 2 and 3: its predicate is one decision per graph execution, and routing needs one per token. Row 3 is the sharpest — no plugin, so no portability cost, and no capability either: the graph translates cleanly precisely *because* the thing worth exporting did not survive the export.

## EXAMPLE: attention plugin

Attention has shifted from the operator view toward a **module / plugin** view (fused attention, KV-cache blocks, MoE routers). This is the **plugin-centric** turn in its clearest form: every runtime ships and maintains its *own* fused-attention module rather than sharing one, and while everyone agrees on the mathematics, no two agree on the boundary — specifically, on where the KV cache lives relative to the op.

| Runtime | Fused attention | Where the KV cache lives |
| ------- | --------------- | ------------------------ |
| **ONNXRuntime** | `Attention` / `MultiHeadAttention` / `GroupQueryAttention` contrib ops — [`contrib_ops/cpu/bert`](https://github.com/microsoft/onnxruntime/tree/v1.27.1/onnxruntime/contrib_ops/cpu/bert) | **Inside the op.** Graph tensors — `past_key`/`past_value` in, `present_key`/`present_value` out — but when past and present are *the same* tensor it is sized to `max_sequence_length` and the kernel appends in place. With `seqlens_k`, `total_sequence_length` and `do_rotary` it applies RoPE in the same kernel. The host allocates the buffer; the op decides what happens to it, and how far into it "now" is. |
| **TensorRT-LLM** | [`gptAttentionPlugin`](https://github.com/NVIDIA/TensorRT-LLM/tree/v1.2.1/cpp/tensorrt_llm/plugins/gptAttentionPlugin), over [`cpp/tensorrt_llm/kernels`](https://github.com/NVIDIA/TensorRT-LLM/tree/v1.2.1/cpp/tensorrt_llm/kernels) | **Beside the op.** No KV tensors in that sense: with paged KV the cache is a pool of blocks handed out per request by a cache manager, and the plugin is passed the block offsets and host-side metadata needed to find them. Shape and behavior are fixed in plugin *fields* at build time rather than expressed in a portable signature. |
| **OpenVINO** | [`ScaledDotProductAttention`](https://docs.openvino.ai/2026/documentation/openvino-ir-format/operation-sets/operation-specs/sequence/scaled-dot-product-attention.html) | **Outside the graph's I/O.** The mathematics alone — `query`/`key`/`value`, optional mask and scale, a `causal` flag. The cache is *state*: `ReadValue`/`Assign` pairs on a `Variable`, carried between `infer()` calls, reachable only through `query_state()`. Serving stacks then rewrite that again — `ov::pass::SDPAToPagedAttention` trades the state for a 28-input `PagedAttentionExtension` and block tables. One vendor, two incompatible KV contracts. |

The incompatibility is not naming, it is **who owns what**: who allocates the cache, who advances the position, who tracks sequence length. A converter cannot mechanically rewrite one into another, because what it would be rewriting is not the same node; it is a different answer to where the runtime ends and the graph begins.

> **So the module level is where the performance lives, and — being plugin-centric — where portability stops.**

A graph containing `GroupQueryAttention` is an *ONNXRuntime* graph, not an ONNX one. Decompose it back to `MatMul`/`Softmax`/`Concat` and the file translates again, at the price of the concatenate-and-copy KV growth, the fused kernel, and often the accelerator partition with it.

With no standard at that level, model publishers, agent frameworks, and distribution hubs each re-implement the same per-target conversion glue. Every converter below reads standard ONNX and nothing else, so an export falls off this table the moment it uses the plugin that made the model fast:

| Path            | Converter                                                                                               |
| --------------- | ------------------------------------------------------------------------------------------------------- |
| ONNX → QNN      | [ONNX2DLC](https://docs.qualcomm.com/doc/80-63442-10/topic/converters.html#onnx-conversion)             |
| ONNX → TensorRT | [ONNX2TRT](https://github.com/onnx/onnx-tensorrt)                                                       |
| ONNX → OpenVINO | [ONNX2OVIR](https://docs.openvino.ai/2026/openvino-workflow/model-preparation/convert-model-onnx.html)  |
| ONNX → IREE     | [ONNX2MLIR](https://iree.dev/guides/ml-frameworks/onnx/)                                                |

---

**Next:** [🧩 Approach](approach.md) — the module boundary as the unit of export.
