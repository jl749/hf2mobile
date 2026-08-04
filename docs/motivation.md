# 🎯 Motivation

*Why operator-level ONNX stopped being enough.* — [back to README](../README.md)

---

## ONNX standardized *operator-level* tracing

ONNX defined a portable vocabulary of computation primitives — `MatMul`, `Conv`, `ReLU`, `Softmax`.
For classical ML graphs, operator-level tracing was enough to cover most model-porting cases.

It worked because the vocabulary was **small and universal**. For instance, a ResNet was a fixed sequence of convolutions: recording the operators that one forward execution touched described the model *completely*; therefore, any backend implementing the same standard operator set could run the exports. Portability was a consequence of the contract being narrow — nothing in the file required knowledge that lived outside the standard.

## The operator level is too low to be the unit of portability today

A single ONNX graph now has to generalize across two independent axes at once:

- **Hardware** — NPU / CPU / GPU, each with different quantization schemes, memory layouts, and operator coverage.
- **Runtimes** — TRT-LLM, vLLM, llama.cpp, ORT — each expecting different graph topology, KV-cache handling, and optimization metadata.

Covering every *(hardware × runtime)* cell at the operator level is manual, per-combination work, and subtle mismatches can silently break correctness or performance.

In practice nobody covers that matrix cell by cell. The ecosystem went **plugin-centric** instead: each runtime grew its **own** extensions — fused kernels, runtime configs, session-level switches — to express the parts of a modern model the standard vocabulary cannot. Every runtime arrived at that answer independently, by lazy tracing, pattern matching, and metadata reading.

## The DAG assumption, and where modern LLMs break it

ONNX is an **interchange format**, and every mobile and edge target consumes it either directly (ONNXRuntime) or through an IR converter (QNN, TensorRT, OpenVINO, IREE). What it assumes in exchange is a **sequential DAG**: a static, acyclic graph you feed *once*, execute in topological order. For dynamic control flow, `If` / `Loop` / `Scan` ops do exist, but as second-class citizens: Qualcomm's ONNX → DLC converter carries no control-flow operators in its supported set, and other converters such as TensorRT and OpenVINO do accept them, but with limited support that makes them impractical — incomplete subgraph fusion, subgraph shape and dtype constraints on the `then` / `else` branches.

Modern LLMs and multimodal pipelines are *stateful* and autoregressive — far more complex than the traditional CV graphs the **DAG** assumption was shaped around. This makes them hard to *export*, since they involve many dynamic components both inside and outside the DAG.

A PyTorch model is a *program* executed in eager mode, whereas ONNX is a *graph* with a predefined execution path. `torch.onnx.export` traces that program into a static graph, and control flow survives only where it was written as `torch.cond` or `torch.while_loop` — which almost nobody does: neither appears anywhere in `transformers`' or `diffusers`' modeling code. So a Python `if` still specializes to the branch it took, `for` unrolls, shapes stay dynamic only where declared, and a data-dependent shape fails the export.

Four axes are where the DAG assumption hurts on modern LLMs. We cite **ONNXRuntime**'s answer for each:

| Axis | What varies, and per what | ONNXRuntime's answer | Where it actually lives |
| ---- | ------------------------- | -------------------- | ----------------------- |
| **Persistent state** | The KV cache, threaded from one decode step into the next — per **step**. A "turn" is not a single DAG pass but a *sequence* of passes sharing memory. | `com.microsoft.GroupQueryAttention` hides the cache append in-kernel, past and present sharing one preallocated buffer so it grows in place instead of being concatenated and copied. The loop *around* it moves out to a separate library, `onnxruntime-genai`, with its own `genai_config.json`. | ❌ a contrib op no converter reads, plus host code every non-ORT runtime rewrites |
| **Data-dependent routing** | An MoE router picks k of n experts — per **token**. A static graph must either evaluate every expert and mask (dense cost for sparse compute) or gather into a compact batch (data-dependent shapes). | `com.microsoft.MoE` takes `router_probs` as an ordinary input and does top-k selection inside the kernel, so the node stays static while the data-dependence happens where ONNX cannot see it. | ❌ contrib op — one opaque node with a fixed idea of what an expert is |
| **Data-dependent control flow** | Mixture-of-Depths skips a block entirely; an early-exit LM stops descending the stack — per **token**. | None. Neither a contrib op nor a runtime mechanism (`onnxruntime-genai` stops early per *sequence*, not per token), so these architectures export dense. | ❌ the saving does not survive the export at all |
| **Config-dependent weights** | Which LoRA adapter applies — per **request**, by config rather than by data. Merging (`W' = W + BA`) collapses to a static graph at the cost of a full weight set per adapter. | Adapters demoted from initializers to graph *inputs* — giving up constant folding and weight pre-packing — selected via `RunOptions.add_active_adapter` from a separate `.onnx_adapter` file. | ❌ not even an op, a session API |

**ONNXRuntime** was best positioned to solve these limitations within the graph, since the organization that authored the standard maintains it — but it could not: each axis varies at runtime along a dimension the graph IR has no way to vary over.
The common workaround, then, is to extend the ecosystem around the IR rather than the IR itself — contrib ops, runtime configs, session-level APIs.

## Example: the attention plugin

Attention is where that workaround is easiest to see, because every runtime performed it independently. Everyone agrees on the mathematics; no two agree on the boundary — where the KV cache lives, and where the op starts and ends.

| Runtime | Fused attention | Where the KV cache lives |
| ------- | --------------- | ------------------------ |
| **ONNXRuntime** | `Attention` / `MultiHeadAttention` / `GroupQueryAttention` contrib ops — [`contrib_ops/cpu/bert`](https://github.com/microsoft/onnxruntime/tree/v1.27.1/onnxruntime/contrib_ops/cpu/bert) | **Inside the op.** Graph tensors — `past_key`/`past_value` in, `present_key`/`present_value` out — but when past and present are *the same* tensor it is sized to `max_sequence_length` and the kernel appends in place. With `seqlens_k`, `total_sequence_length` and `do_rotary` it applies RoPE in the same kernel. The host allocates the buffer; the op decides what happens to it, and how far into it "now" is. |
| **TensorRT-LLM** | [`gptAttentionPlugin`](https://github.com/NVIDIA/TensorRT-LLM/tree/v1.2.1/cpp/tensorrt_llm/plugins/gptAttentionPlugin), over [`cpp/tensorrt_llm/kernels`](https://github.com/NVIDIA/TensorRT-LLM/tree/v1.2.1/cpp/tensorrt_llm/kernels) | **Beside the op.** No KV tensors in that sense: with paged KV the cache is a pool of blocks handed out per request by a cache manager, and the plugin is passed the block offsets and host-side metadata needed to find them. Shape and behavior are fixed in plugin *fields* at build time rather than expressed in a portable signature. |
| **OpenVINO** | [`ScaledDotProductAttention`](https://docs.openvino.ai/2026/documentation/openvino-ir-format/operation-sets/operation-specs/sequence/scaled-dot-product-attention.html) | **Outside the graph's I/O.** The mathematics alone — `query`/`key`/`value`, optional mask and scale, a `causal` flag. The cache is *state*: `ReadValue`/`Assign` pairs on a `Variable`, carried between `infer()` calls, reachable only through `query_state()`. Serving stacks then rewrite that again — `ov::pass::SDPAToPagedAttention` trades the state for a 28-input `PagedAttentionExtension` and block tables. One vendor, two incompatible KV contracts. |

The incompatibility is not about naming but about **who owns what**: who allocates the cache, who advances the position, who tracks sequence length. That is why a converter cannot mechanically rewrite one runtime's attention into another's — the two are not variants of the same node, but different answers to where the runtime ends and the graph begins.

A graph containing `GroupQueryAttention` is an *ONNXRuntime* graph, not a standard ONNX one. Decompose it back to `MatMul`/`Softmax`/`Concat` and the file translates again — but the fused kernel goes, KV growth reverts to concatenate-and-copy, and the accelerator's execution-provider partition fragments. **So the module level is where the performance lives, and — because it is plugin-centric — where portability stops.**

With no standard at that level, model publishers, agent frameworks, and distribution hubs each re-implement the same per-target conversion glue. Every converter in the table below reads standard ONNX and nothing else, so an export falls off it the moment it uses the plugin that made the model fast:

| Path            | Converter                                                                                               |
| --------------- | ------------------------------------------------------------------------------------------------------- |
| ONNX → QNN      | [ONNX2DLC](https://docs.qualcomm.com/doc/80-63442-10/topic/converters.html#onnx-conversion)             |
| ONNX → TensorRT | [ONNX2TRT](https://github.com/onnx/onnx-tensorrt)                                                       |
| ONNX → OpenVINO | [ONNX2OVIR](https://docs.openvino.ai/2026/openvino-workflow/model-preparation/convert-model-onnx.html)  |
| ONNX → IREE     | [ONNX2MLIR](https://iree.dev/guides/ml-frameworks/onnx/)                                                |

> **A static DAG is insufficient to express modern LLMs. Every runtime worked around that outside the IR, and each did it in its own way — a fix that has to be rewritten for every new target does not scale. That cost the portability ONNX was initially designed to provide. Generic LLM tracing for multiple targets (NPU / CPU / GPU) is the gap worth closing.**

---

**Next:** [🧩 Approach](approach.md) — the module boundary as the unit of export.
