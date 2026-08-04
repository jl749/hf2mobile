# Motivation

*Why the operator level is no longer a sufficient unit of portability.* — [back to README](../README.md)

---

## ONNX standardized *operator-level* tracing

ONNX defined a portable vocabulary of computation primitives — `MatMul`, `Conv`, `ReLU`, `Softmax`.
For classical ML graphs, operator-level tracing was enough to cover most model-porting cases.

It worked because the vocabulary was **small and universal**. For instance, a ResNet was a fixed sequence of convolutions: recording the operators that one forward execution touched described the model *completely*; therefore, any backend implementing the same standard operator set could run the exports. Portability was a consequence of the contract being narrow — nothing in the file required knowledge that lived outside the standard.

## The operator level is too low to be the unit of portability today

A single exported graph is expected to generalize along two independent axes:

- **Hardware.** NPU, CPU and GPU targets differ in quantization scheme, memory layout, and operator coverage.
- **Runtimes.** TensorRT-LLM, vLLM, llama.cpp and ONNXRuntime expect different graph topologies, different KV-cache conventions, and different optimization metadata.

Covering the *(hardware × runtime)* product at the operator level is per-cell manual work, and a mismatch in any one cell degrades correctness or performance without a diagnostic. In practice the product is not covered cell by cell. Each runtime instead extends the standard with its own fused kernels, side-car configuration files, and session-level switches, expressing the parts of a modern model that the standard vocabulary cannot. Every runtime arrived at this design independently. We refer to the result as **plugin-centric**: it recovers performance and expressiveness on one runtime at the cost of the property the format existed to provide, namely a single IR that any runtime can interpret.

In practice nobody covers that matrix cell by cell. The ecosystem went **plugin-centric** instead: each runtime grew its **own** extensions — fused kernels, runtime configs, session-level switches — to express the parts of a modern model the standard vocabulary cannot. Every runtime arrived at that answer independently, by lazy tracing, pattern matching, and metadata reading.

ONNX is an interchange format, and every mobile and edge target consumes it, either directly (ONNXRuntime) or through a converter (QNN, TensorRT, OpenVINO, IREE). What the format assumes in return is a **sequential DAG**: a static acyclic graph that is fed once, executed in topological order, and read from, behaving as a pure function of its inputs. `If`, `Loop` and `Scan` exist, but as second-class constructs. Their bodies are attributes rather than values, so a branch cannot be partitioned across backends the way straight-line regions can, and an accelerator that compiles its partition ahead of time cannot claim a region whose trip count is unknown.

ONNX is an **interchange format**, and every mobile and edge target consumes it either directly (ONNXRuntime) or through an IR converter (QNN, TensorRT, OpenVINO, IREE). What it assumes in exchange is a **sequential DAG**: a static, acyclic graph you feed *once*, execute in topological order. For dynamic control flow, `If` / `Loop` / `Scan` ops do exist, but as second-class citizens: Qualcomm's ONNX → DLC converter carries no control-flow operators in its supported set, and other converters such as TensorRT and OpenVINO do accept them, but with limited support that makes them impractical — incomplete subgraph fusion, subgraph shape and dtype constraints on the `then` / `else` branches.

Modern LLMs and multimodal pipelines are *stateful* and autoregressive — far more complex than the traditional CV graphs the **DAG** assumption was shaped around. This makes them hard to *export*, since they involve many dynamic components both inside and outside the DAG.

A PyTorch model is a *program* executed in eager mode, whereas ONNX is a *graph* with a predefined execution path. `torch.onnx.export` traces that program into a static graph, and control flow survives only where it was written as `torch.cond` or `torch.while_loop` — which almost nobody does: neither appears anywhere in `transformers`' or `diffusers`' modeling code. So a Python `if` still specializes to the branch it took, `for` unrolls, shapes stay dynamic only where declared, and a data-dependent shape fails the export.

Four axes are where the DAG assumption hurts on modern LLMs. We cite **ONNXRuntime**'s answer for each:

`If` does not close rows 2 and 3, since its predicate expresses one decision per graph execution while routing requires one per token. Row 3 is the sharpest case: there is no plugin, hence no portability cost and no capability either. The graph translates cleanly precisely because the property worth exporting did not survive the export.

**ONNXRuntime** was best positioned to solve these limitations within the graph, since the organization that authored the standard maintains it — but it could not: each axis varies at runtime along a dimension the graph IR has no way to vary over.
The common workaround, then, is to extend the ecosystem around the IR rather than the IR itself — contrib ops, runtime configs, session-level APIs.

## Example: the attention plugin

Attention is where that workaround is easiest to see, because every runtime performed it independently. Everyone agrees on the mathematics; no two agree on the boundary — where the KV cache lives, and where the op starts and ends.

| Runtime | Fused attention | Where the KV cache lives |
| ------- | --------------- | ------------------------ |
| **ONNXRuntime** | `Attention` / `MultiHeadAttention` / `GroupQueryAttention` contrib ops, in [`contrib_ops/cpu/bert`](https://github.com/microsoft/onnxruntime/tree/v1.27.1/onnxruntime/contrib_ops/cpu/bert) | **Inside the operator.** The cache appears as graph tensors (`past_key`/`past_value` in, `present_key`/`present_value` out), but when past and present are the *same* tensor it is sized to `max_sequence_length` and the kernel appends in place. Given `seqlens_k`, `total_sequence_length` and `do_rotary`, the same kernel also applies RoPE. The host allocates the buffer; the operator determines what happens to it and how far into it the current position lies. |
| **TensorRT-LLM** | [`gptAttentionPlugin`](https://github.com/NVIDIA/TensorRT-LLM/tree/v1.2.1/cpp/tensorrt_llm/plugins/gptAttentionPlugin), over [`cpp/tensorrt_llm/kernels`](https://github.com/NVIDIA/TensorRT-LLM/tree/v1.2.1/cpp/tensorrt_llm/kernels) | **Beside the operator.** There are no KV tensors in that sense: with paged KV the cache is a pool of blocks issued per request by a cache manager, and the plugin receives the block offsets and the host-side metadata needed to locate them. Shape and behavior are fixed in plugin *fields* at build time rather than expressed in a portable signature. |
| **OpenVINO** | [`ScaledDotProductAttention`](https://docs.openvino.ai/2026/documentation/openvino-ir-format/operation-sets/operation-specs/sequence/scaled-dot-product-attention.html) | **Outside the graph's I/O.** The operator carries the mathematics alone: `query`/`key`/`value`, an optional mask and scale, and a `causal` flag. The cache is *state*, expressed as `ReadValue`/`Assign` pairs on a `Variable`, carried between `infer()` calls and reachable only through `query_state()`. Serving stacks then rewrite that again: `ov::pass::SDPAToPagedAttention` exchanges the state for a 28-input `PagedAttentionExtension` and block tables. One vendor, two incompatible KV contracts. |

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

**Next:** [Approach](approach.md) — the module boundary as the unit of export.
