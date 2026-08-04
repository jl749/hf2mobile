# Motivation

*Why the operator level is no longer a sufficient unit of portability.* — [back to README](../README.md)

---

## 1 The operator set as a unit of portability

ONNX defines a portable vocabulary of computational primitives (`MatMul`, `Conv`, `ReLU`, `Attention`), and for the class of models it was designed around, recording the operators that one execution traversed described the model completely. A ResNet is a fixed sequence of convolutions; its trace is its definition.

Portability followed from the narrowness of the contract rather than from its coverage. The vocabulary was small and universal, and nothing in an exported file depended on knowledge held outside the standard, so any backend implementing the standard operator set could execute any graph any exporter produced. This is the same property TVM identifies as absent one level down, where frameworks delegate target-specific optimization to vendor operator libraries that are "too specialized and opaque to be easily ported across hardware devices" [1]. Our concern is the mirror image: the layer at which the specialization now happens has moved *above* the operator set rather than below it.

## 2 Two axes of generalization

A single exported graph is expected to generalize along two independent axes:

- **Hardware.** NPU, CPU and GPU targets differ in quantization scheme, memory layout, and operator coverage.
- **Runtimes.** TensorRT-LLM, vLLM, llama.cpp and ONNXRuntime expect different graph topologies, different KV-cache conventions, and different optimization metadata.

Covering the *(hardware × runtime)* product at the operator level is per-cell manual work, and a mismatch in any one cell degrades correctness or performance without a diagnostic. In practice the product is not covered cell by cell. Each runtime instead extends the standard with its own fused kernels, side-car configuration files, and session-level switches, expressing the parts of a modern model that the standard vocabulary cannot. Every runtime arrived at this design independently. We refer to the result as **plugin-centric**: it recovers performance and expressiveness on one runtime at the cost of the property the format existed to provide, namely a single IR that any runtime can interpret.

## 3 The dataflow-graph assumption

ONNX is an interchange format, and every mobile and edge target consumes it, either directly (ONNXRuntime) or through a converter (QNN, TensorRT, OpenVINO, IREE). What the format assumes in return is a **sequential DAG**: a static acyclic graph that is fed once, executed in topological order, and read from, behaving as a pure function of its inputs. `If`, `Loop` and `Scan` exist, but as second-class constructs. Their bodies are attributes rather than values, so a branch cannot be partitioned across backends the way straight-line regions can, and an accelerator that compiles its partition ahead of time cannot claim a region whose trip count is unknown.

Modern language and multimodal models are stateful and autoregressive, and therefore depart from the assumption the DAG model was shaped around. The departure affects export, not only representation. `torch.export` traces under symbolic shapes, and control flow survives only where it was written as `torch.cond` or `torch.while_loop`; neither construct appears anywhere in the `transformers` modeling code. A Python `if` therefore specializes to the branch it took, a `for` unrolls, shapes remain dynamic only where declared, and a data-dependent size fails the export outright. The resulting graph is valid, but its scope is narrower than that of the model it was taken from.

Four axes make this consequential. Each varies at runtime along a dimension the graph cannot vary over, and each is resolved the same way, by relocating the decision out of the graph and into the runtime. Table 1 records ONNXRuntime's answer in each case. We take ONNXRuntime as the reference point because, as the reference implementation of ONNX maintained by the organization that authored the format, it is the project best positioned to have resolved these *within* the graph.

**Table 1: Four axes of runtime variation, and where each is resolved.**

| Axis | What varies, and per what | ONNXRuntime's answer | Where the decision resides |
| ---- | ------------------------- | -------------------- | -------------------------- |
| **Persistent state** | The KV cache, threaded from one decode step into the next, per **step**. A turn is not a single DAG pass but a sequence of passes sharing memory. | `com.microsoft.GroupQueryAttention` hides the cache append in-kernel, past and present sharing one preallocated buffer so that it grows in place rather than being concatenated and copied. The loop around it moves out to a separate library, `onnxruntime-genai`, with its own `genai_config.json`. | A contrib op no converter reads, together with host code that every non-ORT runtime reimplements. |
| **Data-dependent routing** | An MoE router selects k of n experts, per **token**. A static graph must either evaluate every expert and mask (dense cost for sparse compute) or gather into a compact batch (data-dependent shapes). | `com.microsoft.MoE` takes `router_probs` as an ordinary input and performs top-k selection inside the kernel, so the node remains static while the data dependence occurs where ONNX cannot observe it. | A contrib op: one opaque node with a fixed notion of what an expert is. |
| **Data-dependent control flow** | Mixture-of-Depths skips a block entirely; an early-exit LM stops descending the stack, per **token**. | None. Neither a contrib op nor a runtime mechanism exists (`onnxruntime-genai` stops early per *sequence*, not per token), so these architectures export dense. | Nowhere: the saving does not survive the export. |
| **Config-dependent weights** | Which LoRA adapter applies, per **request**, determined by configuration rather than by data. Merging (`W' = W + BA`) collapses to a static graph at the cost of one full weight set per adapter. | Adapters are demoted from initializers to graph *inputs*, forfeiting constant folding and weight pre-packing, and selected through `RunOptions.add_active_adapter` from a separate `.onnx_adapter` file. | A session API rather than an operator. |

`If` does not close rows 2 and 3, since its predicate expresses one decision per graph execution while routing requires one per token. Row 3 is the sharpest case: there is no plugin, hence no portability cost and no capability either. The graph translates cleanly precisely because the property worth exporting did not survive the export.

## 4 Case study: fused attention

Attention has shifted from an operator view toward a **module** view, comprising fused attention, KV-cache blocks and MoE routers. This is the plugin-centric turn in its clearest form. Every runtime ships and maintains its own fused-attention module rather than sharing one, and while the implementations agree on the mathematics, no two agree on the boundary, specifically on where the KV cache lives relative to the operator.

**Table 2: Three fused-attention modules, and the placement of the KV cache in each.**

| Runtime | Fused attention | Where the KV cache lives |
| ------- | --------------- | ------------------------ |
| **ONNXRuntime** | `Attention` / `MultiHeadAttention` / `GroupQueryAttention` contrib ops, in [`contrib_ops/cpu/bert`](https://github.com/microsoft/onnxruntime/tree/v1.27.1/onnxruntime/contrib_ops/cpu/bert) | **Inside the operator.** The cache appears as graph tensors (`past_key`/`past_value` in, `present_key`/`present_value` out), but when past and present are the *same* tensor it is sized to `max_sequence_length` and the kernel appends in place. Given `seqlens_k`, `total_sequence_length` and `do_rotary`, the same kernel also applies RoPE. The host allocates the buffer; the operator determines what happens to it and how far into it the current position lies. |
| **TensorRT-LLM** | [`gptAttentionPlugin`](https://github.com/NVIDIA/TensorRT-LLM/tree/v1.2.1/cpp/tensorrt_llm/plugins/gptAttentionPlugin), over [`cpp/tensorrt_llm/kernels`](https://github.com/NVIDIA/TensorRT-LLM/tree/v1.2.1/cpp/tensorrt_llm/kernels) | **Beside the operator.** There are no KV tensors in that sense: with paged KV the cache is a pool of blocks issued per request by a cache manager, and the plugin receives the block offsets and the host-side metadata needed to locate them. Shape and behavior are fixed in plugin *fields* at build time rather than expressed in a portable signature. |
| **OpenVINO** | [`ScaledDotProductAttention`](https://docs.openvino.ai/2026/documentation/openvino-ir-format/operation-sets/operation-specs/sequence/scaled-dot-product-attention.html) | **Outside the graph's I/O.** The operator carries the mathematics alone: `query`/`key`/`value`, an optional mask and scale, and a `causal` flag. The cache is *state*, expressed as `ReadValue`/`Assign` pairs on a `Variable`, carried between `infer()` calls and reachable only through `query_state()`. Serving stacks then rewrite that again: `ov::pass::SDPAToPagedAttention` exchanges the state for a 28-input `PagedAttentionExtension` and block tables. One vendor, two incompatible KV contracts. |

The incompatibility is not one of naming but of ownership: which component allocates the cache, which advances the position, and which tracks sequence length. A converter cannot mechanically rewrite one form into another, because what it would be rewriting is not the same node. It is a different answer to the question of where the runtime ends and the graph begins.

The module level is therefore where the performance resides and, being plugin-centric, where portability ends. A graph containing `GroupQueryAttention` is an ONNXRuntime graph rather than an ONNX one. Decomposing it back into `MatMul`, `Softmax` and `Concat` restores translatability at the price of concatenate-and-copy KV growth, the fused kernel, and frequently the accelerator partition along with them.

## 5 The cost at the converter boundary

With no standard at the module level, model publishers, agent frameworks and distribution hubs each reimplement the same per-target conversion glue. Every converter in Table 3 reads standard ONNX and nothing else, so an export leaves the table as soon as it uses the plugin that made the model fast.

**Table 3: ONNX consumers, all of which read the standard operator set only.**

| Path            | Converter                                                                                               |
| --------------- | ------------------------------------------------------------------------------------------------------- |
| ONNX → QNN      | [ONNX2DLC](https://docs.qualcomm.com/doc/80-63442-10/topic/converters.html#onnx-conversion)             |
| ONNX → TensorRT | [ONNX2TRT](https://github.com/onnx/onnx-tensorrt)                                                       |
| ONNX → OpenVINO | [ONNX2OVIR](https://docs.openvino.ai/2026/openvino-workflow/model-preparation/convert-model-onnx.html)  |
| ONNX → IREE     | [ONNX2MLIR](https://iree.dev/guides/ml-frameworks/onnx/)                                                |

This leaves a dilemma that the [approach](approach.md) addresses: a fast export bound to one runtime, or a portable export that has discarded the reason for exporting at all.

## References

[1] T. Chen et al. *TVM: An Automated End-to-End Optimizing Compiler for Deep Learning*. OSDI, 2018. [arXiv:1802.04799](https://arxiv.org/abs/1802.04799)

---

**Next:** [Approach](approach.md) — the module boundary as the unit of export.
