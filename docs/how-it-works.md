# How it works

*Export stages, the graph at each step, and the Rust runtime.* — [back to README](../README.md)

---

## 1 Case study: the CausalLM exporter

Every supported model inherits `CausalLMExporter` (`src/hf2mobile/exporter/causallm.py`), which drives a five-stage export. A single run traces **two cases**, prefill (the full prompt) and generation (single-token decode with KV cache in and out), because those are the two distinct shapes a decoder executes at inference time. For the `ORT` target the deliverable is the generation graph alone, whose dynamic `L` covers both.

**Table 1: The five exporter stages.**

| Stage | What happens | Where it lives |
| ----- | ------------ | -------------- |
| **1. Trace module I/O** | Run `model.generate(...)` once and record the inputs and outputs of each semantic module, that is, the plugin boundaries. | `src/hf2mobile/tracing/` |
| **2. Export subgraphs** | Emit each traced module as a standalone ONNX subgraph, specialized for the target. | `src/hf2mobile/exporter/submodules/` |
| **3. Register plugin ops** | Register the custom ops so that the main-graph export references them by name instead of inlining their operators. | `src/hf2mobile/tracing/register.py` |
| **4. Export model cases** | Run `torch.onnx.export` for each unique case (prefill and generation), with a KV-cache-aware forward so that the cache I/O survives dead-code elimination. | `src/hf2mobile/exporter/causallm.py` |
| **5. Merge and postprocess** | Inline the subgraphs into each main graph, then apply target-specific fusions and postprocessing. | `src/hf2mobile/exporter/onnx/` |

Stage 5 is where target specialization becomes concrete. For the `ORT` target it comprises two passes:

- **Fusion** (`onnx/fusion/`): fuse RMSNorm, RoPE and grouped-query attention into ORT contrib ops (`fuse_rms_norm`, `fuse_rope`, `fuse_group_query_attention`).
- **Postprocess** (`onnx/postprocess/`): make the I/O shapes dynamic and attach the sliding-window mask (`attach_sliding_window_mask_onnx`).

Adding an architecture is normally a thin subclass of `CausalLMExporter` (see `llama.py`, `qwen2.py`, `qwen3.py`, `gemma3.py`), and adding a target is largely new branches under `onnx/fusion` and `onnx/postprocess`.

**On memory.** Stages 4 and 5 hold the graph as a single `onnx_ir.Model` from load to save. `ir.load` memory-maps external tensors, so the weights are page cache the kernel is free to evict rather than heap, and every fusion and postprocessing pass mutates that one live model in place, with no save-and-reload round trip between stages. Exporting a multi-gigabyte model therefore no longer materializes its weights once per stage.

Two further stages follow the exporter proper.

**Table 2: The postprocess and runtime stages.**

| Stage | What happens | Where it lives |
| ----- | ------------ | -------------- |
| **6. Bake the decode policy** | Append a `SampleLogits` node so that the graph returns a token id, and place the EOS ids in the graph as a `Constant`. | `src/hf2mobile/postprocess.py` |
| **7. Run it** | Tokenize, prefill, decode and detokenize, in Rust, over the postprocessed graph. | `src/onnx_inferencer/` |

Figure 1 shows the graph as it progresses through these stages.

| <img src="1_module_level_graph.svg" width="200"> | <img src="2_module_level_postprocessed_graph.svg" width="200"> | <img src="3_final_graph.svg" width="200"> | <img src="4_final_graph_postprocessed.svg" width="200"> |
| :---: | :---: | :---: | :---: |
| Initial module-level graph | Postprocessed module-level graph | Flattened operator-level graph (ORT: CPU-EP target) | Decode policy baked in (`SampleLogits` + EOS ids) |

**Figure 1: The exported graph at four points in the pipeline.**

---

## 2 Case study: the CausalLM runtime

The exported graph is only half of the deliverable, since a runtime has to load it. `hf2mobile` ships two Rust crates that share one source file.

**Table 3: The two runtime crates and the plugin library.**

| Crate | Artifact | Role |
| ----- | -------- | ---- |
| `src/onnx_inferencer/` | `hf2mobile._ortrs_binding`, a Python extension module built by maturin | *Drives* ONNXRuntime: session setup, KV cache, the prefill and decode loop, tokenizer, and timing. Backs `python -m hf2mobile.infer`. |
| `src/onnx_inferencer/` | `hf2mobile-infer`, a standalone executable built by cargo | The same engine without an interpreter, so that it cross-compiles: `adb push` it to a phone together with an export directory and run the graph on the device it was built for. |
| `src/onnx_plugins/` | `libhf2mobile_plugins.so` | *Driven by* ONNXRuntime: a custom-op library exporting the C `RegisterCustomOps` entry point, loadable from Python, C++ or an Android app. |

They are separate crates, in separate cargo workspaces, because they require opposite `ort` configurations: the runtime dlopens onnxruntime, whereas the plugin is already executing inside it. `sample_logits.rs` is nonetheless compiled into both, so that the token a mobile runtime selects and the token the development runtime selects originate from one definition.

Nothing informs the engine *how* to decode. The policy is a `SampleLogits` node and the stop condition is a `hf2mobile_EOS_tokens` constant, both baked into the graph during postprocessing:

```text
logits [1, L, vocab]  --SampleLogits(top_k, top_p, temperature)-->  sampled_token [1, 1] int32
```

A 262k-wide fp32 logits row is 1 MB per token, so reducing it to a single integer *inside* the graph eliminates the copy that the decode loop is most sensitive to. Around that, the engine adds a zero-copy KV cache (tensors remain ORT-side between steps), one dynamic-`L` graph serving both prefill and decode, and TTFT and tokens-per-second measurements per run. Setting `DEBUG=1` additionally dumps the Level3-optimized graph and a `chrome://tracing` profile.

There are two front ends over one engine, and a cargo feature selects which is compiled.

### 2.1 Host

```bash
hf2mobile-inference 2026-08-01__ORT__Qwen-Qwen3-0.6B --prompt "Where is Paris?"
```

```python
from hf2mobile.inference import CausalLMInferencer

lm = CausalLMInferencer("inference.onnx", "tokenizer.json")
text, (ttft_s, tps) = lm.generate("Where is Paris?", num_generation=64)
```

### 2.2 Android

`hf2mobile-infer` is the same engine with the pyo3 layer replaced by a CLI, so that it cross-compiles: no interpreter, no app, no JNI, and no `libhf2mobile_plugins.so`, since the operator is compiled in. Three files go to the phone: the binary, an `arm64-v8a` `libonnxruntime.so` (dlopened, and therefore found beside the binary at runtime), and the export directory.

```bash
nix develop .#android -c cargo build --release --target aarch64-linux-android --bin hf2mobile-infer

D=/data/local/tmp/hf2mobile && adb shell mkdir -p $D
adb push target/aarch64-linux-android/release/hf2mobile-infer libonnxruntime.so $D/
adb push 2026-08-01__ORT__google-gemma-3-270m-it $D/
adb shell chmod +x $D/hf2mobile-infer

adb shell "$D/hf2mobile-infer $D/2026-08-01__ORT__google-gemma-3-270m-it --prompt 'Where is Paris?'"
```

```text
[hf2mobile] INFO     | loaded 39 inputs / 37 outputs (36 KV cache slots) | eos: [1, 106]
Paris is a French city, which is known for its iconic landmarks and rich history.
[hf2mobile] INFO     | 14 prompt tokens, 18 generated (EOS) | TTFT 113.9 ms | 13.38 tok/s
```

That run is the same binary executing on the host. A device's numbers will differ; the point is that the two are directly comparable.

The binary applies the model's chat template itself, performing the work `transformers` does on the host by rendering the export's own Jinja, so that the prompt reaching the model is token-for-token what the Python path produces.

---

**See also:** [Usage (CLI)](../README.md#-usage-cli) — the same stages expressed as commands.
