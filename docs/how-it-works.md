# 🔧 How it works

*Export stages, the graph at each step, and the Rust runtime.* — [back to README](../README.md)

---

## Example: CausalLM exporter

Every supported model inherits `CausalLMExporter` (`src/hf2mobile/exporter/causallm.py`), which drives a five-stage export. A single run traces **two cases** — **prefill** (the full prompt) and **generation** (single-token decode with KV cache in/out) — because those are the two distinct shapes a decoder actually runs at inference time. For `ORT` the deliverable is the generation graph alone, whose dynamic `L` covers both.

| Stage | What happens | Where it lives |
| ----- | ------------ | -------------- |
| **1. Trace module I/O** | Run `model.generate(...)` once and record the inputs/outputs of each semantic module (the "plugin" boundaries). | `src/hf2mobile/tracing/` |
| **2. Export subgraphs** | Emit each traced module as a standalone ONNX subgraph, specialized for the target. | `src/hf2mobile/exporter/submodules/` |
| **3. Register plugin ops** | Register the custom ops so the main-graph export can reference them by name instead of inlining operators. | `src/hf2mobile/tracing/register.py` |
| **4. Export model cases** | `torch.onnx.export` the model for each unique case (prefill + generation), with a KV-cache-aware forward so the cache I/O survives dead-code elimination. | `src/hf2mobile/exporter/causallm.py` |
| **5. Merge + postprocess** | Inline the subgraphs into each main graph, then run target-specific fusions and postprocessing. | `src/hf2mobile/exporter/onnx/` |

Stage 5 is where the target specialization becomes concrete — e.g. for `ORT`:

- **Fusion** (`onnx/fusion/`): fuse RMSNorm, RoPE and Group-Query-Attention into ORT contrib ops (`fuse_rms_norm`, `fuse_rope`, `fuse_group_query_attention`).
- **Postprocess** (`onnx/postprocess/`): make I/O shapes dynamic and attach the sliding-window mask (`attach_sliding_window_mask_onnx`).

Adding a new architecture is usually a thin subclass of `CausalLMExporter` (see `llama.py`, `qwen2.py`, `qwen3.py`, `gemma3.py`); adding a new target is mostly new branches under `onnx/fusion` and `onnx/postprocess`.

> [!NOTE]
> **On memory.** Stages 4–5 hold the graph as a single `onnx_ir.Model` from load to save. `ir.load` mmaps external tensors — the weights are page cache the OS can evict, not heap — and every fusion and postprocess mutates that one live model in place, so there is no save/reload round-trip between stages.

Then, past the exporter:

| Stage | What happens | Where it lives |
| ----- | ------------ | -------------- |
| **6. Bake the decode policy** | Append a `SampleLogits` node so the graph returns a token id, and park the EOS ids in the graph as a `Constant`. | `src/hf2mobile/postprocess.py` |
| **7. Run it** | Tokenize, prefill, decode, detokenize — in Rust, over the postprocessed graph. | `src/onnx_inferencer/` |

The graph progresses through all seven stages like this:

| <img src="1_module_level_graph.svg" width="200"> | <img src="2_module_level_postprocessed_graph.svg" width="200"> | <img src="3_final_graph.svg" width="200"> | <img src="4_final_graph_postprocessed.svg" width="200"> |
| :---: | :---: | :---: | :---: |
| Initial module-level graph | Postprocessed module-level graph | Flattened operator-level graph (ORT: CPU-EP target) | Decode policy baked in (`SampleLogits` + EOS ids) |

---

## Example: CausalLM inferencer

The exported graph is only half the deliverable; a runtime has to load it. `hf2mobile` ships two Rust crates that share one source file:

| Source | Artifact | Role |
| ------ | -------- | ---- |
| `src/onnx_inferencer/` | `hf2mobile._ortrs_binding` (a Python extension module, built by maturin) | *Drives* ONNXRuntime — session setup, KV cache, prefill/decode loop, tokenizer, timing. Backs `python -m hf2mobile.infer`. |
| `src/onnx_inferencer/` | `hf2mobile-infer` (a standalone executable, built by cargo) | The same engine with no interpreter, so it cross-compiles: `adb push` it to a phone with an export directory and run the graph on the device it was built for. |
| `src/onnx_plugins/` | `libhf2mobile_plugins.so` | *Driven by* ONNXRuntime — a custom-op library exporting the C `RegisterCustomOps` entry point, loadable from Python, C++ or an Android app. |

They are separate crates (and separate cargo workspaces) because they need opposite `ort` configurations: the runtime dlopens onnxruntime, while the plugin is already running inside it. But `sample_logits.rs` is compiled into **both**, so the token a mobile runtime picks and the token the dev runtime picks come from one definition.

Nothing tells the engine *how* to decode. The policy is a `SampleLogits` node and the stop ids are a `hf2mobile_EOS_tokens` constant, both baked into the graph by postprocess:

```text
logits [1, L, vocab]  --SampleLogits(top_k, top_p, temperature)-->  sampled_token [1, 1] int32
```

A 262k-wide fp32 logits row is 1 MB per token, so reducing it to one integer *inside* the graph is the copy the decode loop most wants back. What the engine adds around that: a zero-copy KV cache (tensors stay ORT-side between steps), one session driving both prefill and decode over the exporter's dynamic-`L` graph, and TTFT/tok-s per run. `DEBUG=1` also dumps the Level3-optimized graph and a `chrome://tracing` profile.

Two front ends, one engine — a cargo feature picks which is compiled:

### HostPC

```bash
hf2mobile-inference 2026-08-01__ORT__Qwen-Qwen3-0.6B --prompt "Where is Paris?"
```

```python
from hf2mobile.inference import CausalLMInferencer

lm = CausalLMInferencer("inference.onnx", "tokenizer.json")
text, (ttft_s, tps) = lm.generate("Where is Paris?", num_generation=64)
```

### Android

`hf2mobile-infer` is the same engine with the pyo3 layer swapped for a CLI, so it cross-compiles — no interpreter, no app, no JNI, and no `libhf2mobile_plugins.so` (the operator is compiled in). Three files go to the phone: the binary, an `arm64-v8a` `libonnxruntime.so` (dlopened, so it is found beside the binary at runtime), and the export directory.

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

The binary applies the model's chat template itself — the job `transformers` does on the host — rendering the export's own Jinja, so the prompt reaching the model is token-for-token what the Python path produces.

---

**See also:** [🚀 Usage (CLI)](../README.md#-usage-cli) — the same stages as commands to run.
