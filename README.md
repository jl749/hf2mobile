# hf2mobile

**Export HuggingFace `transformers` LMs into mobile-targeted ONNX graphs (ONNXRuntime CPU-EP, QNN-EP).**

![Python](https://img.shields.io/badge/python-%3E%3D3.12-3776AB?logo=python&logoColor=white)
![Rust](https://img.shields.io/badge/rust-extension-000000?logo=rust&logoColor=white)
![ONNX Runtime](https://img.shields.io/badge/onnxruntime-%3E%3D1.22-005CED?logo=onnx&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

```bash
hf2mobile-export    Qwen/Qwen3-0.6B --target ORT --export_dtype float32           # trace + ORT fusions, then bake in sampling + EOS ids -> inference.onnx
hf2mobile-inference 2026-08-01__ORT__Qwen-Qwen3-0.6B --prompt "Where is Paris?"   # rust runtime: graph + tokenizer, nothing else
```

---

## 📖 Table of Contents

- [🎯 Motivation](docs/motivation.md) — why operator-level ONNX stopped being enough
- [🧩 Approach](docs/approach.md) — the module boundary as the unit of export
- [🔧 How it works](docs/how-it-works.md) — the export stages, and the Rust runtime that runs the result on a phone
- [📋 Requirements](#-requirements)
- [📦 Install](#-install)
- [🚀 Usage (CLI)](#-usage-cli) — export → postprocess → infer, on the host or over `adb`
- [🧭 Roadmap](#-roadmap)

---

## 🎯 Motivation

**[docs/motivation.md](docs/motivation.md)** — why operator-level ONNX stopped being enough.

ONNX standardized the *operator* level, and that was enough while the vocabulary stayed small and universal. Modern LLMs are stateful and autoregressive, so the ecosystem went **plugin-centric** instead: every runtime grew its **own** extensions — fused attention, side-car configs, session APIs — to express what the standard vocabulary cannot. The module level is where the performance lives, and — being plugin-centric — where portability stops.

---

## 🧩 Approach

**[docs/approach.md](docs/approach.md)** — the module boundary as the unit of export.

Trace the model as its semantic **modules** (attention, RoPE, RMSNorm, the LM head) rather than a flat operator soup, hold each as a single node, and expand it per target at export time. The portable artifact is the module-level trace, not any one file it produces — which is also what makes the NPU/CPU seam an export-time decision instead of whatever a converter happens to claim.

---

## 🔧 How it works

**[docs/how-it-works.md](docs/how-it-works.md)** — the export stages, and the Rust runtime that runs the result on a phone.

Seven stages: trace module I/O → export the subgraphs → register the plugin ops → export the prefill/generation cases → merge, fuse and postprocess → bake the decode policy into the graph → run it. The runtime is two Rust crates over one engine: a Python extension module for the host, and a standalone `aarch64-linux-android` binary for the phone.

| <img src="docs/1_module_level_graph.svg" width="200"> | <img src="docs/2_module_level_postprocessed_graph.svg" width="200"> | <img src="docs/3_final_graph.svg" width="200"> | <img src="docs/4_final_graph_postprocessed.svg" width="200"> |
| :---: | :---: | :---: | :---: |
| Initial module-level graph | Postprocessed module-level graph | Flattened operator-level graph (ORT: CPU-EP target) | Decode policy baked in (`SampleLogits` + EOS ids) |

---

## 📋 Requirements

- Python **>= 3.12**
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- A **Rust toolchain** + [`maturin`](https://www.maturin.rs/) — only to build from source; `hf2mobile` is a mixed Rust/Python project, and a build compiles the `_ortrs_binding` extension module

Everything above comes from `flake.nix`; nothing needs to be installed globally:

```bash
# for development on host
nix develop

# for android deployment (installs additional few GB - NDK, adb, ...)
nix develop .#android
```

> Pre-built ONNXRuntime so for `arm64-v8a` can be downloaded from [onnxruntime-android AAR](https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/) (Find `jni/arm64-v8a/libonnxruntime.so`).

---

## 📦 Install

### HostPC

For development:

```bash
nix develop
uv sync                     # Python dependencies
maturin develop --release   # compile the Rust extension into the venv
```

Re-run `maturin develop --release` after any change under `src/onnx_inferencer/` — Python imports the *installed* `.so`, so a bare `cargo build` will not be picked up.

For testing:

```bash
nix develop

# 1. Build the whl and so
maturin build --release -o dist/                                    # build the wheel

# 2. Install
uv pip install dist/hf2mobile-0.1.0-cp312-abi3-linux_x86_64.whl

# 3. Run Inference
## METHOD1 (use hf2mobild.infer module)
hf2mobile-inference {export dir} --prompt "Where is Paris?"
## METHOD2 (rewrite inference loop with python api)
cargo build --release --manifest-path src/onnx_plugins/Cargo.toml   # libhf2mobile_plugins.so
python3 << 'EOF'
import onnxruntime as ort
opts = ort.SessionOptions()
opts.register_custom_ops_library("src/onnx_plugins/target/release/libhf2mobile_plugins.so")
session = ort.InferenceSession("{export dir}/inference.onnx", opts)
EOF
```

### Android

Nothing is *installed* on the device — one binary is cross-compiled and pushed (see [Usage](#-usage-cli)):

```bash
nix develop .#android

# 1. Build the android so
cargo build --release --target aarch64-linux-android --bin hf2mobile-infer
file target/aarch64-linux-android/release/hf2mobile-infer # ELF 64-bit LSB pie executable, ARM aarch64, interpreter /system/bin/linker64, for Android 24

# 2. Install
ORT=1.27.0  # match the host: python -c 'import onnxruntime; print(onnxruntime.__version__)'
curl -sLO https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/$ORT/onnxruntime-android-$ORT.aar
unzip -j onnxruntime-android-$ORT.aar 'jni/arm64-v8a/libonnxruntime.so' -d .
file libonnxruntime.so # ELF 64-bit LSB shared object, ARM aarch64, for Android 24, stripped

# 3. Run Inference
D=/data/local/tmp/hf2mobile
adb shell mkdir -p $D
adb push target/aarch64-linux-android/release/hf2mobile-infer libonnxruntime.so $D/
adb shell chmod +x $D/hf2mobile-infer
```

`.cargo/config.toml` points cargo and the `cc` crate at the NDK's API-24 clang — API 24 (Android 7.0) is the floor ONNXRuntime's own Android builds target. `llvm-strip` roughly halves the 9 MB if the push is slow. The same source built for the host is just `cargo build --release`.

---

## 🚀 Usage (CLI)

Supported architectures: `LlamaForCausalLM`, `Qwen2ForCausalLM`, `Qwen3ForCausalLM`, `Gemma3ForCausalLM`.

<details>
<summary><b>💻 HostPC</b> — export, then run it in the venv</summary>

#### 1. `hf2mobile.export` — HF → ONNX

```bash
nix develop .#default

hf2mobile-export {HF repo id} --target {ORT|QNN}
```

| Argument         | Description                                                              | Default |
| ---------------- | ----------------------------------------------------------------------- | ------- |
| `repo_id`        | Hugging Face repo id of the model to export.                            | —       |
| `-t`, `--target` | Export target: `ORT` or `QNN`. Graph topology may change based on it.   | `ORT`   |
| `--export_dtype` | `float32` / `float16` / `bfloat16`. Casts the weights before tracing, fixing the exported graph's dtype. | the model's own dtype |
| `--debug`        | Force a tiny random-init model (2 layers) + DEBUG logging for a fast smoke test. Also keeps the intermediate graphs. | off |
| `--skip_postprocess` | Stop after the ONNX export, leaving `inference.onnx` to a separate `hf2mobile.postprocess` run. | off |
| *(postprocess flags)* | `--top_k`, `--top_p`, `--temp`, `--eos_tokens`, `--keep_logits`, `-o` — the same parser as [2.](#2-hf2mobilepostprocess--bake-in-the-decode-policy), reused here. | from `generation_config.json` |

Output lands in a timestamped directory named `{date}__{target}__{model}`:

```
2026-08-01__ORT__Qwen-Qwen3-0.6B/
├── case2.onnx + case2.onnx.data           the dynamic-L generation graph (serves prefill and decode both)
├── inference.onnx + inference.onnx.data   case2 + the baked-in decode policy — what you ship (see 2.)
├── tokenizer.json                         handed to the runtime, which does all the tokenizing
├── tokenizer_config.json                  the chat template
└── generation_config.json                 the sampling policy + EOS ids that `postprocess` reads
```

Two cases are traced — prefill (`case1`) and generation (`case2`) — but the ORT deliverable is the single dynamic-`L` generation graph that serves both, so `case1.onnx` is deleted at the end of the export. `--debug` keeps a `debug__case1.onnx` / `debug__case2.onnx` snapshot as well as `case1.onnx`, plus the standalone module subgraphs.

#### 2. `hf2mobile.infer` — run it

```bash
nix develop .#default

hf2mobile-inference {export dir} --prompt "Where is Paris?"
```

| Argument           | Description                                                                  | Default |
| ------------------ | ---------------------------------------------------------------------------- | ------- |
| `export_dir`       | A **postprocessed** export directory (must hold `inference.onnx`).           | —       |
| `--prompt`         | User prompt.                                                                  | required |
| `--skip_template`  | Feed `--prompt` verbatim instead of through the tokenizer's chat template.    | off |
| `--num-generation` | Maximum tokens to generate.                                                   | `512` |
| `--intra-threads`  | ORT intra-op thread count.                                                    | one per core |

Streams to stdout, then reports prompt length, tokens generated, TTFT and tok/s. No sampling flags — the policy is in the graph.

#### Examples

```bash
# Gemma 3 (270M), end to end
hf2mobile-export google/gemma-3-270m-it --target ORT --export_dtype float32
hf2mobile-inference 2026-08-01__ORT__google-gemma-3-270m-it --prompt "Where is Paris?"

# Force greedy decoding regardless of what the model's config says
hf2mobile-export google/gemma-3-270m-it --target ORT --temp 0
hf2mobile-inference 2026-08-01__ORT__google-gemma-3-270m-it --prompt "Where is Paris?"
```

</details>

<details>
<summary><b>📱 Android</b> — export on the host, run it over <code>adb</code></summary>

#### 1. `hf2mobile.export` — HF → ONNX
```bash
nix develop .#android

hf2mobile-export {HF repo id} --target {ORT|QNN}
```

Supported architectures: `LlamaForCausalLM`, `Qwen2ForCausalLM`, `Qwen3ForCausalLM`, `Gemma3ForCausalLM`.

| Argument         | Description                                                              | Default |
| ---------------- | ----------------------------------------------------------------------- | ------- |
| `repo_id`        | Hugging Face repo id of the model to export.                            | —       |
| `-t`, `--target` | Export target: `ORT` or `QNN`. Graph topology may change based on it.   | `ORT`   |
| `--export_dtype` | `float32` / `float16` / `bfloat16`. Casts the weights before tracing, fixing the exported graph's dtype. | the model's own dtype |
| `--debug`        | Force a tiny random-init model (2 layers) + DEBUG logging for a fast smoke test. Also keeps the intermediate graphs. | off |
| `--skip_postprocess` | Stop after the ONNX export, leaving `inference.onnx` to a separate `hf2mobile.postprocess` run. | off |
| *(postprocess flags)* | `--top_k`, `--top_p`, `--temp`, `--eos_tokens`, `--keep_logits`, `-o` — the same parser as [2.](#2-hf2mobilepostprocess--bake-in-the-decode-policy), reused here. | from `generation_config.json` |

Output lands in a timestamped directory named `{date}__{target}__{model}`:

```
2026-08-01__ORT__Qwen-Qwen3-0.6B/
├── case2.onnx + case2.onnx.data           the dynamic-L generation graph (serves prefill and decode both)
├── inference.onnx + inference.onnx.data   case2 + the baked-in decode policy — what you ship (see 2.)
├── tokenizer.json                         handed to the runtime, which does all the tokenizing
├── tokenizer_config.json                  the chat template
└── generation_config.json                 the sampling policy + EOS ids that `postprocess` reads
```

Two cases are traced — prefill (`case1`) and generation (`case2`) — but the ORT deliverable is the single dynamic-`L` generation graph that serves both, so `case1.onnx` is deleted at the end of the export. `--debug` keeps a `debug__case1.onnx` / `debug__case2.onnx` snapshot as well as `case1.onnx`, plus the standalone module subgraphs.

#### 2. `hf2mobile-infer` - run it

The three files this needs — the cross-compiled binary, `libonnxruntime.so` and the export directory from step 1 — are pushed to `$D` in [Install → Android](#android).

```bash
nix develop .#android

D=/data/local/tmp/hf2mobile
adb push 2026-08-01__ORT__google-gemma-3-270m-it $D/   # step 1's output, if not already there
adb shell "$D/hf2mobile-infer $D/2026-08-01__ORT__google-gemma-3-270m-it \
           --prompt 'Where is Paris?' --num-generation 64"
# [hf2mobile] INFO     | loaded 39 inputs / 37 outputs (36 KV cache slots) | eos: [1, 106]
# Paris is a French city, which is known for its iconic landmarks and rich history.
# [hf2mobile] INFO     | 14 prompt tokens, 18 generated (EOS) | TTFT 113.9 ms | 13.38 tok/s
```

| Argument           | Description                                                                  | Default |
| ------------------ | ---------------------------------------------------------------------------- | ------- |
| `export_dir`       | A **postprocessed** export directory (must hold `inference.onnx`).           | —       |
| `--prompt`         | User prompt.                                                                  | required |
| `--skip_template`  | Feed `--prompt` verbatim instead of through the tokenizer's chat template. `--skip-template` is accepted too. | off |
| `--num-generation` | Maximum tokens to generate.                                                   | `512` |
| `--intra-threads`  | ORT intra-op thread count.                                                    | one per core |
| `--ort-dylib`      | Where to dlopen `libonnxruntime.so` from.                                     | `$ORT_DYLIB_PATH`, else beside the binary, else the export dir |

The same flags as [2.](#2-hf2mobileinfer--run-it) on the host — a device run is the host command with a different binary in front of it — plus `--ort-dylib`. Its fallback chain is why the push above needs nothing set in the environment. How to decode is not a flag on either side: the sampling policy and stop tokens are baked into the graph.

- **`/data/local/tmp`, not `/sdcard`** — the latter is mounted `noexec`.
- **Sweep `--intra-threads`.** On big.LITTLE the default (one thread per core) puts work on little cores that then hold the fast ones up; matching the big cluster is often quicker.
- **Compare a second run against the first.** Thermal throttling shows up as tok/s falling across a run, so one number on a warm phone is not a measurement.

</details>

---

## 🧭 Roadmap

Grouped by the axis each item unblocks. Checked items ship in the current export path; the rest are ordered roughly by priority within each group.

### 📱 QNN target (Qualcomm HTP)

- [ ] **ONNX → DLC conversion & compilation.** CLI to lower the exported ONNX to a Qualcomm DLC and compile per HTP target (v79, v81, …), fetching the resulting `EPContext` `.so`. This is what makes a QNN export actually loadable on-device.
- [ ] **QNN-scheme quantization.** Quantize following the QNN quantization scheme (HTP prefers `u16` activations over `u8` — see the Quantization group below).

### 🧠 ORT target (ONNXRuntime)

- [ ] **More contrib-op fusions.** Extend beyond RMSNorm/GQA to the remaining layer-norm family (`SkipLayerNormalization`, `SkipSimplifiedLayerNormalization`, `SimplifiedLayerNormalization`) so the graph maps onto ORT's optimized kernels.
- [ ] **MoE and LoRA plugins.** Add module exporters for data-dependent expert routing (MoE) and adapter weights (LoRA) — two of the module types the current static per-case export does not yet cover.
- [ ] **Per-token control flow.** Module exporters for Mixture-of-Depths and early-exit decoders, where whether a block runs at all is decided per token — the one axis in the [motivation](docs/motivation.md) with no de facto answer in any runtime today.
- [ ] **NPU/CPU EP partition.** Assign the fused attention node to the CPU EP and the statically-shaped blocks (QKV projections, MLP) to the NPU EP within a single session, so one graph covers both backends.
- [ ] **Multimodal support.** Extend beyond text-only causal LMs to vision-language models (`Qwen2.5-VL`, `Phi-4-multimodal`, …): a separately traced vision encoder feeding a decoder whose sequence length is set by the input image.
- [ ] **ORT-scheme quantization.** Quantize following the ORT quantization scheme.

### 🎚 Quantization

Targeting `u16` activations for QNN instead of `u8`, to preserve accuracy on HTP.

- [ ] **SmoothQuant (`u8s8`).** Migrate activation outliers into weights (fold diagonal matrix) so both sides quantize cleanly.
- [ ] **SpinQuant / QuaRot (`f16s8`).** Rotation-based outlier suppression (hadamard matrix), starting with the `R1` rotation only(WoQ) before adding the rest.
- [ ] **AWQ / GPTQ (`f16s8`).** Weight-only PTQ methods for the weight-quantized path.

### ⏱ Runtime & benchmarking

- [x] **Host-CPU runtime.** Rust ONNXRuntime engine (`hf2mobile._ortrs_binding`) with a zero-copy KV cache and an in-graph `SampleLogits` operator; reports TTFT and TPS per run.
- [x] **Loadable custom-op library.** `libhf2mobile_plugins.so` — the same operator, as a `.so` any ONNXRuntime binding can register.
- [x] **Android CLI.** `hf2mobile-infer` — the engine as a standalone `aarch64-linux-android` executable (no Python, no app), which renders the model's chat template itself and finds ONNX Runtime beside itself. `adb push` binary + `libonnxruntime.so` + export directory, and an exported model decodes on the device it was exported for.
- [ ] **Android CI / device verification.** The cross build and the produced ELF are checked; running it against a physical device is still manual. Also cross-compile `src/onnx_plugins` for `aarch64-linux-android`, which an app driving ORT through its Java API needs.
- [ ] **Benchmark harness.** Extend the per-run TTFT/TPS numbers into peak memory and a comparison across targets.
- [ ] **More logits dtypes.** `SampleLogits` has an fp32 kernel only; fp16 needs one registration each side.
- [ ] **Regression testing (pytest).** Extend the per-architecture suite under `tests/`, including numerical parity of the exported graph against the `transformers` baseline it was traced from.
