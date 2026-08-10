# 📊 Evaluation

*What hf2mobile's QNN path actually buys, measured on a Galaxy S25+.* — [back to README](../README.md)

---

Decode throughput for the two paths hf2mobile can export to, measured on the same phone:

- 🟦 **NPU + CPU path** — `--target QNN`. The graph is cut at every attention — `GroupQueryAttention` in the exported graph, which is what `torch.nn.functional.scaled_dot_product_attention` becomes after fusion. The attentions stay on the CPU, along with the token embedding, the LM head and the sampling op; each slab of decoder between them goes to the NPU. One session, two execution providers.
- 🟪 **CPU path** — `--target ORT`. The whole graph on ONNXRuntime's CPU execution provider; no NPU involvement.

## Setup

- **Device** — Galaxy S25+ (Snapdragon 8 Elite, HTP v79, 12 GB RAM)
- **Toolchain** — QAIRT SDK 2.48
- **Runtime** — ONNXRuntime 1.27 with the QNN execution provider, driven by `hf2mobile-infer`
- **Workload** — single-token prompt, 32 generated tokens

## Results

<img src="s25plus_decoder_throughput_comparison.svg" width="100%" alt="Decode throughput versus model size. W4 on the NPU is fastest at every size, from 59.5 tok/s at 0.6B to 17.5 at 4B; CPU fp32 reaches 24.8 at 0.6B and 9.0 at 1.7B and does not fit past that.">


Decode throughput (tok/s), higher is better:

| | 🟦 CPU+NPU fp16 | 🟦 CPU+NPU W8A8 | 🟦 CPU+NPU W4A8 | 🟦 CPU+NPU W4A16 | 🟪 CPU fp32 | best vs CPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 38.71 | 51.23 | **59.54** | 58.82 | 24.80 | **2.40×** |
| Qwen3-1.7B | 15.66 | 21.44 | **30.12** | 29.32 | 8.99 | **3.35×** |
| Qwen2.5-3B | 8.48 | 15.63 | **22.25** | 21.77 | n/a | — |
| Qwen3-4B | n/a | 10.76 | **17.49** | 17.45 | n/a | — |

The NPU is **2.4–3.4× the CPU wherever both run**, and past 1.7B it is the only thing that runs at all.

**NPU W4 is the fastest configuration at every size** — the clearest sign that this workload is memory-bound rather than compute-bound: fewer weight bits, proportionally more throughput.

On the missing cells:

- **No CPU fp16 column.** ONNXRuntime's CPU EP does not execute fp16 — it upconverts to fp32 at load (the graph comes back as `Cast → MatMul → Cast`), so the column would restate CPU fp32 while costing *more* memory. This is **not** a hardware limit. The 8 Elite reports `fphp`/`asimdhp` (FEAT_FP16) and the Android ORT binary ships MLAS half-precision GEMM (WIP: figure out why)
- **NPU fp16 at 4B** — the 6.8 GB context binary set crashes and reboots the phone. All context binaries are mapped at session creation, so the whole set has to fit.
- **CPU fp32 at 3B and 4B** — 12.4 GB and 16.1 GB against ~6.2 GB of available RAM. The 3B attempt ran for 4m11s and then rebooted the device; 4B fp32 cannot even be exported on the host (16.1 GB resident against 13 GB free, no swap).

### What each column is

| | weights | activations | graph I/O |
| --- | --- | --- | --- |
| 🟦 CPU+NPU fp16 | float16 | float16 | float16 |
| 🟦 CPU+NPU W8A8 | int8 symmetric, **per-row** | uint8 asymmetric, per-tensor | float16 |
| 🟦 CPU+NPU W4A8 | int4 symmetric, **per-row** | uint8 asymmetric, per-tensor | float16 |
| 🟦 CPU+NPU W4A16 | int4 symmetric, **per-row** | uint16 asymmetric, per-tensor | float16 |
| 🟪 CPU fp32 | float32 | float32 | float32 |

### Artifact size

Deployable bytes — NPU context binaries plus the fp16 main graph — against the CPU fp32 file:

| | 🟪 CPU fp32 | 🟦 CPU+NPU fp16 | 🟦 CPU+NPU W8 | 🟦 CPU+NPU W4 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 2.30 GB | 1.17 GB −49% | 748 MB −67% | **538 MB −77%** |
| Qwen3-1.7B | 6.50 GB | 3.30 GB −49% | 1.95 GB −70% | **1.28 GB −80%** |
| Qwen2.5-3B | 12.4 GB | 5.80 GB −53% | 3.20 GB −74% | **2.00 GB −84%** |
| Qwen3-4B | 16.1 GB | 7.55 GB −53% | 4.15 GB −74% | **2.55 GB −84%** |
