# hf2mobile

Export HuggingFace torch models to mobile targeted (ONNXRuntime - CPUEP, QNNEP) ONNX graphs.


## Tested Architectures
- `LlamaForCausalLM`
- `Qwen2ForCausalLM`
- `Qwen3ForCausalLM`
- `Gemma3ForCausalLM`

---

## Requirements

- Python **>= 3.12**
- [`uv`](https://docs.astral.sh/uv/) for dependency management and builds

The package pins CPU-only PyTorch wheels by default (see the `[tool.uv.index]` block in `pyproject.toml`).
If you would like to use CUDA wheels instead remove that block.

---

## Install

### From a built wheel (production)

```bash
uv pip install dist/hf2mobile-0.1.0-py3-none-any.whl
```

### From source (development)

```bash
uv sync
source .venv/bin/activate
```

---

## Build the wheel

```bash
uv build
```

This produces both a wheel and an sdist in `dist/`:

```
dist/
├── hf2mobile-0.1.0-py3-none-any.whl
└── hf2mobile-0.1.0.tar.gz
```

---

## Usage (CLI)

Export a mobile CPU/NPU targeted ONNX graph.

```bash
python3 -m hf2mobile.export {HF repo id} --target {ORT/QNN}
# OR
hf2mobile-export {HF repo id} --target {ORT/QNN}
```

### Options

| Argument            | Description                                                           | Default |
| ------------------- | --------------------------------------------------------------------- | ------- |
| `repo_id`           | Hugging Face repo id of the model to export.                          | —       |
| `-t`, `--target`    | Export target: `ORT` or `QNN`. Graph topology may change base on it   | `ORT`   |
| `--debug`           | Force a tiny random-init model (2 layers) + DEBUG logging for a fast smoke test. | off |

### Examples

```bash
# Gemma 3 (270M) for ONNXRuntime
python3 -m hf2mobile.export google/gemma-3-270m-it --target ORT

# Qwen2.5 (0.5B) for ONNXRuntime
python3 -m hf2mobile.export Qwen/Qwen2.5-0.5B-Instruct --target ORT

# Fast smoke test with a tiny random-init model
python3 -m hf2mobile.export meta-llama/Llama-3.2-1B-Instruct --target ORT --debug
```
