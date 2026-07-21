# hf2mobile

Export Hugging Face causal language models to hardware-targeted ONNX graphs.

`hf2mobile` loads a causal LM from the Hugging Face Hub, traces it, and emits ONNX
model(s) tuned for a specific inference target (`ORT` for ONNX Runtime, `QNN`
for Qualcomm NPUs). Supported architectures: **Llama**, **Qwen2**, **Qwen3**,
and **Gemma3**.

---

## Requirements

- Python **>= 3.12**
- [`uv`](https://docs.astral.sh/uv/) for dependency management and builds

The package pins CPU-only PyTorch wheels by default (see the `[tool.uv.index]`
block in `pyproject.toml`). Remove that block to resolve CUDA wheels instead.

---

## Install

### From a built wheel (production)

```bash
# Build the wheel (see "Build" below), then install it into any environment:
uv pip install dist/hf2mobile-0.1.0-py3-none-any.whl
# or with plain pip:
pip install dist/hf2mobile-0.1.0-py3-none-any.whl
```

### From source (development)

```bash
# Create the environment and install the project + dev tools, locked & reproducible.
uv sync

# The project is installed in editable mode inside .venv; activate it if you like:
source .venv/bin/activate
```

`uv sync` reads `pyproject.toml`, resolves the dependency graph, writes a
`uv.lock` (commit it for reproducible installs), and materializes `.venv`.

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

The `.onnx` template graph shipped inside the package is bundled automatically.

---

## Usage (CLI)

Export a model by passing its Hugging Face repo id. Run it as a module:

```bash
python3 -m hf2mobile.export google/gemma-3-270m-it --target ORT
```

or via the installed console script:

```bash
hf2mobile-export google/gemma-3-270m-it --target ORT
```

### Options

| Argument            | Description                                                        | Default |
| ------------------- | ------------------------------------------------------------------ | ------- |
| `repo_id`           | Hugging Face repo id of the model to export.                       | —       |
| `-t`, `--target`    | Export target: `ORT` or `QNN`.                                     | `ORT`   |
| `--debug`           | Force a tiny random-init model (2 layers) + DEBUG logging for a fast smoke test. | off |

### Examples

```bash
# Gemma 3 (270M) for ONNX Runtime
python3 -m hf2mobile.export google/gemma-3-270m-it --target ORT

# Qwen2.5 (0.5B) for ONNX Runtime
python3 -m hf2mobile.export Qwen/Qwen2.5-0.5B-Instruct --target ORT

# Fast smoke test with a tiny random-init model
python3 -m hf2mobile.export Qwen/Qwen3-0.6B --target ORT --debug
```

Exported ONNX files (`case0.onnx`, `case1.onnx`, …) are written to the current
working directory.

---

## Python API

```python
from hf2mobile.export import export

export("google/gemma-3-270m-it", target="ORT", debug=False)
```

---

## Reproduce guide (end to end)

```bash
# 1. Get the code
git clone <repo-url> && cd hf-quantizer

# 2. Create a locked, reproducible environment (CPU torch by default)
uv sync

# 3. Build the production wheel
uv build

# 4. Install the wheel into a clean environment and run
uv pip install dist/hf2mobile-0.1.0-py3-none-any.whl
python3 -m hf2mobile.export google/gemma-3-270m-it --target ORT
```
