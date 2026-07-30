"""Rust-backed ONNXRuntime inference.

The heavy lifting lives in the compiled `hf2mobile._ortrs_binding` extension (Rust, built by maturin from ``src/onnx_inferencer``).
That extension links ONNXRuntime dynamically at runtime via `ORT_DYLIB_PATH` to avoid making the user install a second copy of ORT,
we point it at the `libonnxruntime.so` that the pip `onnxruntime` package already ships. This must happen *before* importing `._ortrs_binding`.
"""

import glob
import os

import onnxruntime as _ort_py  # noqa: F401  (imported only to locate its bundled .so)


def _ensure_ort_dylib() -> None:
    if os.environ.get("ORT_DYLIB_PATH"):
        return  # respect an explicit override
    capi = os.path.join(os.path.dirname(_ort_py.__file__), "capi")

    # NOTE: e.g. libonnxruntime.so, libonnxruntime.so.1, libonnxruntime.so.1.27.0
    libs = sorted(glob.glob(os.path.join(capi, "libonnxruntime.so*")))
    if not libs:
        raise ImportError(
            f"could not find libonnxruntime.so under {capi!r}; " "install `onnxruntime` or set ORT_DYLIB_PATH manually"
        )
    os.environ["ORT_DYLIB_PATH"] = libs[-1]


_ensure_ort_dylib()

# `..` (not `.`): the extension is built as `hf2mobile._ortrs_binding`, which is the
# parent package of this `hf2mobile.inference` subpackage.
from .._ortrs_binding import CausalLMInferencer  # noqa: E402  (must follow _ensure_ort_dylib)

__all__ = ["CausalLMInferencer"]
