//! Checking that this machine can execute the model's float type.
//!
//! # Why this is a check and not a fix
//!
//! bf16 is a 16-bit float with fp32's exponent range and a truncated mantissa. Running it
//! needs *two* things, and only one is about the silicon:
//!
//! 1. a CPU with bf16 instructions, and
//! 2. an execution provider with bf16 kernels compiled in.
//!
//! ONNX Runtime's CPU provider has none, so a bf16 graph does not run slowly there — it
//! fails outright.
//!
//! This module used to rewrite such a graph to fp32 on the fly. It no longer does. The
//! dtype is decided once, at export (`hf2mobile.export --export_dtype`), where the weights
//! are still PyTorch tensors and casting is a one-liner. Doing it here meant re-parsing an
//! ONNX protobuf and its external-data sidecars at load time, to paper over a decision
//! made much earlier — and it left two places that could disagree about what dtype a model
//! is. The runtime's job is inference; the exporter's job is producing something runnable.

use anyhow::{bail, Result};
use ort::session::Session;
use ort::tensor::TensorElementType;

use crate::session::tensor_type;

/// Appended to dtype failures, including the one ORT raises from inside `commit_from_file`
/// when it cannot find a kernel. Phrased as the command to run next, because that is the
/// only part the reader can act on.
/// TODO: evaluate the backends (supported dtypes: ...)
pub const REEXPORT_ADVICE: &str =
    "ONNXRuntime's CPU execution provider has no bfloat16 kernels, so a bf16 graph cannot be \
     loaded on this machine at all. Re-export the model in a dtype the CPU provider can run:\n\
     \n    python -m hf2mobile.export <repo_id> --export_dtype float32\n\n\
     bf16 -> fp32 is lossless (fp32 has the same exponent range and more mantissa); the file \
     roughly doubles in size.";

/// Can we run a bf16 graph as-is?
pub fn bf16_is_executable() -> bool {
    cpu_has_bf16()
}

/// Does the CPU have native bf16 instructions?
///
/// x86: the AVX-512 BF16 extension (Cooper Lake and later) adds bf16 dot products.
/// aarch64: the `bf16` feature, from Armv8.6-A.
/// Detection happens at runtime — the same binary gives different answers on different machines.
///
/// `#[cfg(target_arch = ...)]` picks one of these three bodies at *compile* time, and the three
/// together cover every target, so there is always exactly one `cpu_has_bf16` in the build.
#[cfg(target_arch = "x86_64")]
fn cpu_has_bf16() -> bool {
    // Deliberately not also `amx-bf16` (Sapphire Rapids and later). AMX is a second, wider bf16
    // unit, but no shipping x86 part has it without AVX512-BF16 as well — so testing for it
    // would add a name to this line and not one machine to the answer.
    std::arch::is_x86_feature_detected!("avx512bf16")
}

#[cfg(target_arch = "aarch64")]
fn cpu_has_bf16() -> bool {
    std::arch::is_aarch64_feature_detected!("bf16")
}

// Everything else: 32-bit ARM, RISC-V, wasm. No detection macro to call, and no bf16.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn cpu_has_bf16() -> bool {
    false
}

/// Reject a graph whose declared tensors are in a float type we cannot execute.
///
/// Called right after the session opens. In practice ORT usually rejects a bf16 graph
/// first, while resolving kernels inside `commit_from_file` — but its message names a
/// node, not a cause, so [`crate::session::open`] attaches [`REEXPORT_ADVICE`] to that
/// failure too. This check covers the case where the graph loads anyway: bf16 surviving
/// only in the declared inputs and outputs, which would otherwise fail on the first `run`.
pub fn ensure_executable(session: &Session) -> Result<()> {
    if bf16_is_executable() {
        return Ok(());
    }

    // One iterator over inputs *and* outputs, built without allocating: `map` reshapes each side
    // into the same `(name, type)` pair, and `chain` runs the second after the first. Nothing is
    // read until the `for` below asks for it, so this walks the two lists once between them.
    let declared = session
        .inputs
        .iter()
        .map(|i| (&i.name, &i.input_type))
        .chain(session.outputs.iter().map(|o| (&o.name, &o.output_type)));

    for (name, ty) in declared {
        // `if let` matches one shape and ignores every other: a tensor whose element type is
        // bf16. The `_` is the shape, which does not matter here, and a non-tensor value
        // (a sequence, a map) falls through the pattern entirely.
        if let Some((_, TensorElementType::Bfloat16)) = tensor_type(ty) {
            bail!("`{name}` is bfloat16.\n\n{REEXPORT_ADVICE}");
        }
    }
    Ok(())
}
