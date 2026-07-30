//! Opening an ONNX Runtime session, and reading what a graph declares.
//!
//! Deliberately a couple of free functions over `ort::Session` rather than a wrapper
//! struct. A wrapper would only forward calls, and forwarding code is code that drifts
//! out of sync with the library it forwards to.

use std::path::Path;

use anyhow::{Context, Result};
use ort::execution_providers::CPUExecutionProvider;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::tensor::TensorElementType;
use ort::value::ValueType;

use crate::precision::REEXPORT_ADVICE;

/// Load `path` and get it ready to run.
///
/// `intra_threads` caps the threads ORT uses *inside* a single operator (the big
/// MatMuls). `None` lets ORT decide, which is one thread per physical core — usually
/// what you want, unless something else on the machine also needs the cores.
///
/// Set `DEBUG=1` in the environment to also write the optimized graph and a profiling
/// trace to the current directory — see [`debug_artifacts`].
pub fn open(path: &str, intra_threads: Option<usize>) -> Result<Session> {
    let mut builder = Session::builder()?
        // Level3 turns on every graph rewrite ORT has, including layout changes that
        // are specific to the current CPU. Costs a moment at load, pays it back on the
        // first token.
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        // The CPU provider is the only one always compiled into onnxruntime. ORT walks
        // this list in order and hands each node to the first provider that claims it,
        // so adding an accelerator later means inserting it *before* CPU here.
        .with_execution_providers([CPUExecutionProvider::default().build()])?;

    if let Some(n) = intra_threads {
        builder = builder.with_intra_threads(n)?;
    }

    if let Some(artifacts) = debug_artifacts(path) {
        // `.ort` is ONNX Runtime's own serialized format: the graph *after* Level3
        // optimization, ready to mmap. Comparing it against the input `.onnx` is how you
        // see which fusions actually fired.
        builder = builder
            .with_optimized_model_path(&artifacts.optimized_model)?
            .with_config_entry("session.save_model_format", "ORT")?
            // ORT appends a timestamp and `.json` to this prefix, and writes nothing
            // until `Session::end_profiling` is called — see `CausalLMInferencer`'s `Drop`.
            .with_profiling(&artifacts.profile_prefix)?;
        eprintln!(
            "[hf2mobile] DEBUG=1: writing `{}` and a chrome trace `{}*.json`",
            artifacts.optimized_model, artifacts.profile_prefix
        );
    }

    // ORT reports a missing kernel as "Could not find an implementation for <node>", which
    // names the symptom but not the cause. On the CPU provider the cause is nearly always
    // a dtype it has no kernels for, so say that here rather than leaving the reader to
    // work it out.
    builder
        .commit_from_file(path)
        .with_context(|| format!("ONNXRuntime could not load `{path}`.\n\n{REEXPORT_ADVICE}"))
}

/// Where the `DEBUG=1` dumps go, or `None` when debugging is off.
///
/// Both land in the current directory (not next to the model, which may be read-only or
/// on a shared path) and are named after the model file, so two models profiled in one
/// session do not overwrite each other.
pub struct DebugArtifacts {
    /// The Level3-optimized graph, in ORT's own format.
    pub optimized_model: String,
    /// Prefix ORT appends a timestamp and `.json` to, giving a chrome://tracing file.
    pub profile_prefix: String,
}

pub fn debug_artifacts(model_path: &str) -> Option<DebugArtifacts> {
    if std::env::var("DEBUG").as_deref() != Ok("1") {
        return None;
    }
    let stem = Path::new(model_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
    Some(DebugArtifacts {
        optimized_model: format!("{stem}.ort"),
        profile_prefix: format!("{stem}_profile_"),
    })
}

/// The declared shape and element type of one graph input/output.
///
/// Dimensions the graph leaves symbolic — batch, sequence length, cached length — come
/// back as `-1`, because their real value is only known once you feed the graph.
/// Returns `None` for the (unused here) non-tensor value kinds: sequences and maps.
pub fn tensor_type(ty: &ValueType) -> Option<(&[i64], TensorElementType)> {
    match ty {
        ValueType::Tensor { ty, shape, .. } => Some((shape, *ty)),
        _ => None,
    }
}

/// What we report for anything numpy has no name for, including non-tensor inputs.
pub const UNKNOWN_DTYPE: &str = "unsupported";

/// Spell an ORT element type the way numpy spells it (`"float32"`, not ORT's `"f32"`).
///
/// This crosses into Python via `CausalLMInferencer.input_info`, where callers index numpy
/// with the string, so the numpy spelling is the one that has to be right.
pub fn numpy_dtype_name(ty: TensorElementType) -> &'static str {
    match ty {
        TensorElementType::Float32 => "float32",
        TensorElementType::Float64 => "float64",
        TensorElementType::Float16 => "float16",
        TensorElementType::Bfloat16 => "bfloat16",
        TensorElementType::Int64 => "int64",
        TensorElementType::Int32 => "int32",
        TensorElementType::Int16 => "int16",
        TensorElementType::Int8 => "int8",
        TensorElementType::Uint64 => "uint64",
        TensorElementType::Uint32 => "uint32",
        TensorElementType::Uint16 => "uint16",
        TensorElementType::Uint8 => "uint8",
        TensorElementType::Bool => "bool",
        TensorElementType::String => "str",
        // Types numpy has no name for (4-bit ints, complex, undefined). They never
        // appear in the exports we run, and a wrong guess would be worse than a label.
        other => {
            debug_assert!(false, "no numpy name for {other:?}");
            UNKNOWN_DTYPE
        }
    }
}
