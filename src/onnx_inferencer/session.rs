//! Opening an ONNX Runtime session, and reading what a graph declares.
//!
//! Deliberately a few free functions over `ort::Session` rather than a wrapper struct. A
//! wrapper would only forward calls, and forwarding code is code that drifts out of sync with
//! the library it forwards to.

use std::path::Path;

use anyhow::{bail, Context, Result};
use ort::execution_providers::CPUExecutionProvider;
use ort::operator::OperatorDomain;
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ort::session::Session;
use ort::tensor::TensorElementType;
use ort::value::ValueType;

use crate::precision::REEXPORT_ADVICE;
use crate::sample_logits::{SampleLogits, DOMAIN};

/// Load ONNX from `path` and get it ready to run.
///
/// `intra_threads` caps the threads ORT uses *inside* a single operator (the big MatMuls).
/// `None` lets ORT decide, which is one thread per physical core (usually what you want, unless something else on the machine also needs the cores).
///
/// Set `DEBUG=1` in the environment to also write the optimized graph and a profiling trace to the current directory — see [`debug_artifacts`].
pub fn open(path: &str, intra_threads: Option<usize>) -> Result<Session> {
    // Checked here rather than left to ORT. A path that is not there is the most common way
    // this function fails, and ORT reports it with the same "could not load the model" error it
    // uses for a graph whose kernels it cannot resolve — so without this, a typo in a path comes
    // back wearing the dtype advice below and sends the reader off re-exporting a healthy model.
    if !Path::new(path).is_file() {
        bail!("`{path}` is not a file");
    }

    // A builder: each call configures one thing and hands the Result<SessionBuilder> back, so they chain.
    // Each returns a `Result`, hence the `?` after every step.
    let mut builder: SessionBuilder = Session::builder()?
        // Prevent extra `libhf2mobile_plugins.so` load by explicitly fetching the plugin impl
        .with_operators(OperatorDomain::new(DOMAIN)?.add(SampleLogits::<f32>::new())?)?
        // Max OPT level, including layout changes (better mem locality: NCHW -> NCHWc) that are specific to the current CPU.
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        // TODO: when adding new EP add it before CPU (e.g. prioritize QNNEP)
        .with_execution_providers([CPUExecutionProvider::default().build()])?;

    if let Some(n) = intra_threads {
        builder = builder.with_intra_threads(n)?;
    }

    if let Some(artifacts) = debug_artifacts(path) {
        // `.ort` is ONNX Runtime's own serialized format: the graph *after* Level3 optimization, ready to mmap.
        // Comparing it against the input `.onnx` is how you see which fusions actually fired.
        builder = builder
            .with_optimized_model_path(&artifacts.optimized_model)?
            // soptimized model saved by the session => .ort
            .with_config_entry("session.save_model_format", "ORT")?
            // ORT appends a timestamp and `.json` to this prefix.
            // Writes nothing until `Session::end_profiling` is called — see `CausalLMInferencer`'s `Drop`.
            .with_profiling(&artifacts.profile_prefix)?;
        eprintln!(
            "[hf2mobile] DEBUG=1: writing `{}` and a chrome trace `{}*.json`",
            artifacts.optimized_model, artifacts.profile_prefix
        );
    }

    builder
        .commit_from_file(path) // NOTE: Builds the session from ONNX
        // `with_context` adds a line *above* whatever error came out, rather than replacing it,
        // so the reader gets ORT's own message and this one. Phrased as a hypothesis: by this
        // point the file exists, and an unresolvable kernel is usually a dtype the provider does
        // not have — but ORT's message names a node, not a cause, so we cannot know from here.
        .with_context(|| {
            format!(
                "ONNXRuntime could not load `{path}`.\n\n\
                 If the failure above names a node it has no implementation for, the usual cause \
                 is the graph's dtype:\n\n{REEXPORT_ADVICE}"
            )
        })
}

/// Where the `DEBUG=1` dumps go.
///
/// Both land in the current directory (not next to the model, which may be read-only or on a
/// shared path) and are named after the model file, so two models profiled in one session do
/// not overwrite each other.
pub struct DebugArtifacts {
    /// The Level3-optimized graph, in ORT's own format.
    pub optimized_model: String,
    /// Prefix ORT appends a timestamp and `.json` to, giving a chrome://tracing file.
    pub profile_prefix: String,
}

/// The `DEBUG=1` artifact paths for `model_path`, or `None` when debugging is off.
pub fn debug_artifacts(model_path: &str) -> Option<DebugArtifacts> {
    // `var` gives a `Result<String, _>`; `as_deref` turns that into a `Result<&str, _>` so the
    // arms below can match against a plain string literal instead of allocating one to compare.
    match std::env::var("DEBUG").as_deref() {
        Ok("1") => {
            // `file_stem` is the name without its extension (`model.onnx` -> `model`). Both
            // steps can fail — no filename, or one that is not UTF-8 — so `and_then` chains
            // them and `unwrap_or` supplies a name for the case where either does.
            let stem = Path::new(model_path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("model");
            Some(DebugArtifacts {
                optimized_model: format!("{stem}.ort"),
                profile_prefix: format!("{stem}_profile_"),
            })
        }
        Ok(_) => None,
        Err(_) => None,
    }
}

/// The declared shape and element type of one graph input/output.
///
/// Dimensions the graph leaves symbolic — batch, sequence length, cached length — come back as
/// `-1`, because their real value is only known once you feed the graph. Returns `None` for the
/// (unused here) non-tensor value kinds: sequences and maps.
pub fn tensor_type(ty: &ValueType) -> Option<(&[i64], TensorElementType)> {
    match ty {
        // Destructuring an enum variant: this both tests that `ty` is the `Tensor` case and
        // binds the two fields we want out of it. `..` says "and whatever else it holds,
        // ignore it", which is what keeps this compiling when `ort` adds a field.
        // `*ty` copies the element type out of the borrow — it is a small `Copy` enum — while
        // `shape` stays a borrow, which is why the returned slice carries the input's lifetime.
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
        // Types numpy has no name for (4-bit ints, complex, undefined). They never appear in
        // the exports we run, and a wrong guess would be worse than a label. `debug_assert`
        // fires in a debug build only, so a test would catch it while a release build carries
        // on with the label.
        other => {
            debug_assert!(false, "no numpy name for {other:?}");
            UNKNOWN_DTYPE
        }
    }
}
