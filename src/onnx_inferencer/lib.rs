//! ONNX Runtime inference for exported `hf2mobile` models, in Rust.
//!
//! The graph this crate runs is the postprocessed one (`python -m hf2mobile.postprocess`): it
//! returns the token it sampled, not a row of logits, and it carries the ids that end a turn.
//! So a graph and a `tokenizer.json` are the whole configuration — the sampling policy and the
//! stop tokens were decided at export time and travel with the model.
//!
//! # Two front ends, one engine
//!
//! | front end | feature | what it is |
//! |-----------|---------|------------|
//! | [`python`] | `python` | the `hf2mobile._ortrs_binding` extension module — what `hf2mobile.infer` imports |
//! | [`cli`]    | `cli`    | the `hf2mobile-infer` executable — the same run with no interpreter, so it can be cross-compiled and pushed to a phone |
//!
//! Everything below those two is plain Rust with no Python in it, which is what makes the
//! second front end possible at all: an Android build simply leaves the `python` feature off.
//!
//! # Layout
//!
//! | module               | Python? | job |
//! |----------------------|---------|-----|
//! | [`python`]           | yes     | the `#[pyclass]` wrapper — argument checking and nothing else |
//! | [`numpy`]            | yes     | numpy array <-> ORT tensor, for the raw `run` escape hatch |
//! | [`cli`]              | no      | flags, paths and the run report for the standalone binary |
//! | [`chat_template`]    | no      | render the model's Jinja chat template, as `transformers` would |
//! | [`session`]          | no      | open a graph, register the custom op, `DEBUG=1` dumps |
//! | [`precision`]        | no      | reject a dtype this machine cannot execute |
//! | [`kv_cache`]         | no      | carry key/value tensors between decode steps, without copying |
//! | [`onnx_file`]        | no      | read the stop-token ids out of the `.onnx` protobuf |
//! | [`sample_logits`]    | no      | the `SampleLogits` operator — the file `src/onnx_plugins` owns |
//! | [`causallm`]         | no      | tokenize, prefill, decode, time |
//!
//! # Errors
//!
//! Everything fallible returns `anyhow::Result`, and pyo3 turns an `anyhow` error into a
//! Python `RuntimeError` carrying the whole context chain — so a failure deep in the cache
//! surfaces in Python as "opening ONNX model `x.onnx`: ...". The exception is [`numpy`], which
//! raises `ValueError` directly: "that array has the wrong dtype" is a caller mistake, and
//! `ValueError` is what a Python caller would think to catch for it.

pub mod causallm;
pub mod kv_cache;
pub mod onnx_file;
pub mod precision;
pub mod session;

// The `SampleLogits` operator and the sampling policy inside it, compiled straight out of the
// plugin crate rather than reimplemented: `src/onnx_plugins` builds this same file into the
// `.so` a mobile runtime loads, so the token this runtime picks and the token that runtime
// picks come from one definition. The two crates do not depend on each other in any way —
// `#[path]` only tells `cargo` where a source file lives.
#[path = "../onnx_plugins/sample_logits.rs"]
pub mod sample_logits;

// `#[cfg(feature = ...)]` compiles the item only when that feature is on, so an Android build
// never sees the pyo3 modules and a wheel never carries the CLI's argument parser.
#[cfg(feature = "cli")]
pub mod chat_template;
#[cfg(feature = "cli")]
pub mod cli;

#[cfg(feature = "python")]
mod numpy;
#[cfg(feature = "python")]
mod python;
