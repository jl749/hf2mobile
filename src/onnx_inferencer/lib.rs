//! `hf2mobile._ortrs_binding` — ONNX Runtime inference for exported models, in Rust.
//!
//! ```python
//! from hf2mobile.inference import CausalLMInferencer
//!
//! lm = CausalLMInferencer("inference.onnx", "tokenizer.json")
//! print(lm.generate("Where is the capital of France?", num_generation=64))
//! print(lm.ttft_s, lm.tps)
//! ```
//!
//! The graph it runs is the postprocessed one (`python -m hf2mobile.postprocess`): it returns
//! the token it sampled, not a row of logits, and it carries the ids that end a turn. So the
//! two paths above are the whole configuration — the sampling policy and the stop tokens were
//! decided at export time and travel with the model.
//!
//! # Layout
//!
//! Only this file and [`numpy`] touch Python; the runtime underneath is plain Rust and can be
//! tested or reused without an interpreter.
//!
//! | module            | Python? | job |
//! |-------------------|---------|-----|
//! | `lib` (here)      | yes     | the `#[pyclass]` wrapper — argument checking and nothing else |
//! | [`numpy`]         | yes     | numpy array <-> ORT tensor, for the raw `run` escape hatch |
//! | [`session`]       | no      | open a graph, register the custom op, `DEBUG=1` dumps |
//! | [`precision`]     | no      | reject a dtype this machine cannot execute |
//! | [`kv_cache`]      | no      | carry key/value tensors between decode steps, without copying |
//! | [`onnx_file`]     | no      | read the stop-token ids out of the `.onnx` protobuf |
//! | [`sample_logits`] | no      | the `SampleLogits` operator — the file `src/onnx_plugins` owns |
//! | [`causallm`]     | no      | tokenize, prefill, decode, time |
//!
//! # Errors
//!
//! Everything fallible returns `anyhow::Result`, and pyo3 turns an `anyhow` error into a
//! Python `RuntimeError` carrying the whole context chain — so a failure deep in the cache
//! surfaces in Python as "opening ONNX model `x.onnx`: ...". The exception is [`numpy`], which
//! raises `ValueError` directly: "that array has the wrong dtype" is a caller mistake, and
//! `ValueError` is what a Python caller would think to catch for it.

use anyhow::Result;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::causallm::Step;

mod causallm;
mod kv_cache;
mod numpy;
mod onnx_file;
mod precision;
mod session;

// The `SampleLogits` operator and the sampling policy inside it, compiled straight out of the
// plugin crate rather than reimplemented: `src/onnx_plugins` builds this same file into the
// `.so` a mobile runtime loads, so the token this runtime picks and the token that runtime
// picks come from one definition. The two crates do not depend on each other in any way —
// `#[path]` only tells `cargo` where a source file lives.
#[path = "../onnx_plugins/sample_logits.rs"]
mod sample_logits;

/// ONNX Runtime inference over an exported causal-LM graph.
///
/// `#[pyclass]` is what makes this Rust struct visible to Python as a class. `unsendable`
/// tells pyo3 that the object must not travel between Python threads — an ORT session is tied
/// to the thread that built it — so touching it from another thread raises instead of
/// corrupting memory.
#[pyclass(name = "CausalLMInferencer", unsendable)]
pub struct CausalLMInferencer {
    inner: causallm::CausalLm,
    /// Time-to-first-token of the most recent `generate`, in seconds. Zero before prefill.
    /// `#[pyo3(get)]` publishes the field to Python as a read-only attribute.
    #[pyo3(get)]
    ttft_s: f64,
    /// Decode throughput of the most recent `generate`, in tokens/sec. Zero before prefill.
    #[pyo3(get)]
    tps: f64,
    /// Whether `DEBUG=1` asked for a profile, which decides what `Drop` has to flush.
    profiling: bool,
}

#[pymethods]
impl CausalLMInferencer {
    /// Load a postprocessed generation graph (one with KV IO) with HF `tokenizer.json`.
    ///
    /// `onnx_path` is `inference.onnx` — the graph `python -m hf2mobile.postprocess` writes.
    /// It ends in a `SampleLogits` node, so it hands back a token rather than logits, and it
    /// carries the EOS ids that stop a turn; both are read from the graph, so there is nothing
    /// here to pass them as.
    ///
    /// Raises if the graph's dtype is not supported (e.g. bfloat16) by the runtime.
    ///
    /// `DEBUG=1` saves the optimized graph (`<model>.ort`) and a chrome://tracing profile.
    #[new]
    #[pyo3(signature = (onnx_path, tokenizer_path, intra_threads = None))]
    fn new(py: Python<'_>, onnx_path: &str, tokenizer_path: &str, intra_threads: Option<usize>) -> Result<Self> {
        let profiling: bool = session::debug_artifacts(onnx_path).is_some();
        // Loading a multi-gigabyte graph takes seconds and touches no Python objects, so we
        // hand the GIL (Python's one-thread-at-a-time lock) back for the duration and let
        // other Python threads run while ORT works.
        let inner = py.allow_threads(|| causallm::CausalLm::open(onnx_path, tokenizer_path, intra_threads))?;
        Ok(Self {
            inner,
            ttft_s: 0.0,
            tps: 0.0,
            profiling,
        })
    }

    /// Continue the current turn, or start one from `prompt`, and return `(text, (ttft_s, tps))`.
    ///
    /// `text` is `None` until an end-of-sequence token arrives:
    ///
    /// ```python
    /// while True:
    ///     text, (ttft, tps) = lm.generate(prompt, num_generation=1)
    ///     if text is not None:
    ///         break
    /// ```
    ///
    /// The KV cache, the token history and the timings all live across those calls, so
    /// **`prompt` is read only when a turn starts** — it is prefilled once, and `ttft_s` keeps
    /// reporting that first pass. Once a turn ends its cache is released and further calls
    /// warn and do nothing; call `reset()` to start over.
    ///
    /// - `num_generation` — how many new tokens to produce *in this call*.
    /// - `stream_output` — print the text to stdout as it is produced.
    ///
    /// There is no `temperature`/`top_k`/`top_p` here: the graph's `SampleLogits` node holds
    /// the policy, baked in by `python -m hf2mobile.postprocess --temp/--top_k/--top_p`.
    #[pyo3(signature = (prompt, num_generation = 64, stream_output = false))]
    fn generate(
        &mut self,
        py: Python<'_>,
        prompt: &str,
        num_generation: i32,
        stream_output: bool,
    ) -> Result<(Option<String>, (f64, f64))> {
        // Python's `int` is signed and unbounded; Rust wants a count. Clamping at zero turns
        // a negative budget into "generate nothing" rather than a huge unsigned number.
        let budget: usize = num_generation.max(0) as usize;

        // Decoding is compute plus, when streaming, writes to stdout — no Python objects are
        // touched, so the GIL can go here too.
        let step: Step = py.allow_threads(|| self.inner.generate(prompt, budget, stream_output))?;

        // Mirrored onto the object as well as returned, so a caller can ignore the tuple and
        // read `lm.ttft_s` after the fact.
        self.ttft_s = step.ttft_s;
        self.tps = step.tps;
        Ok((step.text, (step.ttft_s, step.tps)))
    }

    /// Abandon the current turn and release its KV cache.
    ///
    /// Required after a turn ends before `generate` will do anything again, and usable at
    /// any time to drop a turn part-way through.
    fn reset(&mut self) -> Result<()> {
        self.inner.reset()?;
        self.ttft_s = 0.0;
        self.tps = 0.0;
        Ok(())
    }

    /// Is a turn open — started, and not yet ended by an end-of-sequence token?
    ///
    /// `False` both before the first `generate` and after a turn has finished, so it
    /// answers "will the next `generate` continue what I was doing?" rather than "has
    /// anything happened?".
    #[getter]
    fn is_generating(&self) -> bool {
        self.inner.is_generating()
    }

    /// The ids emitted so far this turn, before detokenization. `generate` returns text
    /// because that is what callers want; this is here for the cases where the exact
    /// tokens matter — comparing two runs, or checking an export against a reference.
    #[getter]
    fn token_ids(&self) -> Vec<i64> {
        self.inner.token_ids()
    }

    /// How many tokens the current turn's prompt came to, i.e. the length of the prefill
    /// pass. Worth reading next to `ttft_s`, which scales with it.
    #[getter]
    fn prefill_len(&self) -> usize {
        self.inner.prefill_len()
    }

    /// End-of-sequence ids `generate` stops on, as read from the graph's
    /// `hf2mobile_EOS_tokens` node.
    #[getter]
    fn eos_tokens(&self) -> Vec<i64> {
        self.inner.eos_tokens().to_vec()
    }

    /// How many KV cache tensors the graph declared — two per layer, keys and values.
    ///
    /// Reported from the cache the runtime actually built, so it is the count that will be
    /// fed back per token rather than a guess made by matching input names in Python.
    #[getter]
    fn num_kv_slots(&self) -> usize {
        self.inner.kv_slots()
    }

    /// The graph's input names, in declaration order.
    #[getter]
    fn input_names(&self) -> Vec<String> {
        self.inner.session().inputs.iter().map(|i| i.name.clone()).collect()
    }

    /// The graph's output names, in declaration order.
    #[getter]
    fn output_names(&self) -> Vec<String> {
        self.inner.session().outputs.iter().map(|o| o.name.clone()).collect()
    }

    /// Per input, `(name, shape, dtype)`. Symbolic dimensions read as `-1`, and `dtype`
    /// is spelled the numpy way (`"float32"`, `"int64"`).
    #[getter]
    fn input_info(&self) -> Vec<(String, Vec<i64>, String)> {
        self.inner
            .session()
            .inputs
            .iter()
            .map(|input| {
                let (shape, dtype) = match session::tensor_type(&input.input_type) {
                    Some((shape, ty)) => (shape.to_vec(), session::numpy_dtype_name(ty)),
                    None => (Vec::new(), session::UNKNOWN_DTYPE),
                };
                (input.name.clone(), shape, dtype.to_string())
            })
            .collect()
    }

    /// Run one forward pass, numpy in and numpy out.
    ///
    /// The escape hatch, for inspecting a graph a pass at a time. It does *not* share
    /// `generate`'s KV cache — every call stands alone, and every array is copied across
    /// the boundary. `generate` is the path that keeps tensors on the ORT side.
    ///
    /// Clippy reports a `useless_conversion` against this signature. It is `#[pymethods]`'
    /// own expansion, not anything written here: this is the one method that already returns
    /// `PyResult`, so the error conversion the macro wraps it in goes `PyErr` to `PyErr`.
    /// Returning anyhow's `Result` instead would silence it, but then the `?`s below would
    /// surface numpy's `ValueError` as a `RuntimeError`, so the warning is the better trade.
    /// An `#[allow]` here or on the `impl` does not reach the generated code.
    fn run<'py>(&mut self, py: Python<'py>, inputs: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyDict>> {
        let mut ort_inputs: Vec<(String, ort::value::DynValue)> = Vec::with_capacity(inputs.len());
        for (key, value) in inputs.iter() {
            let name: String = key.extract()?;
            ort_inputs.push((name, numpy::to_ort(py, &value)?));
        }

        let outputs = self.inner.session_mut().run(ort_inputs).map_err(anyhow::Error::from)?;

        let result = PyDict::new_bound(py);
        for (name, value) in outputs.iter() {
            result.set_item(name, numpy::from_ort(py, &value)?)?;
        }
        Ok(result)
    }
}

/// `Drop` is Rust's destructor: it runs when the object is freed, which for a `#[pyclass]` is
/// when Python garbage-collects it.
impl Drop for CausalLMInferencer {
    fn drop(&mut self) {
        // ORT buffers profiling events in memory and writes the trace only when profiling
        // is explicitly ended, so without this the `DEBUG=1` file is never created.
        // Dropping is the one moment we know no more runs are coming.
        if self.profiling {
            match self.inner.session_mut().end_profiling() {
                Ok(path) => eprintln!("[hf2mobile] DEBUG=1: wrote chrome trace `{path}`"),
                Err(err) => eprintln!("[hf2mobile] DEBUG=1: could not write the chrome trace: {err}"),
            }
        }
    }
}

/// The module initializer Python calls on `import`. `#[pymodule]` generates the C entry point;
/// the name of this function is the name of the module, which is why it is `_ortrs_binding`.
#[pymodule]
fn _ortrs_binding(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CausalLMInferencer>()?;
    Ok(())
}
