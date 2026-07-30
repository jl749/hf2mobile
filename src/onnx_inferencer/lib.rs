//! `hf2mobile._ortrs_binding` — ONNX Runtime inference for exported models, in Rust.
//!
//! ```python
//! from hf2mobile.inference import CausalLMInferencer
//!
//! lm = CausalLMInferencer("case2.onnx", "tokenizer.json")
//! print(lm.generate("Where is the capital of France?", num_generation=64))
//! print(lm.ttft_s, lm.tps)
//! ```
//!
//! # Layout
//!
//! Only this file and [`numpy`] touch Python
//! the runtime underneath is plain Rust and can be tested or reused without an interpreter.
//!
//! | module          | Python? | job |
//! |-----------------|---------|-----|
//! | `lib` (here)    | yes     | the `#[pyclass]` wrapper — argument checking and nothing else |
//! | [`numpy`]       | yes     | numpy array <-> ORT tensor, for the raw `run` escape hatch |
//! | [`session`]     | no      | open a graph, read what it declares, `DEBUG=1` dumps |
//! | [`precision`]   | no      | reject a dtype this machine cannot execute |
//! | [`kv_cache`]    | no      | carry key/value tensors between decode steps, without copying |
//! | [`sampling`]    | no      | turn a row of logits into a token |
//! | [`causal_lm`]   | no      | tokenize, prefill, decode, time |
//!
//! Errors are `anyhow`, which pyo3 turns into a Python `RuntimeError` carrying the whole context chain
//!  — so a failure deep in the cache surfaces in Python as "opening ONNX model `x.onnx`: ...".
//! The exception is [`numpy`], which raises `ValueError` directly: "that array has the wrong dtype"
//! is a caller mistake, and `ValueError` is the exception a Python caller would actually think to catch for it.

use anyhow::Result;
use pyo3::prelude::*;
use pyo3::types::PyDict;

mod causal_lm;
mod kv_cache;
mod numpy;
mod precision;
mod sampling;
mod session;

/// ONNX Runtime inference over an exported causal-LM graph
///
/// `unsendable`: an ORT session is tied to the thread that built it.
///   this marker enforces pyo3 touching the object from another Python thread raises Err.
///   (prevent data corruption)
#[pyclass(name = "CausalLMInferencer", unsendable)]
pub struct CausalLMInferencer {
    inner: causal_lm::CausalLm,
    /// Time-to-first-token of the most recent `generate`, in seconds. Zero before prefill.
    #[pyo3(get)]
    ttft_s: f64,
    /// Decode throughput of the most recent `generate`, in tokens/sec. Zero before prefill.
    #[pyo3(get)]
    tps: f64,
    /// Triggers the profiler (when `DEBUG=1`)
    profiling: bool,
}

#[pymethods]
impl CausalLMInferencer {
    /// Load a generation graph (one with KV IO) with HF `tokenizer.json`.
    ///
    /// Raises if the graph's dtype is not supported (e.g. bfloat16).
    /// The runtime never rewrites a model(bf16->f32); that is the exporter's job.
    ///
    /// `DEBUG=1` saves the optimized graph (`<model>.ort`) and a chrome://tracing profile.
    #[new]
    #[pyo3(signature = (onnx_path, tokenizer_path, intra_threads = None))]
    fn new(py: Python<'_>, onnx_path: &str, tokenizer_path: &str, intra_threads: Option<usize>) -> Result<Self> {
        let profiling: bool = session::debug_artifacts(onnx_path).is_some();
        // Loading a multi-gigabyte graph takes seconds and touches no Python objects.
        // (release the GIL so that other threads run while ORT works)
        let inner: causal_lm::CausalLm =
            py.allow_threads(|| causal_lm::CausalLm::open(onnx_path, tokenizer_path, intra_threads))?;
        Ok(Self {
            inner,
            ttft_s: 0.0,
            tps: 0.0,
            profiling,
        })
    }

    /// Continue the current turn, or start one from `prompt`, and return
    /// `(text, (ttft_s, tps))`.
    ///
    /// `text` is `None` until an end-of-sequence token arrives, and the finished output
    /// exactly once on the call where it does. So this composes into a poll loop:
    ///
    /// ```python
    /// while True:
    ///     text, (ttft, tps) = lm.generate(prompt, num_generation=1)
    ///     if text is not None:
    ///         break
    /// ```
    ///
    /// The KV cache, the token history and the timings all live across those calls, so the
    /// prompt is prefilled once and `ttft_s` keeps reporting that first pass. **`prompt` is
    /// read only when a turn starts**; later calls continue from the cache and ignore it.
    ///
    /// Once a turn ends, its cache is released and further calls warn and do nothing —
    /// call `reset()` to start another.
    ///
    /// - `num_generation` — how many tokens to produce *in this call*.
    /// - `eos_tokens` — stop on any of these ids. Omitted, the terminators found in the
    ///   tokenizer's vocabulary are used; pass a list to override that. Pass `[]` and the
    ///   turn never ends on its own.
    /// - `stream_output` — print the text to stdout as it is produced. What is printed can
    ///   lag the return value: a token that is only half a UTF-8 character is held back
    ///   until it completes, rather than printed broken, so stdout always holds a prefix of
    ///   the final text and never mojibake.
    /// - `temperature` — `0` (the default) means greedy: always the highest-scoring
    ///   token, and so reproducible. Above zero enables sampling, where `top_k` and
    ///   `top_p` narrow the field first. See [`sampling`] for the order they apply in.
    ///
    /// `ttft_s` and `tps` are updated from this call; `token_ids` and `prefill_len` read
    /// the turn's own state, so they reflect it as of the last call either way.
    #[pyo3(signature = (
        prompt,
        num_generation = 64,
        eos_tokens = None,
        stream_output = false,
        temperature = 0.0,
        top_k = 0,
        top_p = 1.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn generate(
        &mut self,
        py: Python<'_>,
        prompt: &str,
        num_generation: i32,
        eos_tokens: Option<Vec<i32>>,
        stream_output: bool,
        temperature: f32,
        top_k: i32,
        top_p: f32,
    ) -> Result<(Option<String>, (f64, f64))> {
        // The graph speaks int64 token ids; i32 is the friendlier width at the Python
        // boundary (no vocabulary comes close to 2^31) so widen once, here.
        let eos: Vec<i64> = match eos_tokens {
            Some(ids) => ids.into_iter().map(i64::from).collect(),
            None => self.inner.default_eos_tokens(),
        };
        let budget = num_generation.max(0) as usize;
        let cfg = sampling::Sampling {
            temperature,
            top_k: top_k.max(0) as usize,
            top_p,
        };

        // Decoding is compute plus, when streaming, writes to stdout — no Python objects
        // are touched, so the GIL can go.
        let step = py.allow_threads(|| self.inner.generate(prompt, budget, &eos, stream_output, cfg))?;

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
    ///
    /// Read straight from the turn rather than mirrored into a field on every `generate`,
    /// so a caller polling with `num_generation = 1` does not copy the whole history once
    /// per token.
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

    /// Token ids the tokenizer identifies as end-of-turn, i.e. what `generate` stops on
    /// when `eos_tokens` is omitted. Empty means the tokenizer uses a terminator this
    /// runtime does not recognise, and `eos_tokens` should be passed explicitly.
    #[getter]
    fn eos_tokens(&self) -> Vec<i64> {
        self.inner.default_eos_tokens()
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

#[pymodule]
fn _ortrs_binding(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CausalLMInferencer>()?;
    Ok(())
}
