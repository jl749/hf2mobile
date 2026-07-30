//! The KV cache — the only state that survives from one decode step to the next.
//!
//! # What it is
//!
//! A decoder-only LM attends over every token it has seen so far. Recomputing all of
//! them each step would make generation quadratic, so the graph instead returns the
//! per-layer key/value tensors it computed, and we feed them back in on the next step.
//! Step *n* then only computes K/V for the one new token.
//!
//! # Why it needs its own file
//!
//! That cache is big — tens of megabytes for a 0.6B model at a thousand tokens — and it
//! is handed in and out **once per generated token**. Copy it and the copy, not the
//! matrix multiplies, sets your tokens/sec.
//!
//! So we never copy it. ORT hands out a tensor as a reference-counted handle to a
//! buffer *it* owns (`DynValue` is an `Arc` around ONNX Runtime's `OrtValue`), which
//! means:
//!
//! - feeding the cache **in** is [`ort::value::Value::view`] — clone a handle;
//! - taking the update **out** is [`ort::session::SessionOutputs::remove`] — clone a handle.
//!
//! Neither touches the tensor data. Last step's buffers are freed when the last handle
//! to them drops, at the end of [`KvCache::take_update`]. Cost per token: a handful of
//! pointer writes, and zero bytes of `memcpy`.
//!
//! # The graph contract
//!
//! We do not hardcode layer names. A cache slot is *any* input `x` for which the graph
//! also declares an output `x_out` — which is exactly how the exporter in
//! `hf2mobile.exporter` wires `past_keys_0` / `past_keys_0_out` and friends. Discovering
//! them means this file keeps working if the naming or the layer count changes.

use std::collections::HashSet;

use anyhow::{bail, Context, Result};
use ndarray::{ArrayD, IxDyn};
use ort::session::{Session, SessionInputValue, SessionOutputs};
use ort::tensor::TensorElementType;
use ort::value::{DynValue, Tensor};

use crate::session::tensor_type;

/// Suffix the exporter appends to a cache input to name the matching output.
const OUT_SUFFIX: &str = "_out";

/// Cached key/value tensors for every layer, in the graph's own input order.
pub struct KvCache {
    slots: Vec<Slot>,
}

/// One cached tensor: the graph reads it as `input` and returns the grown version as
/// `output`. `value` is the handle we currently hold.
struct Slot {
    input: String,
    output: String,
    /// Element type and shape of this slot when the cache is empty, kept so
    /// [`KvCache::reset`] can rebuild it without re-reading the graph.
    dtype: TensorElementType,
    empty_shape: Vec<usize>,
    value: DynValue,
}

impl KvCache {
    /// Split a graph's inputs into the KV cache and everything else.
    ///
    /// Returns the cache (empty, ready to decode into) alongside the names of the inputs
    /// it does *not* supply — `input_ids` and friends, which are the caller's to fill.
    /// One pass, one classification rule: an input is a cache slot exactly when the graph
    /// also declares `<name>_out`.
    pub fn discover(session: &Session) -> Result<(Self, Vec<&str>)> {
        // Index the output names once. `session.outputs` is a Vec, so scanning it per
        // input would make discovery quadratic in the layer count — a set makes each
        // lookup O(1) instead.
        let outputs: HashSet<&str> = session.outputs.iter().map(|o| o.name.as_str()).collect();

        let mut slots = Vec::new();
        let mut others = Vec::new();
        for input in &session.inputs {
            let output = format!("{}{OUT_SUFFIX}", input.name);
            if !outputs.contains(output.as_str()) {
                others.push(input.name.as_str());
                continue;
            }

            let (shape, dtype) = tensor_type(&input.input_type)
                .with_context(|| format!("cache input `{}` is not a tensor", input.name))?;
            let empty_shape = empty_shape(&input.name, shape)?;

            slots.push(Slot {
                value: empty_tensor(dtype, &empty_shape)?,
                input: input.name.clone(),
                output,
                dtype,
                empty_shape,
            });
        }

        if slots.is_empty() {
            bail!(
                "graph declares no KV cache: expected some input `x` with a matching output `x{OUT_SUFFIX}`. \
                 Is this the generation graph (case2.onnx) rather than the prefill-only one?"
            );
        }

        Ok((Self { slots }, others))
    }

    /// Number of cache tensors — two per layer (keys and values).
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Forget everything. Call this before each generation so prompts don't bleed into
    /// each other.
    pub fn reset(&mut self) -> Result<()> {
        for slot in &mut self.slots {
            slot.value = empty_tensor(slot.dtype, &slot.empty_shape)
                .with_context(|| format!("building empty cache for `{}`", slot.input))?;
        }
        Ok(())
    }

    /// Append the cache to a list of session inputs, as borrowed views.
    ///
    /// The `'a` on both sides is the promise that makes this free: the views borrow
    /// `self`, so the compiler guarantees the cache outlives the run that reads it, and
    /// no data has to be copied to make that true.
    pub fn bind<'a>(&'a self, inputs: &mut Vec<(&'a str, SessionInputValue<'a>)>) {
        for slot in &self.slots {
            inputs.push((slot.input.as_str(), SessionInputValue::from(&slot.value)));
        }
    }

    /// Adopt the grown cache the graph just produced, dropping the previous one.
    ///
    /// `remove` hands over ORT's own output buffer rather than a copy of it, and takes
    /// it *out* of `outputs` so nothing else can claim it twice.
    pub fn take_update(&mut self, outputs: &mut SessionOutputs<'_>) -> Result<()> {
        for slot in &mut self.slots {
            slot.value = outputs
                .remove(&slot.output)
                .with_context(|| format!("graph did not return `{}`", slot.output))?;
        }
        Ok(())
    }
}

/// Shape of a cache slot holding zero tokens.
///
/// The graph declares `[batch, n_kv_heads, cached_len, head_dim]` and leaves the
/// dimensions it cannot know as `-1`. We pin `batch = 1` (one sequence at a time) and
/// `cached_len = 0` (nothing cached yet); `n_kv_heads` and `head_dim` are fixed by the
/// architecture, so the graph must state them.
fn empty_shape(name: &str, declared: &[i64]) -> Result<Vec<usize>> {
    let [_batch, n_kv_heads, _cached_len, head_dim] = *declared else {
        bail!(
            "expected cache input `{name}` to be rank 4 [batch, heads, cached_len, head_dim], got shape {declared:?}"
        );
    };
    if n_kv_heads < 0 || head_dim < 0 {
        bail!("cache input `{name}` leaves heads/head_dim symbolic (shape {declared:?}); it must declare both");
    }
    Ok(vec![1, n_kv_heads as usize, 0, head_dim as usize])
}

/// Build a zero-token tensor of the given element type.
///
/// The KV cache dtype follows the model — fp32 after an upcast, f16 or bf16 when the
/// provider runs those natively — so the empty tensor has to match, or ORT rejects it.
/// We go through `ndarray` because a shape with a `0` in it is not a valid shape for
/// ORT's `(shape, data)` constructor, but is perfectly fine for an array.
fn empty_tensor(dtype: TensorElementType, shape: &[usize]) -> Result<DynValue> {
    // Every arm is the same line at a different type, which is exactly what a macro is
    // for: `$t` is substituted in, and each expansion is type-checked on its own. The
    // empty `Vec` is not a shortcut — a shape with a 0 in it holds no elements, so
    // there is genuinely no data to supply.
    macro_rules! empty {
        ($t:ty) => {
            Ok(Tensor::<$t>::from_array(ArrayD::<$t>::from_shape_vec(IxDyn(shape), Vec::new())?)?.into_dyn())
        };
    }

    match dtype {
        TensorElementType::Float32 => empty!(f32),
        TensorElementType::Float16 => empty!(half::f16),
        TensorElementType::Bfloat16 => empty!(half::bf16),
        TensorElementType::Float64 => empty!(f64),
        other => bail!("unsupported KV cache element type {other:?}; expected a float type"),
    }
}
