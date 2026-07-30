//! `com.hf2mobile:SampleLogits` — the sampling step, as a graph node.
//!
//! ```text
//! logits [1, L, vocab]  --SampleLogits(top_k, top_p, temperature)-->  sampled_token [1, 1] int32
//! ```
//!
//! The policy is baked into the node's attributes at export time
//! (`python -m hf2mobile.postprocess`), not passed per call, because the point of the
//! operator is that the caller never sees the logits: a 262k-wide fp32 row is 1 MB per
//! token, and copying it out of ONNX Runtime only to reduce it to one integer is the most
//! expensive thing a decode step does that isn't arithmetic.
//!
//! What the numbers mean, and the order they apply in, is [`crate::sampling`]'s business —
//! shared with the Rust runtime so both agree. This file is the ONNX Runtime boundary:
//! read three attributes, find the last logits row, write one int32.

use core::marker::PhantomData;

use ort::operator::attribute::FromKernelAttributes;
use ort::operator::io::{OperatorInput, OperatorOutput};
use ort::operator::kernel::{Kernel, KernelAttributes, KernelContext};
use ort::operator::{Operator, ShapeInferenceContext};
use ort::tensor::{PrimitiveTensorElementType, Shape, SymbolicDimensions, TensorElementType};
use ort::value::ValueType;
use ort::Error;

use crate::sampling::{sample, Sampling};

/// The op type the exporter writes into the graph (`SAMPLE_LOGITS_OP` in `constant.py`).
pub const OP_NAME: &str = "SampleLogits";

/// The ONNX domain the exporter writes its custom nodes under (`ONNX_DOMAIN_NAME` in
/// `constant.py`). A node's domain has to match the domain the operator was registered in,
/// or ORT reports the op as simply not found — so it lives next to the operator, and both
/// crates that register it read it from here.
pub const DOMAIN: &str = "com.hf2mobile";

/// A logits element type this operator has a kernel for.
///
/// fp32 only, for now. ORT matches a kernel to a node by input type, so a graph exported in
/// another float type needs its own registration: an `impl LogitElement for half::f16` here
/// and an `.add(SampleLogits::<half::f16>::new())` in [`crate::register`] — the bounds are
/// already the ones [`crate::sampling::sample`] needs, and `Into<f32>` is how a narrower row
/// is widened as the candidate list is built, without copying it first.
pub trait LogitElement: PrimitiveTensorElementType + Copy + PartialOrd + Into<f32> + Send + 'static {}
impl LogitElement for f32 {}

/// The operator descriptor: one instance per logits dtype, all under the one op name.
pub struct SampleLogits<T: LogitElement>(PhantomData<T>);

impl<T: LogitElement> SampleLogits<T> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T: LogitElement> Operator for SampleLogits<T> {
    fn name(&self) -> &str {
        OP_NAME
    }

    fn inputs(&self) -> Vec<OperatorInput> {
        vec![OperatorInput::required(T::into_tensor_element_type())]
    }

    fn outputs(&self) -> Vec<OperatorOutput> {
        vec![OperatorOutput::required(TensorElementType::Int32)]
    }

    /// Read the policy off the node. Once per session, not once per token.
    ///
    /// Every attribute is required. A missing or wrongly-typed one could be defaulted
    /// instead, but the default that reads naturally — greedy — silently ignores the two
    /// filters, and "my top-p had no effect" is a far worse afternoon than a failure at
    /// session build naming the attribute.
    fn create_kernel(&self, attributes: &KernelAttributes) -> ort::Result<Box<dyn Kernel>> {
        let top_k: i64 = required_attr(attributes, "top_k", "int")?;
        Ok(Box::new(SampleKernel::<T> {
            cfg: Sampling {
                temperature: required_attr(attributes, "temperature", "float")?,
                top_k: top_k.max(0) as usize,
                top_p: required_attr(attributes, "top_p", "float")?,
            },
            _element: PhantomData,
        }))
    }

    /// Tell ONNX Runtime's shape inference what comes out, so the `(1, 1)` int32 the
    /// exporter declares for `sampled_token` is checked rather than taken on trust.
    fn infer_shape(&self, ctx: &mut ShapeInferenceContext) -> ort::Result<()> {
        ctx.set_output(
            0,
            &ValueType::Tensor {
                ty: TensorElementType::Int32,
                shape: Shape::from([1_i64, 1]),
                dimension_symbols: SymbolicDimensions::empty(2),
            },
        )
    }
}

/// Fetch a required node attribute, or say which one is missing and what type it wanted.
fn required_attr<'a, T: FromKernelAttributes<'a>>(
    attributes: &'a KernelAttributes,
    name: &str,
    onnx_type: &str,
) -> ort::Result<T> {
    attributes.get(name).ok_or_else(|| {
        Error::new(format!(
            "`{OP_NAME}` node has no `{name}` attribute readable as an ONNX {onnx_type}. \
             Build the graph with `python -m hf2mobile.postprocess <export dir>`."
        ))
    })
}

/// `logits[:, -1:, :]` — the final vocabulary row of a `[1, L, vocab]` tensor.
///
/// `None` if `shape` is not that: a wrong rank, a batch this operator cannot speak for
/// (it returns one token, so more than one batch row has nowhere to go), an empty
/// vocabulary, or fewer elements than the shape claims.
fn last_row<'a, T>(shape: &[i64], data: &'a [T]) -> Option<&'a [T]> {
    let [1, sequence, vocabulary] = *shape else {
        return None;
    };
    let vocabulary = usize::try_from(vocabulary).ok().filter(|&v| v > 0)?;
    usize::try_from(sequence).ok().filter(|&l| l > 0)?;
    data.len().checked_sub(vocabulary).map(|start| &data[start..])
}

/// The compiled node: a sampling policy, and the dtype of the row it reads.
struct SampleKernel<T: LogitElement> {
    cfg: Sampling,
    _element: PhantomData<T>,
}

impl<T: LogitElement> Kernel for SampleKernel<T> {
    fn compute(&mut self, ctx: &KernelContext) -> ort::Result<()> {
        let logits = ctx
            .input(0)?
            .ok_or_else(|| Error::new(format!("`{OP_NAME}` was given no input")))?;
        // Borrows ORT's own buffer — no copy of the row, at any vocab size.
        let (shape, data) = logits.try_extract_tensor::<T>()?;

        // Prefill hands over `[1, L, vocab]` and decode `[1, 1, vocab]`; both are the same
        // case, because only the final row predicts the next token — the earlier ones
        // predict tokens the caller already has. So this is `logits[:, -1:, :]`, done by
        // slicing the tail of the buffer rather than by a Slice node in the graph. It reads
        // no more than it needs: on a 1k-token prompt with a 150k vocab, the rows skipped
        // here are hundreds of megabytes.
        let row = last_row(shape, data).ok_or_else(|| {
            Error::new(format!(
                "`{OP_NAME}` wants logits shaped [1, L, vocab] (L >= 1), and got {:?}",
                shape.to_vec()
            ))
        })?;

        let mut output = ctx
            .output(0, [1_i64, 1])?
            .ok_or_else(|| Error::new(format!("`{OP_NAME}` was given nowhere to write its output")))?;
        let (_, slot) = output.try_extract_tensor_mut::<i32>()?;
        // `sample` dispatches to argmax by itself when the policy is greedy. The id is an
        // index into the vocabulary, so int32 holds every real one.
        slot[0] = sample(row, self.cfg) as i32;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn last_row_takes_the_tail_of_a_prefill_pass() {
        let data: Vec<i32> = (0..12).collect();
        // Decode: one row, taken whole.
        assert_eq!(last_row(&[1, 1, 12], &data), Some(&data[..]));
        // Prefill: three rows, only the last one.
        assert_eq!(last_row(&[1, 3, 4], &data), Some(&data[8..]));
        assert_eq!(last_row(&[1, 12, 1], &data), Some(&data[11..]));
    }

    #[test]
    fn last_row_rejects_what_it_cannot_speak_for() {
        let data: Vec<i32> = (0..12).collect();
        assert_eq!(last_row(&[12], &data), None, "rank 1");
        assert_eq!(last_row(&[1, 3, 2, 2], &data), None, "rank 4");
        assert_eq!(
            last_row(&[2, 3, 2], &data),
            None,
            "batch > 1 has nowhere to put a second token"
        );
        assert_eq!(last_row(&[1, 3, 0], &data), None, "empty vocabulary");
        assert_eq!(last_row(&[1, 0, 4], &data), None, "no rows at all");
        assert_eq!(
            last_row(&[1, 1, 16], &data),
            None,
            "shape claims more than the buffer holds"
        );
    }
}
