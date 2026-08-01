//! `com.hf2mobile:SampleLogits` — the sampling step, as a graph node.
//!
//! ```text
//! logits [1, L, vocab]  --SampleLogits(top_k, top_p, temperature)-->  sampled_token [1, 1] int32
//! ```
//!
//! The policy is baked into the node's attributes at export time (`python -m
//! hf2mobile.postprocess`), not passed per call, because the point of the operator is that the
//! caller never sees the logits: a 262k-wide fp32 row is 1 MB per token, and copying it out of
//! ONNX Runtime only to reduce it to one integer is the most expensive thing a decode step does
//! that isn't arithmetic.
//!
//! # The two halves of this file
//!
//! 1. **The ONNX Runtime boundary** — read three attributes once per session, find the last
//!    logits row, write one int32. This is the part that knows about ORT.
//! 2. **The policy** ([`sample`]) — temperature, top_k, top_p, and the draw. Plain arithmetic
//!    over a slice of floats, which is why it is also the part that can be unit-tested.
//!
//! # Who compiles it
//!
//! Both of them, twice. `src/onnx_plugins` builds this file into `libhf2mobile_plugins.so` for
//! ONNX Runtime to load (from Python, C++, or an Android app), and `src/onnx_inferencer` pulls
//! the same source in with `#[path]` and registers the operator in-process. One definition, so
//! the token a mobile runtime picks and the token the dev runtime picks cannot drift apart.

use core::marker::PhantomData;

use ort::operator::attribute::FromKernelAttributes;
use ort::operator::io::{OperatorInput, OperatorOutput};
use ort::operator::kernel::{Kernel, KernelAttributes, KernelContext};
use ort::operator::{Operator, ShapeInferenceContext};
use ort::tensor::{PrimitiveTensorElementType, Shape, SymbolicDimensions, TensorElementType};
use ort::value::ValueType;
use ort::Error;
use rand::Rng;

/// The op type the exporter writes into the graph (`SAMPLE_LOGITS_OP` in `constant.py`).
pub const OP_NAME: &str = "SampleLogits";

/// The ONNX domain the exporter writes its custom nodes under (`ONNX_DOMAIN_NAME` in
/// `constant.py`). A node's domain has to match the domain the operator was registered in,
/// or ORT reports the op as simply not found — so it lives next to the operator, and both
/// crates that register it read it from here.
pub const DOMAIN: &str = "com.hf2mobile";

// ══════════════════════════════ the ONNX Runtime boundary ══════════════════════════════

/// A logits element type this operator has a kernel for.
///
/// fp32 only, for now. ORT matches a kernel to a node by input type, so a graph exported in
/// another float type needs its own registration: an `impl LogitElement for half::f16` here
/// and an `.add(SampleLogits::<half::f16>::new())` where the domain is built — the bounds are
/// already the ones [`sample`] needs, and `Into<f32>` is how a narrower row is widened as the
/// candidate list is built, without copying it first.
pub trait LogitElement: PrimitiveTensorElementType + Copy + PartialOrd + Into<f32> + Send + 'static {}
impl LogitElement for f32 {}

/// The operator descriptor: one instance per logits dtype, all under the one op name.
///
/// `PhantomData<T>` is a zero-sized field that exists only to record which `T` this descriptor
/// is for. Without it the compiler would reject a type parameter that appears nowhere in the
/// struct's data.
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

// ═════════════════════════════════════ the policy ═════════════════════════════════════
//
// The knobs, and the order they apply in:
//
// 1. **temperature** divides the logits. Below 1 sharpens the distribution toward what the
//    model is confident about; above 1 flattens it. `0` means greedy — take the single best
//    token — which is deterministic, and so the right setting for checking an export against
//    the model it came from.
// 2. **top_k** keeps only the `k` highest-scoring tokens.
// 3. **top_p** (nucleus) keeps the smallest set of tokens whose probabilities already sum to
//    `p`, so a confident step considers few candidates and an uncertain one considers many.
//
// This is the order HuggingFace's `generate` uses. It matters: top_k on raw logits then top_p
// on the renormalized survivors is not the same as the reverse.

/// How to turn logits into a token. Built once per session, from the graph node's attributes.
#[derive(Clone, Copy)]
pub struct Sampling {
    /// Divides the logits. `<= 0` selects greedy decoding and ignores the other two.
    pub temperature: f32,
    /// Keep only the `k` best tokens. `0` disables the filter.
    pub top_k: usize,
    /// Keep the smallest set of tokens whose probabilities sum to at least this.
    /// `>= 1.0` disables the filter.
    pub top_p: f32,
}

impl Sampling {
    /// Is this configuration just "take the best token"?
    ///
    /// Worth asking, because greedy needs no copy of the logits — [`argmax`] reads ORT's
    /// buffer where it lies, while sampling has to materialize and rescale a row that can be
    /// 150k floats wide.
    fn is_greedy(self) -> bool {
        self.temperature <= 0.0
    }
}

/// Draw a token from `row` under `cfg`. Returns an index into the vocabulary.
///
/// Generic over the element type so it can take the model's own logits row — `f32`, `f16` or
/// `bf16` — and convert as it builds the candidate list, which means widening to `f32` costs no
/// allocation of its own. (`Into<f32>` is the bound that says "this type can widen to f32";
/// `half`'s types implement it.)
pub fn sample<T: Copy + PartialOrd + Into<f32>>(row: &[T], cfg: Sampling) -> usize {
    if row.is_empty() {
        return 0;
    }
    if cfg.is_greedy() {
        return argmax(row);
    }

    // (id, score) pairs, because every filter below reorders the scores and we still need to
    // know which token each one belongs to. `u32` rather than `usize` keeps the pair at 8
    // bytes instead of 16 (a `usize` pairs with an `f32` only after 4 bytes of padding) — at a
    // 262k vocab that is 2 MB per sampled token rather than 4 MB, and no vocabulary comes
    // anywhere near 2^32.
    let mut candidates: Vec<(u32, f32)> = row
        .iter()
        .enumerate()
        .map(|(id, &logit)| (id as u32, logit.into() / cfg.temperature))
        .collect();

    // top_k first, and via `select_nth_unstable` rather than a sort: it partitions in
    // O(vocab) instead of O(vocab log vocab), which is worth doing once per token.
    if cfg.top_k > 0 && cfg.top_k < candidates.len() {
        candidates.select_nth_unstable_by(cfg.top_k, |a, b| b.1.total_cmp(&a.1));
        candidates.truncate(cfg.top_k);
    }

    // Sorting is what makes top_p a prefix scan, and it is the *only* thing that needs the
    // order: the softmax below wants nothing but the maximum. So sort only when top_p will
    // actually scan, and otherwise take the maximum in one O(n) pass.
    //
    // This is not a corner case. `temperature` alone — top_k and top_p left at their
    // defaults — is the natural way to ask for sampling, and it skips the top_k filter above
    // too, so the sort would be an O(V log V) route to a maximum over the *whole* vocabulary,
    // every token.
    let max = if cfg.top_p < 1.0 {
        candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        candidates[0].1
    } else {
        // `f32::max` returns the other operand when one is NaN, so NaNs drop out here the
        // same way they lose every comparison in `argmax`.
        candidates
            .iter()
            .map(|&(_, score)| score)
            .fold(f32::NEG_INFINITY, f32::max)
    };

    // Softmax, shifted by the maximum so `exp` cannot overflow.
    let mut total = 0.0f32;
    for (_, score) in candidates.iter_mut() {
        *score = (*score - max).exp();
        total += *score;
    }

    if cfg.top_p < 1.0 {
        // Scaling the threshold by `total` once beats normalizing every probability by it:
        // the same cut, with one multiply instead of a division per candidate.
        let cutoff = cfg.top_p * total;
        let mut cumulative = 0.0f32;
        let mut keep = 0;
        for (_, probability) in candidates.iter() {
            cumulative += probability;
            keep += 1;
            if cumulative >= cutoff {
                break;
            }
        }
        // `keep` is at least 1: the loop always runs once for a non-empty candidate list,
        // so even a tiny top_p leaves the most likely token rather than nothing.
        candidates.truncate(keep);
        total = candidates.iter().map(|(_, p)| p).sum();
    }

    // Roulette wheel over the survivors: walk the list subtracting probabilities from a random
    // point until it runs out. `total` is their unnormalized sum, so there is no need to divide
    // through first.
    let mut point = rand::rng().random_range(0.0..total);
    for &(id, probability) in &candidates {
        point -= probability;
        if point <= 0.0 {
            return id as usize;
        }
    }
    // Only reachable through floating-point drift in the subtraction above.
    candidates[candidates.len() - 1].0 as usize
}

/// Index of the largest element. Ties go to the first. NaNs lose every comparison, so they are
/// skipped rather than poisoning the result — including a NaN in the *first* position, which is
/// the case worth spelling out: seeding the running best with one would pin the answer to index
/// 0, because every later `score > NaN` is false. Returns 0 for an empty row or one that is
/// entirely NaN.
///
/// The running best is carried in a local rather than re-read as `row[best]`, and the walk goes
/// through the iterator rather than indexing: both spare the loop a bounds check and a reload
/// per element, which at a 262k vocab it pays once per greedy token.
fn argmax<T: PartialOrd + Copy>(row: &[T]) -> usize {
    // Seed from the first score that can be compared at all: a NaN does not even order
    // against itself, so `partial_cmp` returns `None` for one and `Some` for everything
    // else. (`v == v` says the same thing, but clippy reads it as a mistake.) This stops on
    // the first element in the normal case, and keeping the test out here leaves the loop
    // below with one comparison per element rather than an is-it-seeded branch as well.
    let seed = row
        .iter()
        .enumerate()
        .find_map(|(i, &v)| v.partial_cmp(&v).is_some().then_some((i, v)));
    // `let ... else` is the early return for "no comparable element at all".
    let Some((mut best, mut best_score)) = seed else {
        return 0;
    };
    for (i, &score) in row.iter().enumerate().skip(best + 1) {
        if score > best_score {
            best = i;
            best_score = score;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(temperature: f32, top_k: usize, top_p: f32) -> Sampling {
        Sampling {
            temperature,
            top_k,
            top_p,
        }
    }

    // ── the row the kernel reads ──────────────────────────────────────────────

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

    // ── picking a token ───────────────────────────────────────────────────────

    #[test]
    fn argmax_basics() {
        assert_eq!(argmax(&[1.0f32, 5.0, 3.0]), 1);
        assert_eq!(argmax(&[5.0f32, 5.0, 1.0]), 0, "ties go to the first");
        assert_eq!(argmax::<f32>(&[]), 0);
        assert_eq!(argmax(&[-3.0f32, -1.0, -2.0]), 1, "all-negative");
    }

    #[test]
    fn argmax_skips_nan_wherever_it_sits() {
        assert_eq!(argmax(&[1.0f32, f32::NAN, 3.0]), 2, "interior NaN must not win");
        // The regression this guards: a leading NaN used to become the running best and
        // pin the answer to index 0, because every later `score > NaN` is false.
        assert_eq!(argmax(&[f32::NAN, 2.0]), 1, "leading NaN must not win");
        assert_eq!(argmax(&[f32::NAN, 2.0, 9.0, 4.0]), 2, "leading NaN, max later");
        assert_eq!(argmax(&[f32::NAN, f32::NAN]), 0, "all NaN falls back to 0");
    }

    #[test]
    fn argmax_over_f16() {
        let row: Vec<half::f16> = [1.0f32, 9.0, 4.0].iter().map(|&v| half::f16::from_f32(v)).collect();
        assert_eq!(argmax(&row), 1);
    }

    // A spike so sharp that every path must land on it, exercising both the
    // sorted (top_p < 1) and unsorted (top_p == 1) max branches.
    #[test]
    fn both_max_branches_agree_on_a_spike() {
        let mut row = vec![-50.0f32; 4096];
        row[1234] = 50.0;
        for _ in 0..64 {
            assert_eq!(sample(&row, cfg(1.0, 0, 1.0)), 1234, "unsorted max branch");
            assert_eq!(sample(&row, cfg(1.0, 0, 0.9)), 1234, "sorted max branch");
            assert_eq!(sample(&row, cfg(1.0, 8, 1.0)), 1234, "top_k, unsorted max");
            assert_eq!(sample(&row, cfg(1.0, 8, 0.9)), 1234, "top_k then top_p");
        }
    }

    // top_p must never widen the field beyond the true top tokens.
    #[test]
    fn top_p_keeps_only_the_head() {
        let row: Vec<f32> = (0..512).map(|i| i as f32 * 0.5).collect(); // max at 511
        for _ in 0..256 {
            let id = sample(&row, cfg(1.0, 0, 0.5));
            assert!(id > 480, "top_p=0.5 leaked a low-probability token: {id}");
        }
    }

    #[test]
    fn greedy_and_empty_are_handled() {
        let row = vec![1.0f32, 7.0, 2.0];
        assert_eq!(sample(&row, cfg(0.0, 0, 1.0)), 1, "temperature 0 is greedy");
        assert_eq!(sample::<f32>(&[], cfg(1.0, 0, 1.0)), 0);
    }

    // top_k larger than the vocabulary must not panic or truncate wrongly.
    #[test]
    fn top_k_beyond_vocab() {
        let row = vec![1.0f32, 9.0, 2.0];
        for _ in 0..32 {
            assert!(sample(&row, cfg(0.5, 999, 1.0)) < 3);
        }
    }
}
