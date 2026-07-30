//! Choosing the next token from a row of logits.
//!
//! The model hands back one score per vocabulary entry. Turning those into a single token
//! is a policy decision, kept here so the decode loop in [`crate::causal_lm`] stays about
//! running the graph.
//!
//! # The knobs, and the order they apply in
//!
//! 1. **temperature** divides the logits. Below 1 sharpens the distribution toward what
//!    the model is confident about; above 1 flattens it. `0` means greedy — take the
//!    single best token — which is the default because it is deterministic, and comparing
//!    an export against the original model needs determinism.
//! 2. **top_k** keeps only the `k` highest-scoring tokens.
//! 3. **top_p** (nucleus) keeps the smallest set of tokens whose probabilities already sum
//!    to `p`, so a confident step considers few candidates and an uncertain one considers
//!    many.
//!
//! This is the order HuggingFace's `generate` uses. It matters: top_k on raw logits then
//! top_p on the renormalized survivors is not the same as the reverse.

use rand::Rng;

/// How to turn logits into a token.
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

impl Default for Sampling {
    /// Greedy: reproducible, and the right baseline for checking an export.
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
        }
    }
}

impl Sampling {
    /// Is this configuration just "take the best token"?
    ///
    /// Worth asking, because greedy needs no copy of the logits — [`argmax`] reads ORT's
    /// buffer where it lies, while sampling has to materialize and rescale a row that can
    /// be 150k floats wide.
    pub fn is_greedy(self) -> bool {
        self.temperature <= 0.0
    }
}

/// Index of the largest element. Ties go to the first, and NaNs lose every comparison, so
/// they are skipped rather than poisoning the result.
pub fn argmax<T: PartialOrd>(row: &[T]) -> usize {
    let mut best = 0;
    for i in 1..row.len() {
        if row[i] > row[best] {
            best = i;
        }
    }
    best
}

/// Draw a token from `logits` under `cfg`.
///
/// `logits` is consumed as a scratch buffer — the caller already had to copy the row to
/// convert it to `f32`, so there is nothing to gain by copying it again here.
pub fn sample(logits: Vec<f32>, cfg: Sampling) -> usize {
    if logits.is_empty() {
        return 0;
    }
    if cfg.is_greedy() {
        return argmax(&logits);
    }

    // (id, score) pairs, because every filter below reorders the scores and we still need
    // to know which token each one belongs to.
    let mut candidates: Vec<(usize, f32)> = logits
        .into_iter()
        .enumerate()
        .map(|(id, logit)| (id, logit / cfg.temperature))
        .collect();

    // top_k first, and via `select_nth_unstable` rather than a sort: it partitions in
    // O(vocab) instead of O(vocab log vocab), which is worth doing once per token.
    if cfg.top_k > 0 && cfg.top_k < candidates.len() {
        candidates.select_nth_unstable_by(cfg.top_k, |a, b| b.1.total_cmp(&a.1));
        candidates.truncate(cfg.top_k);
    }

    // Sorting is what makes top_p a prefix scan. After a top_k filter this is over `k`
    // elements, not the whole vocabulary.
    candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

    // Softmax, shifted by the maximum so `exp` cannot overflow — the scores are already
    // sorted, so the maximum is the first one.
    let max = candidates[0].1;
    let mut total = 0.0f32;
    for (_, score) in candidates.iter_mut() {
        *score = (*score - max).exp();
        total += *score;
    }

    if cfg.top_p < 1.0 {
        let mut cumulative = 0.0f32;
        let mut keep = 0;
        for (_, probability) in candidates.iter() {
            cumulative += probability / total;
            keep += 1;
            if cumulative >= cfg.top_p {
                break;
            }
        }
        // `keep` is at least 1: the loop always runs once for a non-empty candidate list,
        // so even a tiny top_p leaves the most likely token rather than nothing.
        candidates.truncate(keep);
        total = candidates.iter().map(|(_, p)| p).sum();
    }

    // Roulette wheel over the survivors. `total` is their unnormalized sum, so there is no
    // need to divide through first.
    let mut point = rand::rng().random_range(0.0..total);
    for (id, probability) in &candidates {
        point -= probability;
        if point <= 0.0 {
            return *id;
        }
    }
    // Only reachable through floating-point drift in the subtraction above.
    candidates[candidates.len() - 1].0
}
