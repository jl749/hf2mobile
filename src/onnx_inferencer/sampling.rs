//! Choosing the next token from a row of logits.
//!
//! # Who calls this
//!
//! Not the decode loop — [`crate::causal_lm`] never sees logits. This is the body of the
//! `SampleLogits` operator in [`crate::sample_logits`], i.e. it runs *inside* the graph, once
//! per pass, on a row that never leaves ONNX Runtime.
//!
//! The file sits here rather than beside that operator for a build reason: `src/onnx_plugins`
//! compiles the same source into `libhf2mobile_plugins.so` for a mobile runtime to load. One
//! definition, two binaries, so the two can never disagree about which token a policy picks.
//!
//! # The knobs, and the order they apply in
//!
//! 1. **temperature** divides the logits. Below 1 sharpens the distribution toward what the
//!    model is confident about; above 1 flattens it. `0` means greedy — take the single best
//!    token — which is deterministic, and so the right setting for checking an export against
//!    the model it came from.
//! 2. **top_k** keeps only the `k` highest-scoring tokens.
//! 3. **top_p** (nucleus) keeps the smallest set of tokens whose probabilities already sum to
//!    `p`, so a confident step considers few candidates and an uncertain one considers many.
//!
//! This is the order HuggingFace's `generate` uses. It matters: top_k on raw logits then top_p
//! on the renormalized survivors is not the same as the reverse.

use rand::Rng;

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
