//! Text generation over an exported causal-LM graph: tokenize, prefill, decode.
//!
//! # The one graph, two jobs trick
//!
//! The exported graph takes `input_ids` of a *dynamic* length `L`, so the same graph
//! does both halves of generation:
//!
//! - **prefill** — `L = prompt length`, empty [`KvCache`]. One pass reads the whole
//!   prompt and predicts the first token. This is what time-to-first-token measures.
//! - **decode** — `L = 1`, cache holding everything so far. One pass per token.
//!
//! Both are the same call to [`forward`]; only the slice of tokens differs.
//!
//! # Turns
//!
//! [`CausalLm::generate`] does not run a generation to completion — it advances one. A
//! *turn* opens on the first call, survives as many further calls as the caller makes, and
//! closes when an end-of-sequence token arrives. The [`KvCache`], the token history and the
//! timings all live in [`Turn`] across those calls, so a caller can poll with
//! `num_generation = 1` and still pay for prefill exactly once.
//!
//! This is what makes token-at-a-time control cheap. The alternative — re-prefilling the
//! prompt on every call — is quadratic in the length of the reply.
//!
//! # Sampling
//!
//! How a token is picked from the logits lives in [`crate::sampling`]; this file only
//! decides *when* to pick one. The default is greedy, which is deterministic and so the
//! right baseline for checking that an export still produces what the original model did.
//!
//! # Timing
//!
//! TTFT and tokens/sec are measured here rather than in Python, around the
//! `Session::run` call and nothing else. Tokenizing, sampling, and streaming to stdout are
//! all excluded, so the numbers describe the model rather than the harness. ONNX Runtime
//! has no per-run latency accessor — its profiler reports per-*operator* spans to a
//! chrome trace (see `DEBUG=1`) — so an `Instant` around the call is the measurement.

use std::io::Write;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use ort::session::{Session, SessionInputValue};
use ort::tensor::TensorElementType;
use ort::value::{DynValue, TensorRef};
use tokenizers::Tokenizer;

use crate::kv_cache::KvCache;
use crate::sampling::{self, Sampling};
use crate::session;

/// Token strings that mark end-of-turn across the model families this repo exports.
///
/// `tokenizer.json` records a vocabulary, not a chat protocol — it has no "this is the
/// EOS" field — so the id has to be recovered by name. Any model whose terminator is not
/// on this list needs `eos_tokens` passed explicitly, which is why that argument exists.
const KNOWN_EOS_TOKENS: [&str; 6] = [
    "<|im_end|>",    // Qwen2 / Qwen3 chat
    "<|endoftext|>", // Qwen base, GPT-2 lineage
    "<|eot_id|>",    // Llama 3 chat
    "</s>",          // Llama 2, Mistral
    "<end_of_turn>", // Gemma chat
    "<|end|>",       // Phi-3
];

/// What one `generate` call produced.
pub struct Step {
    /// The turn's complete output — but only once an end-of-sequence token has arrived.
    /// `None` while the turn is still open, which is how a caller polling with
    /// `num_generation = 1` learns whether to keep going.
    pub text: Option<String>,
    /// Time to first token: the prefill `Session::run`, in seconds. Fixed once the turn
    /// starts, and grows with prompt length since prefill reads the whole prompt at once.
    pub ttft_s: f64,
    /// Decode throughput in tokens/sec, over the whole turn: every token emitted so far
    /// divided by the summed decode `Session::run` time, however many calls that took.
    pub tps: f64,
}

/// Everything that survives from one `generate` call to the next.
///
/// A *turn* is one prompt and the reply being built from it. It opens on the first
/// `generate`, stays open across as many further calls as it takes, and closes when an
/// end-of-sequence token appears — at which point the KV cache is released and only
/// [`CausalLm::reset`] can start another.
struct Turn {
    /// How many tokens the KV cache holds, and so where the next one sits.
    kv_len: i64,
    /// The token the model has predicted but that has not been emitted yet. Held back so
    /// the next call can test it against `eos_tokens` before committing to it.
    pending: i64,
    /// Every token emitted this turn, across all calls.
    generated: Vec<i64>,
    /// How many tokens the prompt came to, i.e. the length of the prefill pass.
    prefill_len: usize,
    /// Prefill latency. Measured once, reported by every call.
    ttft: Duration,
    /// Summed decode `Session::run` time across every call in this turn.
    decode_time: Duration,
    /// Incremental-detokenizer state. Kept here rather than as a `tokenizers::DecodeStream`
    /// because that type borrows the tokenizer, which cannot outlive a single call — and
    /// this state has to. Without it, a character split across a call boundary would be
    /// dropped, which with `num_generation = 1` is *every* boundary.
    stream: StreamState,
    /// Set when an end-of-sequence token comes up. The turn is over.
    finished: bool,
}

/// The three pieces `tokenizers::step_decode_stream` threads through each step.
#[derive(Default)]
struct StreamState {
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
}

impl Turn {
    fn new(prefill_len: usize, pending: i64, ttft: Duration) -> Self {
        Self {
            kv_len: prefill_len as i64,
            pending,
            generated: Vec::new(),
            prefill_len,
            ttft,
            decode_time: Duration::ZERO,
            stream: StreamState::default(),
            finished: false,
        }
    }

    fn report(&self, text: Option<String>) -> Step {
        let decode_s = self.decode_time.as_secs_f64();
        Step {
            text,
            ttft_s: self.ttft.as_secs_f64(),
            tps: if decode_s > 0.0 {
                self.generated.len() as f64 / decode_s
            } else {
                0.0
            },
        }
    }
}

/// A per-step graph input that is *not* part of the KV cache.
///
/// Which of these a graph asks for depends on how it was exported. A plain attention
/// export wants `position_ids` and works out rotary embeddings from them; an export
/// where ONNX Runtime's `GroupQueryAttention` has been fused in computes positions
/// internally and wants two length counters instead.
///
/// We resolve the list once at load time — so an unfamiliar input fails immediately with
/// a clear message rather than on the first token — and only rebuild the (tiny) tensors
/// per step.
#[derive(Clone, Copy)]
enum StepInput {
    /// The tokens for this pass, `[1, L]` int64.
    InputIds,
    /// Where each token sits in the full sequence, `[1, L]` int64.
    PositionIds,
    /// GroupQueryAttention: total sequence length minus one, `[1]` int32.
    SeqLensK,
    /// GroupQueryAttention: cached + new tokens, `[1]` int32.
    TotalSequenceLength,
}

impl StepInput {
    /// Recognise a graph input by name, or `None` if it is not one of ours.
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "input_ids" => Some(Self::InputIds),
            "position_ids" => Some(Self::PositionIds),
            "seqlens_k" => Some(Self::SeqLensK),
            "total_sequence_length" => Some(Self::TotalSequenceLength),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::InputIds => "input_ids",
            Self::PositionIds => "position_ids",
            Self::SeqLensK => "seqlens_k",
            Self::TotalSequenceLength => "total_sequence_length",
        }
    }
}

/// A loaded model: ONNX graph, tokenizer, and the cache that ties decode steps together.
pub struct CausalLm {
    session: Session,
    tokenizer: Tokenizer,
    cache: KvCache,
    /// The non-cache inputs this particular graph declares, in its own order.
    step_inputs: Vec<StepInput>,
    /// The turn in progress, if any. `None` means the next `generate` starts a fresh one.
    turn: Option<Turn>,
}

impl CausalLm {
    /// Load a generation graph and its tokenizer.
    pub fn open(onnx_path: &str, tokenizer_path: &str, intra_threads: Option<usize>) -> Result<Self> {
        let session =
            session::open(onnx_path, intra_threads).with_context(|| format!("opening ONNX model `{onnx_path}`"))?;
        // A dtype this machine cannot execute is a re-export, not a runtime problem.
        crate::precision::ensure_executable(&session)?;
        // Whatever the cache does not supply is ours to fill in.
        let (cache, other_inputs) = KvCache::discover(&session)?;
        let step_inputs = other_inputs
            .into_iter()
            .map(|name| {
                StepInput::from_name(name)
                    .with_context(|| format!("graph wants an input this runtime does not know how to fill: `{name}`"))
            })
            .collect::<Result<Vec<_>>>()?;

        // `tokenizers` reports errors as boxed trait objects rather than an error type,
        // so they need `map_err` before `?` will accept them.
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("loading tokenizer `{tokenizer_path}`"))?;

        Ok(Self {
            session,
            tokenizer,
            cache,
            step_inputs,
            turn: None,
        })
    }

    /// Abandon the current turn and release its KV cache, so the next `generate` starts
    /// fresh from a new prompt.
    pub fn reset(&mut self) -> Result<()> {
        self.turn = None;
        self.cache.reset()
    }

    /// Is a turn open — started, and not yet ended by an end-of-sequence token?
    pub fn is_generating(&self) -> bool {
        self.turn.as_ref().is_some_and(|turn| !turn.finished)
    }

    /// Tokens emitted so far in the current turn.
    pub fn token_ids(&self) -> Vec<i64> {
        self.turn
            .as_ref()
            .map(|turn| turn.generated.clone())
            .unwrap_or_default()
    }

    /// How many tokens the current turn's prompt came to.
    pub fn prefill_len(&self) -> usize {
        self.turn.as_ref().map_or(0, |turn| turn.prefill_len)
    }

    /// The underlying session, for reading graph metadata.
    pub fn session(&self) -> &Session {
        &self.session
    }

    /// The underlying session, for the raw `run` escape hatch. Bypasses the KV cache, so
    /// a caller using this is running passes that this type knows nothing about.
    pub fn session_mut(&mut self) -> &mut Session {
        &mut self.session
    }

    /// End-of-turn ids recovered from the tokenizer's vocabulary, used when the caller
    /// does not name any. See [`KNOWN_EOS_TOKENS`] for why this is a lookup by name.
    pub fn default_eos_tokens(&self) -> Vec<i64> {
        KNOWN_EOS_TOKENS
            .iter()
            .filter_map(|token| self.tokenizer.token_to_id(token))
            .map(i64::from)
            .collect()
    }

    /// Continue the current turn, or open one if there is none.
    ///
    /// `prompt` is only read when a turn opens; later calls continue from the KV cache and
    /// ignore it. That is the point: a caller can poll with `num_generation = 1` and the
    /// prefill is paid for once.
    ///
    /// Returns [`Step::text`] as `Some` exactly once — on the call where an end-of-sequence
    /// token arrives. After that the turn is closed and further calls warn and do nothing
    /// until [`reset`](Self::reset).
    pub fn generate(
        &mut self,
        prompt: &str,
        num_generation: usize,
        eos_tokens: &[i64],
        stream_output: bool,
        sampling: Sampling,
    ) -> Result<Step> {
        // Destructure `self` into its fields up front. The decode loop borrows the
        // tokenizer (for streaming) at the same time as it mutably borrows the session,
        // which the compiler only allows once it can see the two are separate fields.
        let Self {
            session,
            tokenizer,
            cache,
            step_inputs,
            turn,
        } = self;

        if let Some(closed) = turn.as_ref().filter(|t| t.finished) {
            eprintln!(
                "[hf2mobile] WARNING: generate() did nothing. This turn already reached an \
                 end-of-sequence token, so its KV cache has been released. Call reset() before \
                 starting another generation."
            );
            return Ok(closed.report(None));
        }

        // Opening a turn means prefilling the prompt in one pass, which is what TTFT
        // measures. `forward` reports ORT's own time, so tokenizing is not counted.
        if turn.is_none() {
            let encoding = tokenizer.encode(prompt, true).map_err(anyhow::Error::msg)?;
            let prompt_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
            if prompt_ids.is_empty() {
                // A causal LM predicts token n+1 from tokens 0..n, so it needs at least one
                // to start from. Caught here because the alternative is a `[1, 0]` input and
                // an opaque complaint from ORT about a zero-length dimension.
                anyhow::bail!("prompt is empty: it tokenized to no tokens at all");
            }
            cache.reset()?;
            let (pending, ttft) = forward(session, cache, step_inputs, &prompt_ids, 0, sampling)?;
            *turn = Some(Turn::new(prompt_ids.len(), pending, ttft));
        }
        let turn = turn.as_mut().expect("a turn was just opened");

        for _ in 0..num_generation {
            // The pending token is tested *before* it is emitted, so a terminator ends the
            // turn without appearing in the output.
            if eos_tokens.contains(&turn.pending) {
                turn.finished = true;
                break;
            }
            turn.generated.push(turn.pending);

            if stream_output {
                let piece = tokenizers::step_decode_stream(
                    tokenizer,
                    turn.pending as u32,
                    true,
                    &mut turn.stream.ids,
                    &mut turn.stream.prefix,
                    &mut turn.stream.prefix_index,
                );
                // `None` means the text so far ends mid-character; the decoder holds it
                // back rather than printing a broken one.
                if let Ok(Some(piece)) = piece {
                    print!("{piece}");
                    let _ = std::io::stdout().flush();
                }
            }

            let (next, elapsed) = forward(session, cache, step_inputs, &[turn.pending], turn.kv_len, sampling)?;
            turn.pending = next;
            turn.kv_len += 1;
            // Accumulating per-run durations, rather than timing the loop, keeps the
            // streaming `print!` above (a syscall per token) out of the throughput.
            turn.decode_time += elapsed;
        }

        if !turn.finished {
            return Ok(turn.report(None));
        }

        if stream_output {
            println!();
        }
        let ids: Vec<u32> = turn.generated.iter().map(|&id| id as u32).collect();
        let text = tokenizer.decode(&ids, true).map_err(anyhow::Error::msg)?;
        // The turn is over and the cache is the largest thing we hold, so let it go now
        // rather than at the next `reset`.
        cache.reset()?;
        Ok(turn.report(Some(text)))
    }
}

/// Run `tokens` through the graph; return the next token id and how long ORT took.
///
/// The returned [`Duration`] covers `Session::run` alone — not building the input
/// tensors, not the argmax — because that is the number a change to the model or the
/// provider actually moves.
///
/// `kv_len` is how many tokens the KV cache already holds, which is also the absolute
/// position of `tokens[0]`. Every length the graph asks for is derived from it, so there
/// is only one counter to get wrong.
///
/// A free function rather than a method so it can borrow the session and the cache
/// separately — `&mut self` would lock both together and the caller could not also hold
/// the tokenizer.
fn forward(
    session: &mut Session,
    cache: &mut KvCache,
    step_inputs: &[StepInput],
    tokens: &[i64],
    kv_len: i64,
    sampling: Sampling,
) -> Result<(i64, Duration)> {
    let len = tokens.len() as i64;
    let total = kv_len + len; // sequence length once this pass is done

    // Built up front so they outlive the borrowed views below. All are a handful of
    // bytes — during decode, `len` is 1.
    let positions: Vec<i64> = (kv_len..total).collect();
    let seqlens_k = [(total - 1) as i32];
    let total_length = [total as i32];

    let mut inputs: Vec<(&str, SessionInputValue)> = Vec::with_capacity(step_inputs.len() + cache.slot_count());
    for &step in step_inputs {
        // `from_array_view` wraps a slice we already have rather than copying it; the
        // borrow ends when `run` below consumes `inputs`.
        let value = match step {
            StepInput::InputIds => TensorRef::from_array_view((vec![1, len], tokens))?.into_dyn(),
            StepInput::PositionIds => TensorRef::from_array_view((vec![1, len], positions.as_slice()))?.into_dyn(),
            StepInput::SeqLensK => TensorRef::from_array_view((vec![1], seqlens_k.as_slice()))?.into_dyn(),
            StepInput::TotalSequenceLength => {
                TensorRef::from_array_view((vec![1], total_length.as_slice()))?.into_dyn()
            }
        };
        inputs.push((step.name(), value.into()));
    }
    cache.bind(&mut inputs);

    let started = Instant::now();
    let mut outputs = session.run(inputs)?;
    let elapsed = started.elapsed();

    cache.take_update(&mut outputs)?;
    let logits = outputs.get("logits").context("graph has no `logits` output")?;
    Ok((pick_next(logits, sampling)?, elapsed))
}

/// Pick the next token from a `[1, seq, vocab]` logits tensor.
///
/// Only the final row predicts the next token — during prefill the earlier rows predict
/// tokens we already have — so we slice off the tail and ignore the rest. That matters:
/// on a 1k-token prompt with a 150k vocab, the rows we skip are hundreds of megabytes.
///
/// Greedy decoding reads ORT's buffer in place. Sampling has to rescale the scores, so it
/// copies the one row it needs into `f32` first — which also folds away the f16/bf16 case
/// before [`sampling`] ever sees it.
fn pick_next(logits: &DynValue, cfg: Sampling) -> Result<i64> {
    let (_, dtype) = session::tensor_type(logits.dtype()).context("`logits` is not a tensor")?;

    // `try_extract_tensor` borrows ORT's buffer directly — no copy, at any vocab size.
    // One arm per float type the model might be running in.
    macro_rules! pick {
        ($t:ty, $to_f32:expr) => {{
            let (shape, data) = logits.try_extract_tensor::<$t>()?;
            let vocab = *shape.last().context("`logits` has no vocab dimension")? as usize;
            let last_row = data
                .len()
                .checked_sub(vocab)
                .map(|start| &data[start..])
                .context("`logits` is smaller than one vocab row")?;

            let id = if cfg.is_greedy() {
                sampling::argmax(last_row)
            } else {
                sampling::sample(last_row.iter().copied().map($to_f32).collect(), cfg)
            };
            Ok(id as i64)
        }};
    }

    match dtype {
        TensorElementType::Float32 => pick!(f32, |v| v),
        TensorElementType::Float16 => pick!(half::f16, f32::from),
        TensorElementType::Bfloat16 => pick!(half::bf16, f32::from),
        other => anyhow::bail!("unsupported `logits` element type {other:?}"),
    }
}
