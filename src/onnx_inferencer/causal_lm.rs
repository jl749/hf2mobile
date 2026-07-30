//! Text generation over an exported causal-LM graph: tokenize, prefill, decode.
//!
//! # The one graph, two jobs trick
//!
//! The exported graph takes `input_ids` of a *dynamic* length `L`, so the same graph does
//! both halves of generation:
//!
//! - **prefill** — `L = prompt length`, empty [`KvCache`]. One pass reads the whole prompt
//!   and predicts the first token. This is what time-to-first-token measures.
//! - **decode** — `L = 1`, cache holding everything so far. One pass per token.
//!
//! Both are the same call to [`forward`]; only the slice of tokens differs.
//!
//! # Turns
//!
//! [`CausalLm::generate`] does not run a generation to completion — it advances one. A
//! *turn* opens on the first call, survives as many further calls as the caller makes, and
//! closes when an end-of-sequence token arrives. The [`KvCache`], the token history and the
//! timings all live in [`Turn`] across those calls, so a caller can poll one token at a time
//! and still pay for prefill exactly once. Re-prefilling the prompt on every call instead
//! would be quadratic in the length of the reply.
//!
//! # Where the token comes from
//!
//! Nothing here reads logits. The graph this file runs (`inference.onnx`, from `python -m
//! hf2mobile.postprocess`) ends in a `SampleLogits` node and returns the token it picked as a
//! `[1, 1]` int32. Two consequences: the sampling policy is not an argument here, because it
//! was baked into that node at export time; and a row of logits never leaves ONNX Runtime,
//! which at a 262k vocabulary is 1 MB per token not copied.
//!
//! Where a turn *stops* comes out of the graph too — the ids in `hf2mobile_EOS_tokens`, read
//! by [`crate::onnx_file`]. So opening a model needs a graph and a tokenizer, nothing else.
//!
//! # Timing
//!
//! TTFT and tokens/sec are measured here, around the `Session::run` call and nothing else,
//! so tokenizing and streaming to stdout stay out of the model's numbers. ONNX Runtime has no
//! per-run latency accessor — its profiler reports per-*operator* spans to a chrome trace
//! (see `DEBUG=1`) — so an `Instant` around the call is the measurement.

use std::io::Write;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use ort::session::{Session, SessionInputValue};
use ort::value::TensorRef;
use tokenizers::Tokenizer;

use crate::kv_cache::KvCache;
use crate::onnx_file;
use crate::session;

/// The graph output `SampleLogits` writes its token into (`SAMPLED_TOKEN_NAME` in
/// `constant.py`).
const SAMPLED_TOKEN: &str = "sampled_token";

/// The `Constant` node `python -m hf2mobile.postprocess` parks the stop ids in
/// (`EOS_TOKENS_CONST_NAME` in `constant.py`).
const EOS_TOKENS_CONST: &str = "hf2mobile_EOS_tokens";

/// What one `generate` call produced.
pub struct Step {
    /// The turn's complete output — but only once an end-of-sequence token has arrived.
    /// `None` while the turn is still open, which is how a caller polling one token at a
    /// time learns whether to keep going.
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
    /// dropped, which when polling one token at a time is *every* boundary.
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

    /// This turn's numbers so far, plus `text` if the turn just closed.
    fn report(&self, text: Option<String>) -> Step {
        let decode_s = self.decode_time.as_secs_f64();
        Step {
            text,
            ttft_s: self.ttft.as_secs_f64(),
            // Guard the division: before any decode pass there is no elapsed time to divide
            // by, and 0/0 would be a NaN crossing into Python.
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
/// Which of these a graph asks for depends on how it was exported. A plain attention export
/// wants `position_ids` and works out rotary embeddings from them; an export where ONNX
/// Runtime's `GroupQueryAttention` has been fused in computes positions internally and wants
/// two length counters instead.
///
/// We resolve the list once at load time — so an unfamiliar input fails immediately with a
/// clear message rather than on the first token — and only rebuild the (tiny) tensors per
/// step. `Clone, Copy` are derived because this is four bytes describing which input it is:
/// copying one is cheaper than referring to it, so the compiler is told to just copy.
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

    /// The name to feed it back to ORT under. The mirror image of `from_name`, kept next to
    /// it so the two spellings of each name are one line apart.
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
    /// End-of-sequence ids `generate` stops on, as the graph itself declares them — see
    /// [`open`](Self::open).
    eos_tokens: Vec<i64>,
    /// The turn in progress. `Option` is how Rust spells "maybe there is one": `None` means
    /// the next `generate` opens a fresh turn, and the compiler will not let any code read
    /// the inside without saying which case it is handling.
    turn: Option<Turn>,
}

impl CausalLm {
    /// Load a generation graph and its tokenizer.
    ///
    /// The stop ids are read out of the graph rather than passed in. They cannot be guessed
    /// from `tokenizer.json` — that records a vocabulary, not a chat protocol, so it has no
    /// "this is the EOS" field, and a guess that misses turns into generation that never
    /// stops. What does know is the export, and `python -m hf2mobile.postprocess` writes the
    /// answer into the graph as `hf2mobile_EOS_tokens`. An empty list there is a real answer
    /// (this model names no stop token, so only the caller's budget ends a turn); a *missing*
    /// node means the graph never went through postprocess, and is refused.
    pub fn open(onnx_path: &str, tokenizer_path: &str, intra_threads: Option<usize>) -> Result<Self> {
        // `?` on a fallible call means "unwrap it, or return the error to my caller".
        // `with_context` hangs a sentence off that error on the way out, and anyhow keeps the
        // whole chain — which is what turns a failure deep in ORT into a Python exception
        // reading "opening ONNX model `x.onnx`: ...".
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
            // `collect` into a `Result<Vec<_>>` stops at the first input we did not
            // recognise and returns that error, instead of building a list of maybes.
            .collect::<Result<Vec<_>>>()?;

        // `tokenizers` reports errors as boxed trait objects rather than an error type,
        // so they need `map_err` before `?` will accept them.
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("loading tokenizer `{tokenizer_path}`"))?;

        // Read last, so a graph that is wrong in a more fundamental way (bf16, no KV cache)
        // reports that first.
        let eos_tokens = onnx_file::int_constant(onnx_path, EOS_TOKENS_CONST)?.with_context(|| {
            format!(
                "`{onnx_path}` carries no `{EOS_TOKENS_CONST}` node, so nothing in it says where a \
                 turn ends. This runtime decodes the postprocessed graph — build it with:\n\
                 \n    python -m hf2mobile.postprocess <export dir>\n"
            )
        })?;
        if eos_tokens.is_empty() {
            eprintln!(
                "[hf2mobile] WARNING: `{EOS_TOKENS_CONST}` is empty, so this graph names no \
                 end-of-sequence token. Generation will run until the caller's budget is spent."
            );
        }

        Ok(Self {
            session,
            tokenizer,
            cache,
            step_inputs,
            eos_tokens,
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
        // `map` looks inside the `Option` if there is a turn; `unwrap_or_default` supplies an
        // empty `Vec` if there is not. The clone is deliberate: the caller is Python, which
        // cannot hold a borrow of Rust's own list.
        self.turn
            .as_ref()
            .map(|turn| turn.generated.clone())
            .unwrap_or_default()
    }

    /// How many tokens the current turn's prompt came to.
    pub fn prefill_len(&self) -> usize {
        self.turn.as_ref().map_or(0, |turn| turn.prefill_len)
    }

    /// How many KV cache tensors the graph declared — two per layer.
    pub fn kv_slots(&self) -> usize {
        self.cache.slot_count()
    }

    /// End-of-sequence ids `generate` stops on, as the graph declared them.
    pub fn eos_tokens(&self) -> &[i64] {
        &self.eos_tokens
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

    /// Continue the current turn, or open one if there is none.
    ///
    /// `prompt` is only read when a turn opens; later calls continue from the KV cache and
    /// ignore it. That is the point: a caller can poll one token at a time and the prefill
    /// is paid for once.
    ///
    /// Returns [`Step::text`] as `Some` exactly once — on the call where an end-of-sequence
    /// token arrives. After that the turn is closed and further calls warn and do nothing
    /// until [`reset`](Self::reset).
    pub fn generate(&mut self, prompt: &str, num_generation: usize, stream_output: bool) -> Result<Step> {
        // Take the fields apart up front. The loop below borrows the tokenizer (to stream
        // text) at the same time as it mutably borrows the session (to run the graph). Rust
        // forbids handing out `&mut self` and `&self` at once, but it happily allows one
        // borrow per *field* — and naming them like this is how the compiler sees that these
        // are different fields and cannot alias.
        let Self {
            session,
            tokenizer,
            cache,
            step_inputs,
            eos_tokens,
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
            // `as i64` widens each id: the tokenizer counts in u32, the graph wants int64.
            let prompt_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
            if prompt_ids.is_empty() {
                // A causal LM predicts token n+1 from tokens 0..n, so it needs at least one
                // to start from. Caught here because the alternative is a `[1, 0]` input and
                // an opaque complaint from ORT about a zero-length dimension.
                anyhow::bail!("prompt is empty: it tokenized to no tokens at all");
            }
            cache.reset()?;
            let (pending, ttft) = forward(session, cache, step_inputs, &prompt_ids, 0)?;
            *turn = Some(Turn::new(prompt_ids.len(), pending, ttft));
        }
        // Safe to unwrap the `Option` now: either it held a turn on entry or the block above
        // just put one there. `expect` documents that reasoning and would panic if it broke.
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
                stream_token(tokenizer, turn);
            }

            let (next, elapsed) = forward(session, cache, step_inputs, &[turn.pending], turn.kv_len)?;
            turn.pending = next;
            turn.kv_len += 1;
            // Accumulating per-run durations, rather than timing the loop, keeps the write
            // to stdout above (a syscall per token) out of the throughput.
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

/// Print the text `turn.pending` adds to what is already on screen.
///
/// Detokenizing incrementally, not by re-decoding the whole reply each token: the stream
/// state in [`Turn`] remembers how much text has been printed. A `None` piece means the text
/// so far ends mid-character, and the decoder holds it back rather than printing a broken
/// one — so errors here are silently ignored, because streaming is a nicety and losing a
/// character to it must not fail a generation.
fn stream_token(tokenizer: &Tokenizer, turn: &mut Turn) {
    let piece = tokenizers::step_decode_stream(
        tokenizer,
        turn.pending as u32,
        true,
        &mut turn.stream.ids,
        &mut turn.stream.prefix,
        &mut turn.stream.prefix_index,
    );
    if let Ok(Some(piece)) = piece {
        print!("{piece}");
        // stdout is line-buffered, and tokens rarely end in a newline, so without this the
        // text would appear in chunks instead of as it is produced.
        let _ = std::io::stdout().flush();
    }
}

/// Run `tokens` through the graph; return the next token id and how long ORT took.
///
/// The returned [`Duration`] covers `Session::run` alone — not building the input tensors —
/// because that is the number a change to the model or the provider actually moves. Picking
/// the token is inside that window now: it is a node in the graph.
///
/// `kv_len` is how many tokens the KV cache already holds, which is also the absolute
/// position of `tokens[0]`. Every length the graph asks for is derived from it, so there is
/// only one counter to get wrong.
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
    Ok((sampled_token(&outputs)?, elapsed))
}

/// Read the one id the graph picked out of its `sampled_token` output.
///
/// Four bytes, whatever the prompt length and whatever the vocabulary size — the
/// `SampleLogits` node at the end of the graph already did the reducing.
fn sampled_token(outputs: &ort::session::SessionOutputs<'_>) -> Result<i64> {
    let value = outputs.get(SAMPLED_TOKEN).with_context(|| {
        format!(
            "graph has no `{SAMPLED_TOKEN}` output. This runtime decodes the postprocessed graph, \
             which ends in a `SampleLogits` node — build it with:\n\
             \n    python -m hf2mobile.postprocess <export dir>\n"
        )
    })?;
    // `try_extract_tensor` borrows ORT's buffer and checks the element type while doing so;
    // the shape (the discarded first half of the pair) is `[1, 1]` and tells us nothing.
    let (_, data) = value
        .try_extract_tensor::<i32>()
        .with_context(|| format!("`{SAMPLED_TOKEN}` is not an int32 tensor"))?;
    let token = *data
        .first()
        .with_context(|| format!("`{SAMPLED_TOKEN}` came back empty; it should hold one id"))?;
    Ok(i64::from(token))
}
