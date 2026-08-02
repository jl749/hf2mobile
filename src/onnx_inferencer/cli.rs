//! `hf2mobile-infer` — run an export from a shell, with no Python anywhere.
//!
//! The same run as `python -m hf2mobile.infer`, and deliberately the same flags, but as a
//! single executable: nothing here links against libpython, so the binary cross-compiles to
//! `aarch64-linux-android` and can be pushed to a phone next to the export directory. That is
//! the point of it — measuring a graph on the device it was exported for, without building an
//! app around it first.
//!
//! ```text
//! adb push target/aarch64-linux-android/release/hf2mobile-infer /data/local/tmp/
//! adb push 2026-08-01__ORT__Qwen-Qwen3-0.6B /data/local/tmp/
//! adb push libonnxruntime.so /data/local/tmp/
//! adb shell '/data/local/tmp/hf2mobile-infer /data/local/tmp/2026-08-01__ORT__Qwen-Qwen3-0.6B \
//!            --prompt "Where is Paris?" --num-generation 64'
//! ```
//!
//! # What it needs on the device
//!
//! The export directory, as written by `hf2mobile-export` — `inference.onnx` (plus its
//! `.onnx.data` if the weights are external), `tokenizer.json`, and the chat template
//! (`chat_template.jinja` / `tokenizer_config.json`, see [`crate::chat_template`]) — and an
//! ONNX Runtime shared library for the device's architecture.
//! The `SampleLogits` operator is compiled into this binary (see [`crate::session`]), so
//! `libhf2mobile_plugins.so` is *not* needed here; that one is for a runtime that is not this
//! one, such as an Android app driving ORT through its Java API.
//!
//! ONNX Runtime is dlopened rather than linked (the `load-dynamic` feature), so its path is a
//! runtime decision — see [`resolve_ort_dylib`].

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::Parser;

use crate::causallm::{CausalLm, Step};
use crate::chat_template::ChatTemplate;

/// The generation graph, as `hf2mobile.postprocess` names it (`CAUSALLM_INFERENCE_GRAPH`).
const INFERENCE_GRAPH: &str = "inference.onnx";
/// The tokenizer the Rust runtime reads (`TOKENIZER_FILE`).
const TOKENIZER: &str = "tokenizer.json";

/// The environment variable `ort`'s `load-dynamic` feature reads to find ONNX Runtime.
const ORT_DYLIB_ENV: &str = "ORT_DYLIB_PATH";

/// The prefix every ONNX Runtime shared library starts with, on every platform we ship to
/// (`libonnxruntime.so`, `libonnxruntime.so.1.22.0`, `libonnxruntime.dylib`).
const ORT_DYLIB_PREFIX: &str = "libonnxruntime.";

#[derive(Parser)]
#[command(
    name = "hf2mobile-infer",
    about = "Run an exported hf2mobile ONNX graph — the CLI mirror of `python -m hf2mobile.infer`.",
    long_about = "Run an exported hf2mobile ONNX graph — the CLI mirror of `python -m hf2mobile.infer`.\n\n\
                  EXPORT_DIR is a directory written by `hf2mobile-export`, holding inference.onnx, \
                  tokenizer.json and tokenizer_config.json.\n\n\
                  How to decode is not configurable here: the sampling policy and the stop tokens were \
                  baked into the graph by `hf2mobile.postprocess` and travel with the model."
)]
pub struct Args {
    /// An export directory produced by `hf2mobile-export` (must hold inference.onnx).
    pub export_dir: PathBuf,

    /// User prompt.
    #[arg(long)]
    pub prompt: String,

    /// Feed --prompt to the model verbatim, instead of through the tokenizer's chat template.
    ///
    /// Spelled out rather than left to clap, which would kebab-case the field into
    /// `--skip-template`: `hf2mobile.infer` takes `--skip_template`, and a device run should be
    /// the host command with a different binary in front of it. The kebab spelling is accepted
    /// too, for whoever guesses it from the other two flags.
    #[arg(long = "skip_template", alias = "skip-template")]
    pub skip_template: bool,

    /// Maximum tokens to generate.
    #[arg(long = "num-generation", default_value_t = 512, value_name = "N")]
    pub num_generation: i64,

    /// ORT intra-op thread count [default: one per core].
    #[arg(long = "intra-threads", value_name = "N")]
    pub intra_threads: Option<usize>,

    /// Path to libonnxruntime.so [default: $ORT_DYLIB_PATH, else beside this binary or the export].
    #[arg(long = "ort-dylib", value_name = "PATH")]
    pub ort_dylib: Option<PathBuf>,
}

/// Everything the binary does. `main` only reports what this returns.
pub fn run(args: Args) -> Result<()> {
    let graph = args.export_dir.join(INFERENCE_GRAPH);
    let tokenizer = args.export_dir.join(TOKENIZER);
    if !graph.is_file() {
        bail!(
            "`{}` holds no {INFERENCE_GRAPH}. Build the graph first:\n\n    \
             hf2mobile-export <hf repo id>\n\n\
             or, for an export you already have:\n\n    \
             python -m hf2mobile.postprocess {}\n",
            args.export_dir.display(),
            args.export_dir.display()
        );
    }
    if !tokenizer.is_file() {
        bail!(
            "`{}` is not a complete export: {TOKENIZER} is missing",
            args.export_dir.display()
        );
    }

    let dylib = resolve_ort_dylib(&args)?;
    log(format!("ONNX Runtime: `{}`", dylib.display()));
    // Read by `ort` when it first opens a session, i.e. inside `CausalLm::open` below. Setting
    // it here rather than asking the user to export it is what makes the adb one-liner work.
    std::env::set_var(ORT_DYLIB_ENV, &dylib);

    let prompt = build_prompt(&args)?;

    let graph = path_str(&graph)?;
    log(format!("constructing CausalLm with `{graph}`"));
    let mut lm = CausalLm::open(graph, path_str(&tokenizer)?, args.intra_threads)?;

    // `num_kv_slots` comes from the cache the runtime actually built, so it is the number of
    // tensors that will be fed back per token rather than a guess made from the input names.
    let eos = if lm.eos_tokens().is_empty() {
        "none — bounded by --num-generation".to_string()
    } else {
        format!("{:?}", lm.eos_tokens())
    };
    log(format!(
        "loaded {} inputs / {} outputs ({} KV cache slots) | eos: {eos}",
        lm.session().inputs.len(),
        lm.session().outputs.len(),
        lm.kv_slots()
    ));
    let threads = match args.intra_threads {
        Some(n) => format!("{n} intra-op threads"),
        None => "one thread per core".to_string(),
    };
    log(format!(
        "generating up to {} tokens on {threads} — streaming to stdout",
        args.num_generation
    ));

    let budget = args.num_generation.max(0) as usize;
    let step: Step = lm.generate(&prompt, budget, true)?;

    if step.text.is_none() {
        println!(); // the streamed line is still open; `generate` only closes it on EOS
    }
    let ending = match step.text {
        Some(_) => "EOS".to_string(),
        None => format!("budget of {} exhausted, EOS not reached", args.num_generation),
    };
    log(format!(
        "{} prompt tokens, {} generated ({ending}) | TTFT {:.1} ms | {:.2} tok/s",
        lm.prefill_len(),
        lm.token_ids().len(),
        step.ttft_s * 1000.0,
        step.tps
    ));
    Ok(())
}

/// The prompt as the model expects to receive it.
///
/// Templated by default, because an instruct model handed a bare prompt tends to continue it
/// rather than answer it. A model that ships no template, or a template this renderer chokes
/// on, warns and falls through to the raw prompt — the same call is still worth making, and
/// `--skip_template` is the way to ask for that on purpose.
fn build_prompt(args: &Args) -> Result<String> {
    if args.skip_template {
        return Ok(args.prompt.clone());
    }
    match ChatTemplate::load(&args.export_dir)? {
        Some(template) => match template.render(&args.prompt) {
            Ok(prompt) => Ok(prompt),
            Err(err) => {
                warn(format!(
                    "could not apply the chat template ({err:#}); using the prompt as-is"
                ));
                Ok(args.prompt.clone())
            }
        },
        None => {
            warn("tokenizer carries no chat template; using the prompt as-is".to_string());
            Ok(args.prompt.clone())
        }
    }
}

/// Where to dlopen ONNX Runtime from.
///
/// In order: `--ort-dylib`, then `$ORT_DYLIB_PATH` (so an existing environment keeps working),
/// then a `libonnxruntime.*` sitting next to this binary, then one in the export directory.
/// The last two are what make a `adb push` of three files enough — there is no system library
/// path on a phone to install into.
fn resolve_ort_dylib(args: &Args) -> Result<PathBuf> {
    if let Some(path) = &args.ort_dylib {
        if !path.is_file() {
            bail!("--ort-dylib `{}` does not exist", path.display());
        }
        return Ok(path.clone());
    }
    if let Some(path) = std::env::var_os(ORT_DYLIB_ENV).filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(path));
    }

    let beside_exe = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf));
    let searched: Vec<PathBuf> = beside_exe.into_iter().chain([args.export_dir.clone()]).collect();
    for dir in &searched {
        if let Some(found) = find_ort_dylib(dir) {
            return Ok(found);
        }
    }
    bail!(
        "no ONNX Runtime library found. Looked for `{ORT_DYLIB_PREFIX}*` in:\n{}\n\n\
         Pass one with --ort-dylib, or set {ORT_DYLIB_ENV}. On Android, take \
         `jni/arm64-v8a/libonnxruntime.so` out of the `onnxruntime-android` AAR and push it \
         next to this binary.",
        searched
            .iter()
            .map(|dir| format!("  {}", dir.display()))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// The first `libonnxruntime.*` in `dir`, if it has one.
///
/// Matched by prefix because the file is versioned as often as not (`libonnxruntime.so.1.22.0`),
/// and `dlopen` is happy to be handed any of those spellings.
fn find_ort_dylib(dir: &Path) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(ORT_DYLIB_PREFIX))
        })
        .collect();
    // Sorted so the choice is the same on every run — `read_dir` order is the filesystem's,
    // not alphabetical, and a directory holding both `libonnxruntime.so` and a versioned
    // symlink to it should not pick a different one each time.
    candidates.sort();
    candidates.into_iter().next()
}

/// A path as the `&str` the runtime's constructors take.
///
/// Fails rather than lossily converting: a path that is not UTF-8 would otherwise be opened
/// under a name that is not the one on disk, and the resulting "file not found" would name a
/// file that looks correct.
fn path_str(path: &Path) -> Result<&str> {
    path.to_str()
        .with_context(|| format!("path is not valid UTF-8: `{}`", path.display()))
}

/// The log line shape `hf2mobile.infer` prints, so a device run and a host run read alike.
fn log(message: String) {
    eprintln!("[hf2mobile] INFO     | {message}");
}

fn warn(message: String) {
    eprintln!("[hf2mobile] WARNING  | {message}");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(export_dir: &Path) -> Args {
        Args {
            export_dir: export_dir.to_path_buf(),
            prompt: "hi".into(),
            skip_template: false,
            num_generation: 8,
            intra_threads: None,
            ort_dylib: None,
        }
    }

    #[test]
    fn ort_dylib_is_found_in_the_export_directory() {
        let dir = std::env::temp_dir().join("hf2mobile-cli-test-dylib");
        std::fs::create_dir_all(&dir).unwrap();
        let lib = dir.join("libonnxruntime.so.1.22.0");
        std::fs::write(&lib, b"").unwrap();

        // The environment is process-wide and the other tests do not read it; clearing it here
        // keeps a developer's own `ORT_DYLIB_PATH` from deciding the result.
        std::env::remove_var(ORT_DYLIB_ENV);
        assert_eq!(resolve_ort_dylib(&args(&dir)).unwrap(), lib);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_missing_ort_dylib_names_the_directories_it_looked_in() {
        let dir = std::env::temp_dir().join("hf2mobile-cli-test-empty");
        std::fs::create_dir_all(&dir).unwrap();

        std::env::remove_var(ORT_DYLIB_ENV);
        let err = format!("{:#}", resolve_ort_dylib(&args(&dir)).unwrap_err());
        assert!(err.contains(&dir.display().to_string()), "{err}");

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn skip_template_leaves_the_prompt_alone() {
        let mut args = args(Path::new("does-not-exist"));
        args.skip_template = true;
        assert_eq!(build_prompt(&args).unwrap(), "hi");
    }

    #[test]
    fn no_tokenizer_config_falls_back_to_the_raw_prompt() {
        assert_eq!(build_prompt(&args(Path::new("does-not-exist"))).unwrap(), "hi");
    }
}
