//! The `hf2mobile-infer` executable.
//!
//! Deliberately three lines: everything it does is [`_ortrs_binding::cli`], so the engine, the
//! Python module and this binary all sit in one crate and cannot drift apart. `_ortrs_binding`
//! is the library's name because that is what Python imports it as (see `[lib]` in
//! `Cargo.toml`) — an odd thing to type here, but better than a second name for one crate.

use clap::Parser;

use _ortrs_binding::cli::{self, Args};

fn main() {
    if let Err(err) = cli::run(Args::parse()) {
        // `{:#}` prints the whole `anyhow` context chain on one line — "opening ONNX model
        // `x.onnx`: ..." rather than just the innermost failure.
        eprintln!("[hf2mobile] ERROR    | {err:#}");
        std::process::exit(1);
    }
}
