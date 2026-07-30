//! `libhf2mobile_plugins.so` — ONNX Runtime custom operators for exported hf2mobile graphs.
//!
//! One operator so far: [`sample_logits`], which turns a row of logits into a token id
//! inside the graph. `python -m hf2mobile.postprocess <export dir>` is what puts the node
//! in the graph; this library is what lets a runtime execute it.
//!
//! # Loading it
//!
//! ```console
//! $ cargo build --release --manifest-path src/onnx_plugins/Cargo.toml
//! ```
//!
//! ```python
//! import onnxruntime as ort
//! opts = ort.SessionOptions()
//! opts.register_custom_ops_library("src/onnx_plugins/target/release/libhf2mobile_plugins.so")
//! session = ort.InferenceSession("case2.onnx", opts)
//! ```
//!
//! (Its own `target/`, because it is its own workspace — see `Cargo.toml`.)
//!
//! Any ONNX Runtime binding can load it — it exports the plain C entry point
//! [`RegisterCustomOps`] that ORT's own `RegisterCustomOpsLibrary` calls, so the same `.so`
//! works from Python, C++, or the Rust runtime in `src/onnx_inferencer`.
//!
//! # How this differs from the rest of the repo
//!
//! `src/onnx_inferencer` *drives* ONNX Runtime; this crate is *driven by* it. So it never
//! opens the onnxruntime dylib: ORT calls [`RegisterCustomOps`] with an `OrtApiBase` and
//! that becomes the API `ort` uses, via [`ort::set_api`]. Hence the `alternative-backend`
//! feature in `Cargo.toml`, and hence this being a separate crate — see the comment there.

use std::ffi::{CStr, CString};
use std::ptr;

use ort::operator::OperatorDomain;
use ort::sys as ort_sys;
use ort::AsPointer;

use crate::sample_logits::{SampleLogits, DOMAIN};

mod sample_logits;

// The sampling policy itself is shared with the Rust runtime rather than reimplemented:
// same file, one definition of what top-k/top-p/temperature mean, so the plugin and
// `hf2mobile.quantize` cannot drift into picking tokens differently. Outside this package
// directory, which `cargo build` is fine with (this crate is never published).
#[path = "../onnx_inferencer/sampling.rs"]
mod sampling;

/// The ONNX Runtime C API version this was built against — `ort` rc.10 targets onnxruntime
/// 1.22, i.e. API version 22.
const BUILT_AGAINST: u32 = ort_sys::ORT_API_VERSION;

const VERSION_MISMATCH: &CStr = c"libhf2mobile_plugins was built against the ONNX Runtime 1.22 C API, \
     which this runtime is too old to provide. Upgrade onnxruntime to 1.22 or newer.";

/// ONNX Runtime's entry point into a custom-op library.
///
/// Called once per `RegisterCustomOpsLibrary` (i.e. per `SessionOptions`), before any
/// session is built from those options. Returning a non-null `OrtStatus` fails that call
/// with the message inside it; returning null means the domain was registered.
///
/// # Safety
///
/// Called by ONNX Runtime with a live `OrtSessionOptions` and `OrtApiBase`. Not meant to be
/// called by hand.
#[no_mangle]
pub unsafe extern "system" fn RegisterCustomOps(
    options: *mut ort_sys::OrtSessionOptions,
    api_base: *const ort_sys::OrtApiBase,
) -> ort_sys::OrtStatusPtr {
    if api_base.is_null() {
        // No API means no way to build an `OrtStatus` to complain with, and no ORT that
        // could read one. Nothing to do but decline quietly.
        return ort_sys::OrtStatusPtr(ptr::null_mut());
    }

    let get_api = (*api_base).GetApi;
    let api = get_api(BUILT_AGAINST);
    if api.is_null() {
        // The host runtime is older than the API this was compiled against, so its `OrtApi`
        // is missing fields `ort` would read. Ask for older versions purely to get hold of
        // `CreateStatus` — it is the first field of `OrtApi` in every version, so a struct
        // from an older ORT is still enough to hand back a legible error.
        return match (1..BUILT_AGAINST)
            .rev()
            .map(|version| get_api(version))
            .find(|api| !api.is_null())
        {
            Some(api) => ((*api).CreateStatus)(ort_sys::OrtErrorCode::ORT_INVALID_ARGUMENT, VERSION_MISMATCH.as_ptr()),
            None => ort_sys::OrtStatusPtr(ptr::null_mut()),
        };
    }

    // Copies the API table into `ort`'s global slot. A no-op on the second and later calls,
    // which is the normal case: one process, several sessions.
    ort::set_api(*api);
    into_status(register(options))
}

/// Build the operator domain and hand it to `options`.
fn register(options: *mut ort_sys::OrtSessionOptions) -> ort::Result<()> {
    // fp32 logits only, for now. ORT picks a kernel by input type, so a graph exported in
    // another float type needs a second registration under the same op name — see
    // [`sample_logits::LogitElement`].
    let domain = OperatorDomain::new(DOMAIN)?.add(SampleLogits::<f32>::new())?;

    // ORT stores the pointer and dereferences it while building every session from these
    // options, without taking ownership — so the domain has to outlive this call. Leaking
    // is the simplest way to say "forever", and costs a few hundred bytes per
    // `register_custom_ops_library` call.
    let domain: &'static mut OperatorDomain = Box::leak(Box::new(domain));

    // The one place we reach for the raw API: `ort`'s safe wrapper for this lives on its own
    // `SessionOptions` builder, and what we have here is ORT's.
    let status = unsafe { (ort::api().AddCustomOpDomain)(options, domain.ptr_mut()) };
    unsafe { ort::error::status_to_result(status) }
}

/// Turn a Rust result into what ORT's C API expects: null for success, an owned `OrtStatus`
/// carrying the message for failure.
fn into_status(result: ort::Result<()>) -> ort_sys::OrtStatusPtr {
    match result {
        Ok(()) => ort_sys::OrtStatusPtr(ptr::null_mut()),
        Err(err) => {
            // A C string cannot hold an interior NUL; dropping them keeps the message
            // rather than losing it to an `expect`.
            let message = CString::new(err.to_string().replace('\0', "")).expect("no interior nul remains");
            unsafe { (ort::api().CreateStatus)(ort_sys::OrtErrorCode::ORT_FAIL, message.as_ptr()) }
        }
    }
}
