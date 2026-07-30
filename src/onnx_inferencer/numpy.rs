//! numpy array <-> ORT tensor, for `CausalLMInferencer.run` — the raw single-pass escape
//! hatch, not `generate`.
//!
//! The only file that speaks both the numpy and the ort tensor APIs, so when the pinned
//! `ort` release candidate moves, this is where it hurts and nowhere else.
//!
//! Both directions **copy**. Handing ORT a pointer into a numpy buffer would mean proving
//! to the compiler that Python cannot free or resize that buffer mid-run, and the payoff
//! is not worth the unsafe code: `run` exists for poking at a graph one pass at a time,
//! not for throughput. `generate` never comes through here at all — it keeps tensors on
//! the ORT side from step to step (see [`crate::kv_cache`]).

use half::f16;
use numpy::{dtype_bound, PyArrayDescrMethods, PyArrayDyn, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods};
use ort::tensor::TensorElementType;
use ort::value::{DynValue, Tensor};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Every dtype that can cross the boundary, as `rust type => ORT element type`.
///
/// One list, used by both directions below, so they can never drift apart. Each `$to`
/// macro is handed the list and expands it into its own shape — a repetition macro
/// calling a callback macro, which is Rust's usual answer to "generate the same match
/// arms twice".
///
/// bf16 is deliberately absent: numpy has no built-in bfloat16 dtype, so there is
/// nothing on the Python side to map it to. bf16 models reach the runtime through
/// [`crate::CausalLMInferencer`], which never converts to numpy.
macro_rules! with_dtypes {
    ($to:ident) => {
        $to! {
            f32 => Float32,
            f64 => Float64,
            f16 => Float16,
            i64 => Int64,
            i32 => Int32,
            u8  => Uint8,
            i8  => Int8,
        }
    };
}

/// Copy a numpy array into an ORT tensor.
pub fn to_ort(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<DynValue> {
    let array: &Bound<'_, PyUntypedArray> = obj
        .downcast()
        .map_err(|_| PyValueError::new_err("expected a numpy ndarray"))?;
    let dtype = array.dtype();

    macro_rules! convert {
        ($($rust:ty => $_ort:ident,)*) => {$(
            if dtype.is_equiv_to(&dtype_bound::<$rust>(py)) {
                // The downcast re-reads the array at a concrete element type, which is
                // what lets us name `$rust` in the body below.
                let typed: &Bound<'_, PyArrayDyn<$rust>> = obj.downcast()?;
                let readonly = typed.readonly();
                // `to_owned` copies into a fresh contiguous array, which is both what
                // ORT requires and what flattens a transposed or strided numpy view.
                // Going through an `ndarray` rather than a `(shape, data)` pair also
                // keeps dimensions of length 0 legal — an empty KV cache is `[1, H, 0, E]`.
                let owned = readonly.as_array().to_owned();
                return Ok(Tensor::<$rust>::from_array(owned).map_err(to_py_err)?.into_dyn());
            }
        )*};
    }
    with_dtypes!(convert);

    Err(PyValueError::new_err(format!("unsupported numpy dtype {dtype:?}")))
}

/// Copy an ORT tensor into a fresh numpy array.
pub fn from_ort(py: Python<'_>, value: &DynValue) -> PyResult<PyObject> {
    let (_, ty) = crate::session::tensor_type(value.dtype())
        .ok_or_else(|| PyValueError::new_err(format!("output is not a tensor: {:?}", value.dtype())))?;

    macro_rules! convert {
        ($($rust:ty => $ort:ident,)*) => {
            match ty {
                $(TensorElementType::$ort => {
                    let view = value.try_extract_array::<$rust>().map_err(to_py_err)?;
                    Ok(PyArrayDyn::<$rust>::from_array_bound(py, &view).into_any().unbind())
                })*
                other => Err(PyValueError::new_err(format!("unsupported output dtype {other:?}"))),
            }
        };
    }
    with_dtypes!(convert)
}

fn to_py_err(err: ort::Error) -> PyErr {
    PyValueError::new_err(err.to_string())
}
