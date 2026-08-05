#[cfg(all(
    feature = "candle",
    any(feature = "coreml", feature = "cuda", feature = "nnapi")
))]
compile_error!("feature `candle` is mutually exclusive with `coreml`/`cuda`/`nnapi`");

#[cfg(all(
    feature = "ane",
    any(
        feature = "coreml",
        feature = "cuda",
        feature = "nnapi",
        feature = "candle"
    )
))]
compile_error!("feature `ane` is mutually exclusive with `coreml`/`cuda`/`nnapi`/`candle`");

pub mod error;
pub mod export;
pub mod inference;
pub mod itn;
pub mod lexicon;
pub mod model;
pub mod protocol;
pub mod punctuation;

/// INT8 quantizer, re-exported so `gigastt_core::quantize::quantize_model`
/// keeps working. Behind the default-on `quantize` feature: lean embedders that
/// side-load a pre-quantized model can turn it off and drop `prost` — and the
/// `protoc` build requirement — entirely.
#[cfg(feature = "quantize")]
pub use gigastt_quantize as quantize;
pub(crate) mod runtime;
mod sha256;
pub mod vad;
mod wordpiece;

pub use runtime::cpu_factory;

/// Runtime abstraction surface needed to drive backends directly (e.g. parity
/// tests that construct and compare the ort and candle encoder sessions).
pub mod runtime_api {
    #[cfg(all(feature = "ane", target_os = "macos"))]
    pub use crate::runtime::ane_factory;
    #[cfg(feature = "candle")]
    pub use crate::runtime::candle_factory;
    pub use crate::runtime::{
        Runtime, RuntimeError, RuntimeFactory, RuntimeSession, Shape, Tensor, TensorData,
        TensorDataView, cpu_factory, production_factory,
    };
}
