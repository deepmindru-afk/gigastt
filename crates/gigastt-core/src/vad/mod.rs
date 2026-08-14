//! Voice activity detection (VAD) via the Silero v5 ONNX model.
//!
//! Used for two optional, opt-in features on top of the recognition engine:
//!
//! 1. **File silence skipping** — [`SileroVad::speech_regions`] returns the
//!    speech spans of a clip so the engine can decode only those, skipping long
//!    pauses. Speedup is proportional to the silence fraction.
//! 2. **Streaming endpointing** — [`VadEndpointer`] tracks trailing silence
//!    across streamed chunks and signals when an utterance has ended. When a
//!    VAD is attached it owns endpointing: the decoder's blank-run heuristic
//!    is ignored, so `min_silence_ms` fully controls finalization.
//!
//! The model is loaded through the same `ort` runtime the recognition engine
//! already uses (no extra dependency, no second ONNX Runtime). The Silero v5
//! graph (opset 16, conv + LSTM) takes a fixed 512-sample window at 16 kHz plus
//! a recurrent state tensor `[2, 1, 128]`, and returns a speech probability in
//! `[0, 1]` together with the next state.
//!
//! All of the segmentation / endpointing decision logic is split into pure
//! functions ([`regions_from_probs`], [`Hangover`]) so it can be unit-tested on
//! synthetic probability sequences without loading the model.

/// Silero VAD ONNX filename on disk. Single source of truth shared with the
/// model-download path in [`crate::model`].
pub const VAD_MODEL_FILE: &str = "silero_vad.onnx";

/// Sample rate the engine (and Silero) operate at.
pub const VAD_SAMPLE_RATE: i64 = 16000;

/// Fixed Silero v5 window at 16 kHz (~32 ms). The model only accepts this size.
pub const VAD_FRAME_SAMPLES: usize = 512;

/// Length of the Silero recurrent state tensor (`[2, 1, 128]` flattened).
pub(crate) const VAD_STATE_LEN: usize = 2 * 128;

mod config;
mod endpoint;
mod regions;
#[cfg(feature = "file-decode")]
mod segmenter;
mod silero;

pub use config::VadConfig;
pub use endpoint::{Hangover, VadEndpointer};
pub use regions::{regions_from_probs, remap_compressed_seconds};
#[cfg(feature = "file-decode")]
pub(crate) use segmenter::VadSegmenter;
pub use silero::SileroVad;

#[cfg(test)]
mod tests;
