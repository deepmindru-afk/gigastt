//! VAD thresholds shared by file-region and streaming-endpoint paths.

use super::VAD_SAMPLE_RATE;

/// Tunable thresholds for turning a per-frame speech-probability sequence into
/// speech spans (file path) and endpoint decisions (streaming).
#[derive(Debug, Clone, Copy)]
pub struct VadConfig {
    /// Speech-probability threshold in `[0, 1]`; frames at or above are speech.
    pub threshold: f32,
    /// Minimum trailing silence before a speech region is closed / an utterance
    /// is considered ended (endpointing).
    pub min_silence_ms: u32,
    /// Speech runs shorter than this are dropped as noise (file path only).
    pub min_speech_ms: u32,
    /// Padding added on each side of a kept speech region so onsets/offsets are
    /// not clipped (file path only).
    pub speech_pad_ms: u32,
}

impl Default for VadConfig {
    fn default() -> Self {
        // Silero's own defaults, lightly adapted: 0.5 threshold, ~500 ms of
        // silence to close a turn, 250 ms minimum speech, 100 ms pad.
        Self {
            threshold: 0.5,
            min_silence_ms: 500,
            min_speech_ms: 250,
            speech_pad_ms: 100,
        }
    }
}

impl VadConfig {
    pub(crate) fn ms_to_samples(ms: u32) -> usize {
        (VAD_SAMPLE_RATE as usize * ms as usize) / 1000
    }
}
