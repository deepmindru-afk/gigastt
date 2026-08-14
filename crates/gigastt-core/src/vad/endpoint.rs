//! Streaming endpoint detector and trailing-silence hangover.

use anyhow::Result;

use super::{SileroVad, VAD_FRAME_SAMPLES, VAD_STATE_LEN, VadConfig};

/// Streaming endpoint detector: feeds streamed audio through the VAD in fixed
/// frames, tracks trailing silence, and reports when an utterance has ended
/// (≥ `min_silence_ms` of silence *after* speech was seen).
///
/// Owns its recurrent state and a small leftover buffer so callers can push
/// arbitrary chunk sizes. The threshold/silence logic is exercised directly in
/// tests via [`Hangover`].
pub struct VadEndpointer {
    state: [f32; VAD_STATE_LEN],
    leftover: Vec<f32>,
    hangover: Hangover,
}

impl VadEndpointer {
    /// New endpointer for the given config.
    pub fn new(cfg: &VadConfig) -> Self {
        Self {
            state: [0.0f32; VAD_STATE_LEN],
            leftover: Vec::with_capacity(VAD_FRAME_SAMPLES),
            hangover: Hangover::new(cfg),
        }
    }

    /// Feed a chunk of 16 kHz samples. Returns `true` exactly once per utterance
    /// when trailing silence first crosses `min_silence_ms` after speech — the
    /// caller should finalize the current segment. Resets internally so the next
    /// speech run can trigger again.
    ///
    /// On model inference failure the chunk is treated as non-endpointing (logged
    /// by the caller) so streaming is never blocked by VAD.
    pub fn push(&mut self, vad: &SileroVad, samples: &[f32]) -> Result<bool> {
        self.leftover.extend_from_slice(samples);
        let mut endpoint = false;
        let mut off = 0;
        while off + VAD_FRAME_SAMPLES <= self.leftover.len() {
            let prob = vad.run_frame(
                &self.leftover[off..off + VAD_FRAME_SAMPLES],
                &mut self.state,
            )?;
            off += VAD_FRAME_SAMPLES;
            if self.hangover.update(prob, VAD_FRAME_SAMPLES) {
                endpoint = true;
            }
        }
        // Retain only the unprocessed tail.
        if off > 0 {
            self.leftover.drain(..off);
        }
        Ok(endpoint)
    }
}

/// Pure trailing-silence state machine shared by the streaming endpointer.
///
/// `update` is fed one frame's probability at a time and returns `true` on the
/// single frame where trailing silence first reaches `min_silence_ms` after
/// speech has been observed. After firing it disarms until speech resumes, so
/// one utterance yields exactly one endpoint.
#[derive(Debug)]
pub struct Hangover {
    threshold: f32,
    min_silence_samples: usize,
    seen_speech: bool,
    trailing_silence: usize,
    armed: bool,
}

impl Hangover {
    pub(crate) fn new(cfg: &VadConfig) -> Self {
        Self {
            threshold: cfg.threshold,
            min_silence_samples: VadConfig::ms_to_samples(cfg.min_silence_ms),
            seen_speech: false,
            trailing_silence: 0,
            armed: false,
        }
    }

    /// Advance by one frame of `frame_samples` samples with speech probability
    /// `prob`. Returns `true` on the endpoint-crossing frame. Thresholds are
    /// fixed at construction ([`Hangover::new`]).
    pub(crate) fn update(&mut self, prob: f32, frame_samples: usize) -> bool {
        if prob >= self.threshold {
            self.seen_speech = true;
            self.armed = true;
            self.trailing_silence = 0;
            return false;
        }
        if !self.seen_speech {
            return false;
        }
        self.trailing_silence += frame_samples;
        if self.armed && self.trailing_silence >= self.min_silence_samples {
            self.armed = false; // fire once until speech resumes
            return true;
        }
        false
    }
}
