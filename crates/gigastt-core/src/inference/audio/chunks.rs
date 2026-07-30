//! Fixed-size streaming decode of a container to 16 kHz mono.
//!
//! [`FileWindows`] yields the *overlapping* windows the long-form file decode
//! wants. Callers that drive the **streaming** recognizer instead — SSE file
//! transcription, and any embedder feeding `Engine::process_chunk` — want plain
//! consecutive chunks of a fixed length, exactly what `slice.chunks(n)` gives
//! over a fully decoded buffer, without the fully decoded buffer.
//!
//! That is this type. It holds one pull block plus at most one chunk, so peak
//! audio memory is O(chunk) rather than O(file), and it emits the same sample
//! sequence in the same chunk boundaries `chunks(n)` would.

use anyhow::Result;
use bytes::Bytes;

use super::stream::{FileWindows, PcmWindows, WindowSpec};

/// Samples pulled from the container per step, independent of the emitted chunk
/// size. Frame-aligned so [`WindowSpec`] keeps stride == window.
const PULL_SAMPLES: usize = 32_000; // 2 s @16 kHz

/// Consecutive fixed-size chunks of 16 kHz mono PCM, decoded on demand.
///
/// Yields `chunk_samples` per call until the stream ends; the final chunk is the
/// remainder and may be shorter — the same shape as `slice.chunks(n)`. An empty
/// stream yields nothing.
pub struct AudioChunks {
    src: FileWindows,
    /// Decoded-but-not-yet-emitted samples.
    pending: Vec<f32>,
    /// Emission buffer, reused across calls so a chunk costs no allocation.
    chunk: Vec<f32>,
    chunk_samples: usize,
    eof: bool,
}

impl AudioChunks {
    /// Open `data` for chunked decode. The container is probed here, so an
    /// unsupported or malformed **header** fails now rather than mid-stream;
    /// packet decoding is lazy.
    ///
    /// `max_audio_secs` is honoured verbatim — `None` means unbounded, since
    /// peak memory does not scale with length. `chunk_samples` is clamped to at
    /// least 1.
    ///
    /// # Errors
    ///
    /// Returns an error if the container cannot be probed or has no audio track.
    pub fn from_bytes(
        data: Bytes,
        chunk_samples: usize,
        max_audio_secs: Option<f64>,
    ) -> Result<Self> {
        let chunk_samples = chunk_samples.max(1);
        Ok(Self {
            src: FileWindows::from_bytes(
                data,
                WindowSpec::new(0, PULL_SAMPLES, 0),
                max_audio_secs,
            )?,
            pending: Vec::with_capacity(chunk_samples + PULL_SAMPLES),
            chunk: Vec::with_capacity(chunk_samples),
            chunk_samples,
            eof: false,
        })
    }

    /// Total 16 kHz samples decoded so far. Exact once the stream is drained.
    pub fn total_16k_samples(&self) -> usize {
        self.src.total_16k_samples()
    }

    /// Lend the next chunk, or `Ok(None)` once the stream is exhausted.
    ///
    /// # Errors
    ///
    /// Returns an error if a packet fails to decode or the length budget trips.
    pub fn next_chunk(&mut self) -> Result<Option<&[f32]>> {
        while !self.eof && self.pending.len() < self.chunk_samples {
            match self.src.next_window().map_err(anyhow::Error::from)? {
                Some(w) => self.pending.extend_from_slice(w.samples),
                None => self.eof = true,
            }
        }
        if self.pending.is_empty() {
            return Ok(None);
        }
        let take = self.chunk_samples.min(self.pending.len());
        self.chunk.clear();
        self.chunk.extend_from_slice(&self.pending[..take]);
        self.pending.drain(..take);
        Ok(Some(&self.chunk))
    }
}
