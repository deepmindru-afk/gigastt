//! Thin public `transcribe_file` / `transcribe_bytes` wrappers.

use super::*;

impl Engine {
    /// Transcribe an audio file to text (WAV, MP3, M4A/AAC, OGG, Opus, WebM, FLAC).
    ///
    /// Decodes the file to mono 16kHz, runs the full encoder+decoder pipeline,
    /// and returns the recognized text with word-level details and duration.
    ///
    /// Thin wrapper over [`Engine::transcribe_request`].
    ///
    /// # Errors
    ///
    /// Returns [`GigasttError::InvalidAudio`] if the file cannot be decoded, or
    /// [`GigasttError::Inference`] if the ONNX runtime fails.
    #[cfg(feature = "file-decode")]
    pub fn transcribe_file(
        &self,
        path: &str,
        triplet: &mut SessionTriplet,
    ) -> Result<TranscribeResult, GigasttError> {
        self.transcribe_request(
            TranscribeRequest::new(TranscribeSource::Path(path)),
            triplet,
        )
    }

    /// Like [`Engine::transcribe_file`] but applies per-request recognition-knob
    /// [`overrides`](TranscribeOverrides). With `TranscribeOverrides::default()`
    /// this is byte-for-byte [`Engine::transcribe_file`]; the plain method
    /// delegates here so binding call sites (FFI / UniFFI / Node) keep the
    /// no-override signature unchanged.
    #[cfg(feature = "file-decode")]
    pub fn transcribe_file_with_overrides(
        &self,
        path: &str,
        triplet: &mut SessionTriplet,
        overrides: &TranscribeOverrides,
    ) -> Result<TranscribeResult, GigasttError> {
        self.transcribe_file_with_overrides_hotwords(path, triplet, overrides, None)
    }

    /// Like [`Engine::transcribe_file_with_overrides`] with optional per-request
    /// [`HotwordOverride`] (semver-additive sibling so the no-hotwords signature
    /// stays byte-stable).
    #[cfg(feature = "file-decode")]
    pub fn transcribe_file_with_overrides_hotwords(
        &self,
        path: &str,
        triplet: &mut SessionTriplet,
        overrides: &TranscribeOverrides,
        hotwords: Option<&HotwordOverride>,
    ) -> Result<TranscribeResult, GigasttError> {
        self.transcribe_request(
            TranscribeRequest::new(TranscribeSource::Path(path))
                .with_overrides(*overrides)
                .with_hotwords(hotwords),
            triplet,
        )
    }

    /// Transcribe audio from raw bytes in memory (no temp file needed).
    ///
    /// Backwards-compatible shim: clones `data` into a [`bytes::Bytes`] and
    /// delegates to [`Engine::transcribe_bytes_shared`]. Prefer the shared
    /// variant on hot paths (REST/SSE) to avoid the extra copy.
    #[cfg(feature = "file-decode")]
    pub fn transcribe_bytes(
        &self,
        data: &[u8],
        triplet: &mut SessionTriplet,
    ) -> Result<TranscribeResult, GigasttError> {
        self.transcribe_bytes_shared(bytes::Bytes::copy_from_slice(data), triplet)
    }

    /// Transcribe audio from a reference-counted [`bytes::Bytes`] buffer
    /// without cloning.
    ///
    /// Reuses the same decode/inference pipeline as [`Engine::transcribe_bytes`]
    /// but hands the buffer straight to symphonia via [`audio::decode_audio_bytes_shared`].
    /// This is the zero-copy entry point used by the REST upload handler so a
    /// 50 MiB `axum::body::Bytes` body stays as a single in-memory buffer
    /// instead of being cloned into a `Vec<u8>` before decode.
    #[cfg(feature = "file-decode")]
    pub fn transcribe_bytes_shared(
        &self,
        data: bytes::Bytes,
        triplet: &mut SessionTriplet,
    ) -> Result<TranscribeResult, GigasttError> {
        self.transcribe_bytes_shared_with_overrides(data, triplet, &TranscribeOverrides::default())
    }

    /// Like [`Engine::transcribe_bytes_shared`] but applies per-request
    /// recognition-knob [`overrides`](TranscribeOverrides). With
    /// `TranscribeOverrides::default()` this is byte-for-byte
    /// [`Engine::transcribe_bytes_shared`]; the plain method delegates here so
    /// the zero-copy REST call site can opt into overrides without changing the
    /// no-override signature that other callers rely on.
    #[cfg(feature = "file-decode")]
    pub fn transcribe_bytes_shared_with_overrides(
        &self,
        data: bytes::Bytes,
        triplet: &mut SessionTriplet,
        overrides: &TranscribeOverrides,
    ) -> Result<TranscribeResult, GigasttError> {
        self.transcribe_bytes_shared_with_overrides_hotwords(data, triplet, overrides, None)
    }

    /// Like [`Engine::transcribe_bytes_shared_with_overrides`] with optional
    /// per-request [`HotwordOverride`].
    #[cfg(feature = "file-decode")]
    pub fn transcribe_bytes_shared_with_overrides_hotwords(
        &self,
        data: bytes::Bytes,
        triplet: &mut SessionTriplet,
        overrides: &TranscribeOverrides,
        hotwords: Option<&HotwordOverride>,
    ) -> Result<TranscribeResult, GigasttError> {
        self.transcribe_request(
            TranscribeRequest::new(TranscribeSource::Bytes(data))
                .with_overrides(*overrides)
                .with_hotwords(hotwords),
            triplet,
        )
    }

    /// Like [`Engine::transcribe_bytes_shared_with_overrides`], but also runs
    /// offline speaker diarization (labels each word's `speaker`) when a speaker
    /// encoder is loaded. Diarization is **opt-in per request** (`?diarization=true`
    /// on the REST surface): the non-diarized transcribe methods never label
    /// speakers, so a plain transcript — and the `channels=split` dual-mono
    /// fallback — carries no speaker labels. Without the `diarization` feature or a
    /// loaded speaker encoder this is byte-for-byte the non-diarized method.
    #[cfg(feature = "file-decode")]
    pub fn transcribe_bytes_shared_with_overrides_diarized(
        &self,
        data: bytes::Bytes,
        triplet: &mut SessionTriplet,
        overrides: &TranscribeOverrides,
    ) -> Result<TranscribeResult, GigasttError> {
        self.transcribe_bytes_shared_with_overrides_diarized_hotwords(
            data, triplet, overrides, None,
        )
    }

    /// Like [`Engine::transcribe_bytes_shared_with_overrides_diarized`] with
    /// optional per-request [`HotwordOverride`].
    #[cfg(feature = "file-decode")]
    pub fn transcribe_bytes_shared_with_overrides_diarized_hotwords(
        &self,
        data: bytes::Bytes,
        triplet: &mut SessionTriplet,
        overrides: &TranscribeOverrides,
        hotwords: Option<&HotwordOverride>,
    ) -> Result<TranscribeResult, GigasttError> {
        self.transcribe_request(
            TranscribeRequest::new(TranscribeSource::Bytes(data))
                .with_overrides(*overrides)
                .with_hotwords(hotwords)
                .with_diarization(true),
            triplet,
        )
    }
}
