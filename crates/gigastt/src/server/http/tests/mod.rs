//! Unit tests for HTTP handlers (no model required).

pub(super) use super::super::config::{RuntimeLimits, pool_retry_after_ms, pool_retry_after_secs};
pub(super) use super::super::metrics::MetricsRegistry;
pub(super) use super::admin::peer_is_loopback;
pub(super) use super::error::{
    api_error, api_inference_timeout_error, api_pool_closed_error, api_timeout_error,
};
pub(super) use super::export::render_export_response;
pub(super) use super::stream::{StreamError, sse_data_payload};
pub(super) use super::transcribe::{raw_codec_to_wav, resolve_raw_codec};
pub(super) use super::*;

pub(super) use arc_swap::ArcSwap;
pub(super) use axum::body::Bytes;
pub(super) use axum::extract::{Query, State};
pub(super) use axum::http::{StatusCode, header};
pub(super) use gigastt_core::inference::Engine;
pub(super) use std::sync::Arc;

pub(super) fn sample_export_result() -> gigastt_core::inference::TranscribeResult {
    use gigastt_core::inference::WordInfo;
    gigastt_core::inference::TranscribeResult {
        text: "привет мир".into(),
        words: vec![
            WordInfo::new("привет", 0.0, 0.5, 0.98, Some(0)),
            WordInfo::new("мир", 0.6, 1.0, 0.97, Some(0)),
        ],
        duration_s: 1.0,
        confidence: None,
    }
}

pub(super) fn test_engine() -> Arc<Engine> {
    use std::sync::OnceLock;
    static ENGINE: OnceLock<Arc<Engine>> = OnceLock::new();
    ENGINE
        .get_or_init(|| {
            let tmp = tempfile::tempdir().expect("tempdir");
            gigastt_core::test_support::write_rnnt_layout(tmp.path()).expect("layout");
            Arc::new(
                gigastt_core::test_support::load_rnnt_engine(tmp.path(), 1).expect("mock engine"),
            )
        })
        .clone()
}

pub(super) fn fresh_engine() -> Arc<Engine> {
    let tmp = tempfile::tempdir().expect("tempdir");
    gigastt_core::test_support::write_rnnt_layout(tmp.path()).expect("layout");
    Arc::new(gigastt_core::test_support::load_rnnt_engine(tmp.path(), 1).expect("mock engine"))
}

/// Wrap an engine handle in the `ArcSwap` the `AppState` now holds. Keeps
/// the model-gated test constructors terse after the hot-reload swap change.
pub(super) fn engine_swap(engine: Arc<Engine>) -> Arc<ArcSwap<Engine>> {
    Arc::new(ArcSwap::new(engine))
}

pub(super) fn minimal_wav() -> Bytes {
    let data_size = 4u32;
    let file_size = 44 + data_size - 8;
    let mut wav = vec![];
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&file_size.to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&16000u32.to_le_bytes());
    wav.extend_from_slice(&(16000u32 * 2).to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_size.to_le_bytes());
    wav.extend_from_slice(&0i16.to_le_bytes());
    wav.extend_from_slice(&0i16.to_le_bytes());
    Bytes::from(wav)
}

pub(super) fn short_wav() -> Bytes {
    let sample_rate = 16000u32;
    let duration_s = 0.1f32;
    let num_samples = (sample_rate as f32 * duration_s) as u32;
    let data_size = num_samples * 2;
    let file_size = 44 + data_size - 8;
    let mut wav = vec![];
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&file_size.to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_size.to_le_bytes());
    for _ in 0..num_samples {
        wav.extend_from_slice(&0i16.to_le_bytes());
    }
    Bytes::from(wav)
}

mod admin;
mod codec;
mod error;
mod export;
mod handlers;
mod jobs;
mod serde_contract;
mod sse;
