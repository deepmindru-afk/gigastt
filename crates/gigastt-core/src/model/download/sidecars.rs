//! Optional sidecar model downloads (speaker, punctuation, VAD).

use anyhow::{Context, Result};
use std::path::Path;

use super::super::variant::{PUNCT_FILES, PUNCT_HF_REPO, VAD_MODEL_SHA256, VAD_MODEL_URL};
#[cfg(feature = "diarization")]
use super::super::variant::{SPEAKER_HF_REPO, SPEAKER_MODEL_FILE, SPEAKER_MODEL_SHA256};

#[cfg(unix)]
use super::acquire_download_lock;
use super::fetch::stream_to_partial_then_finalize;

/// Ensure the speaker diarization model exists in `model_dir`, downloading from HuggingFace if missing.
///
/// Downloads `wespeaker_resnet34.onnx` from `onnx-community/wespeaker-voxceleb-resnet34-LM`
/// into `<model_dir>/wespeaker_resnet34.onnx.partial`, verifies its SHA-256 against
/// `SPEAKER_MODEL_SHA256`, and atomically renames it into place. On checksum mismatch or
/// crash the final path is never observable, so a subsequent `ensure_speaker_model` call
/// will re-download from scratch rather than loading a tampered model.
#[cfg(feature = "diarization")]
#[cfg(feature = "net")]
pub async fn ensure_speaker_model(model_dir: &str) -> Result<()> {
    let dir = Path::new(model_dir);
    let final_dest = dir.join(SPEAKER_MODEL_FILE);

    if final_dest.exists() {
        tracing::info!("Speaker model found at {}", final_dest.display());
        return Ok(());
    }

    tracing::info!("Speaker model not found, downloading from HuggingFace...");
    std::fs::create_dir_all(dir).context("Failed to create model directory")?;

    let url = format!("https://huggingface.co/{SPEAKER_HF_REPO}/resolve/main/onnx/model.onnx");
    stream_to_partial_then_finalize(
        &url,
        &final_dest,
        Some(SPEAKER_MODEL_SHA256),
        SPEAKER_MODEL_FILE,
    )
    .await
}

/// Ensure the optional punctuation model exists in `punct_model_dir`,
/// downloading any missing files from the `ekhodzitsky/rupunct-small-onnx`
/// HuggingFace repo (public, MIT).
///
/// Downloads the three files the punctuation pass needs
/// (`rupunct_small_int8.onnx`, `tokenizer.json`, `config.json`) — only those
/// not already present — using the same streaming-download + atomic-rename +
/// SHA-256 infra as the main model download. Files already on disk are left
/// untouched, so a second call is a no-op (no re-download).
///
/// The pass is strictly optional: callers treat a download error as
/// "punctuation unavailable" and proceed with bare text.
#[cfg(feature = "net")]
pub async fn ensure_punct_model(punct_model_dir: &str) -> Result<()> {
    let dir = Path::new(punct_model_dir);

    if PUNCT_FILES.iter().all(|(file, _)| dir.join(file).exists()) {
        tracing::info!("Punctuation model found at {punct_model_dir}");
        return Ok(());
    }

    tracing::info!("Punctuation model not found, downloading from HuggingFace...");
    std::fs::create_dir_all(dir).context("Failed to create punctuation model directory")?;

    #[cfg(unix)]
    let _lock = acquire_download_lock(dir)?;

    for (file, sha256) in PUNCT_FILES {
        let final_dest = dir.join(file);
        if final_dest.exists() {
            continue;
        }
        let url = format!("https://huggingface.co/{PUNCT_HF_REPO}/resolve/main/{file}");
        stream_to_partial_then_finalize(&url, &final_dest, Some(sha256), file).await?;
    }

    tracing::info!("Punctuation model download complete");
    Ok(())
}

/// Ensure the optional Silero VAD model exists in `vad_model_dir`, downloading
/// it from the pinned Silero release (MIT) if missing.
///
/// Uses the same streaming-download + atomic-rename + SHA-256 infra as the main
/// model download. A file already on disk is left untouched (no re-download).
///
/// VAD is strictly optional: callers treat a download error as "VAD
/// unavailable" and proceed without silence skipping / VAD endpointing.
#[cfg(feature = "net")]
pub async fn ensure_vad_model(vad_model_dir: &str) -> Result<()> {
    let dir = Path::new(vad_model_dir);
    let final_dest = dir.join(crate::vad::VAD_MODEL_FILE);

    if final_dest.exists() {
        tracing::info!("VAD model found at {}", final_dest.display());
        return Ok(());
    }

    tracing::info!("VAD model not found, downloading from {VAD_MODEL_URL}...");
    std::fs::create_dir_all(dir).context("Failed to create VAD model directory")?;

    #[cfg(unix)]
    let _lock = acquire_download_lock(dir)?;

    // Another process may have finished while we waited for the lock.
    if final_dest.exists() {
        return Ok(());
    }

    stream_to_partial_then_finalize(
        VAD_MODEL_URL,
        &final_dest,
        Some(VAD_MODEL_SHA256),
        crate::vad::VAD_MODEL_FILE,
    )
    .await?;

    tracing::info!("VAD model download complete");
    Ok(())
}
