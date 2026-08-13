//! Model presence, ensure/download, and the streaming SHA-256 fetch.

use crate::error::GigasttError;
use crate::sha256::{Sha256, hex_lower};
#[cfg(feature = "net")]
use anyhow::Context;
use anyhow::Result;
use std::path::Path;

#[cfg(unix)]
#[cfg(feature = "net")]
use std::os::fd::AsRawFd;

use super::variant::ModelVariant;
#[cfg(feature = "net")]
use super::variant::{
    PREQUANT_RELEASE_BASE, PUNCT_FILES, PUNCT_HF_REPO, VAD_MODEL_SHA256, VAD_MODEL_URL,
};

#[cfg(all(feature = "ane", not(feature = "net")))]
use super::variant::ANE_BUCKETS;
#[cfg(all(feature = "net", feature = "ane"))]
use super::variant::{ANE_BUCKETS, ANE_RELEASE_BASE, ANE_TAR_CHECKSUMS};

#[cfg(all(feature = "diarization", feature = "net"))]
use super::variant::{SPEAKER_HF_REPO, SPEAKER_MODEL_FILE, SPEAKER_MODEL_SHA256};

/// Stream a file and return its lowercase SHA-256 hex digest.
///
/// Used at engine load (not just download) so a tampered file in
/// `~/.gigastt/models/` cannot be silently mapped.
pub(crate) fn hash_file_sha256(path: &Path) -> std::io::Result<String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex_lower(&hasher.finalize()))
}

/// Refuse `path` when its digest is not `expected`.
pub(crate) fn verify_pinned_checksum(path: &Path, expected: &str) -> Result<(), GigasttError> {
    let actual = hash_file_sha256(path).map_err(|e| GigasttError::ModelLoad {
        path: path.display().to_string(),
        source: Some(e.into()),
    })?;
    if actual != expected {
        return Err(GigasttError::ModelLoad {
            path: path.display().to_string(),
            source: Some(format!("SHA-256 mismatch: expected {expected}, got {actual}").into()),
        });
    }
    Ok(())
}

pub(super) fn home_dir() -> Option<std::path::PathBuf> {
    #[cfg(unix)]
    {
        std::env::var_os("HOME").map(std::path::PathBuf::from)
    }
    #[cfg(windows)]
    {
        std::env::var_os("USERPROFILE").map(std::path::PathBuf::from)
    }
}

/// Return the default model directory path (`~/.gigastt/models/`).
///
/// Falls back to `.gigastt/models` if the home directory cannot be determined.
pub fn default_model_dir() -> String {
    home_dir()
        .map(|h| {
            h.join(".gigastt")
                .join("models")
                .to_string_lossy()
                .into_owned()
        })
        .unwrap_or_else(|| ".gigastt/models".into())
}

/// Return the default punctuation-model directory (`~/.gigastt/models/punct/`),
/// a sibling of [`default_model_dir`].
///
/// Holds the optional RUPunct ONNX punctuation/casing restorer used to
/// post-process the plain `rnnt` head's bare lowercase output. The artifact
/// auto-downloads from `ekhodzitsky/rupunct-small-onnx` via
/// [`ensure_punct_model`] when the punct pass is enabled (see
/// [`crate::punctuation`]); a download failure simply disables the punct pass.
pub fn default_punct_model_dir() -> String {
    home_dir()
        .map(|h| {
            h.join(".gigastt")
                .join("models")
                .join("punct")
                .to_string_lossy()
                .into_owned()
        })
        .unwrap_or_else(|| ".gigastt/models/punct".into())
}

/// Return the default VAD-model directory (`~/.gigastt/models/vad/`), a sibling
/// of [`default_model_dir`].
///
/// Holds the optional Silero v5 ONNX voice-activity detector used for file
/// silence skipping and streaming endpointing. The artifact auto-downloads via
/// [`ensure_vad_model`] when VAD is enabled (see [`crate::vad`]); a download
/// failure simply disables VAD.
pub fn default_vad_model_dir() -> String {
    home_dir()
        .map(|h| {
            h.join(".gigastt")
                .join("models")
                .join("vad")
                .to_string_lossy()
                .into_owned()
        })
        .unwrap_or_else(|| ".gigastt/models/vad".into())
}

/// Acquire an advisory exclusive lock on a file inside `dir` so that only
/// one process downloads models at a time. The lock is released when the
/// returned file is dropped.
#[cfg(unix)]
#[cfg(feature = "net")]
pub(super) fn acquire_download_lock(dir: &Path) -> Result<std::fs::File> {
    let lock_path = dir.join(".download.lock");
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&lock_path)
        .context("Failed to create download lock file")?;
    let fd = file.as_raw_fd();
    // SAFETY: `fd` is valid because it comes from `as_raw_fd()` on an owned
    // `File` that outlives this call. `flock` is an advisory lock; the file
    // remains owned by `file` and is closed (releasing the lock) when this
    // function's caller drops the returned `File`.
    let ret = unsafe { libc::flock(fd, libc::LOCK_EX) };
    if ret != 0 {
        anyhow::bail!("Failed to acquire download lock (another process is downloading)");
    }
    Ok(file)
}

/// Decision returned by [`resolve_variant`].
#[derive(Debug, PartialEq, Eq)]
pub enum VariantAction {
    /// Use the variant already present on disk — no download needed.
    Use(ModelVariant),
    /// Download (or re-download) the specified variant.
    Download(ModelVariant),
}

/// Pure decision function: given an optional user-requested variant and the
/// variant already fully present on disk, return what `ensure_model` should do.
///
/// Precedence rules:
/// - **Explicit request + matching install** → `Use` (no-op).
/// - **Explicit request + different/no install** → `Download` the requested variant.
/// - **No request + existing install** → `Use` that install (never clobber it).
/// - **No request + empty dir** → `Download` the default (`Rnnt`).
pub fn resolve_variant(
    requested: Option<ModelVariant>,
    existing: Option<ModelVariant>,
) -> VariantAction {
    match (requested, existing) {
        (Some(req), Some(ex)) if req == ex => VariantAction::Use(req),
        (Some(req), _) => VariantAction::Download(req),
        (None, Some(ex)) => VariantAction::Use(ex),
        (None, None) => VariantAction::Download(ModelVariant::default()),
    }
}

/// Ensure a model is present in `model_dir`, auto-detecting the installed
/// variant and downloading the default (`Rnnt`) only when the directory holds
/// no usable model. Equivalent to `ensure_model_variant(None, model_dir)` with
/// the resolved variant discarded. Preserves the pre-variant public signature.
#[cfg(feature = "net")]
pub async fn ensure_model(model_dir: &str) -> Result<()> {
    ensure_model_variant(None, model_dir).await?;
    Ok(())
}

/// Ensure an appropriate model variant's files exist in `model_dir`,
/// downloading from HuggingFace if missing.
///
/// When `requested` is `Some(v)`, the function enforces that variant `v` is
/// present, downloading it if it isn't (or if the dir holds a different variant).
///
/// When `requested` is `None`, the function respects whatever is already
/// installed: if any variant's complete **prequantized INT8 set** is in
/// `model_dir`, it is used as-is and **no network request is made**. Only when
/// the directory holds no usable INT8 model does it fall back to downloading
/// the default (`Rnnt`) **pre-quantized INT8** set. FP32-only installs are
/// ignored (not considered usable).
///
/// Returns the variant that is now ready in `model_dir`.
#[cfg(feature = "net")]
pub async fn ensure_model_variant(
    requested: Option<ModelVariant>,
    model_dir: &str,
) -> Result<ModelVariant> {
    let dir = Path::new(model_dir);

    // Determine the variant that is fully usable on disk. `detect_in_dir` only
    // checks for an encoder file, so we filter to variants whose complete INT8
    // set is present (FP32-only is not usable).
    let existing = ModelVariant::detect_in_dir(dir).filter(|&v| is_usable_present(v, dir));

    let variant = match resolve_variant(requested, existing) {
        VariantAction::Use(v) => {
            tracing::info!("Using existing {v:?} model at {model_dir}");
            return Ok(v);
        }
        VariantAction::Download(v) => v,
    };

    if let Some(other) = existing
        && other != variant
    {
        tracing::warn!(
            "Model directory {model_dir} holds {other:?} files but {variant:?} was \
             requested; downloading the {variant:?} set (variants are never mixed)"
        );
    }

    // Create the directory before acquiring the lock so the lock file can be
    // created inside it.
    std::fs::create_dir_all(dir).context("Failed to create model directory")?;

    #[cfg(unix)]
    let _lock = acquire_download_lock(dir)?;

    // Double-check after acquiring the lock in case another process finished
    // the download while we were waiting (prequantized INT8 set).
    if is_usable_present(variant, dir) {
        tracing::info!("Model ({variant:?}) found at {model_dir} after lock acquisition");
        return Ok(variant);
    }

    // Default fetch is the lean pre-quantized INT8 bundle for RNN-T heads
    // (GitHub Release). CTC heads already download INT8 from HuggingFace.
    // Use the FP32 HF path only when the lean path is unavailable for a head
    // that has no prequantized set (should not happen for shipped variants).
    if variant.is_ctc() {
        tracing::info!("Model ({variant:?}) not found, downloading from HuggingFace...");
        for file in variant.download_files() {
            download_file(variant, file, dir).await?;
        }
    } else {
        tracing::info!(
            "Model ({variant:?}) not found, downloading pre-quantized INT8 bundle from {PREQUANT_RELEASE_BASE}..."
        );
        for file in variant.prequantized_files() {
            let final_dest = dir.join(file);
            if final_dest.exists() {
                continue;
            }
            let url = format!("{PREQUANT_RELEASE_BASE}/{file}");
            let expected = variant.prequantized_checksum(file);
            stream_to_partial_then_finalize(&url, &final_dest, expected, file).await?;
        }
    }

    tracing::info!("Model download complete");
    Ok(variant)
}

/// Ensure the **FP32 download set** for `requested` (or the variant already on
/// disk, else default `Rnnt`) exists in `model_dir`, fetching from HuggingFace
/// when missing.
///
/// **Packaging / quantize source only** — the product runtime never loads FP32.
/// Prefer [`ensure_model_variant`] / [`ensure_prequantized_model_variant`] for
/// inference. Does not quantize; callers run [`crate::quantize`] separately.
#[cfg(feature = "net")]
pub async fn ensure_fp32_model_variant(
    requested: Option<ModelVariant>,
    model_dir: &str,
) -> Result<ModelVariant> {
    let dir = Path::new(model_dir);
    let existing = ModelVariant::detect_in_dir(dir).filter(|&v| is_model_present(v, dir));
    let variant = match resolve_variant(requested, existing) {
        VariantAction::Use(v) => {
            tracing::info!("Using existing FP32 {v:?} model at {model_dir}");
            return Ok(v);
        }
        VariantAction::Download(v) => v,
    };

    std::fs::create_dir_all(dir).context("Failed to create model directory")?;
    #[cfg(unix)]
    let _lock = acquire_download_lock(dir)?;

    if is_model_present(variant, dir) {
        tracing::info!("FP32 model ({variant:?}) found at {model_dir} after lock");
        return Ok(variant);
    }

    tracing::info!("Downloading FP32 {variant:?} model set from HuggingFace...");
    for file in variant.download_files() {
        download_file(variant, file, dir).await?;
    }
    tracing::info!("FP32 model download complete");
    Ok(variant)
}

/// Ensure the **pre-quantized** INT8 model bundle for `requested` (or the
/// variant already on disk, else the default `Rnnt`) exists in `model_dir`,
/// downloading it from the pinned GitHub Release if missing.
///
/// This is the product download path: INT8 encoder + decoder + joiner + vocab
/// (no FP32, no on-device quantize). Each file is SHA-256-verified and atomically
/// renamed, reusing the same download primitive as [`ensure_model_variant`].
///
/// If the **pre-quantized INT8 set** is already present, it is used as-is.
/// An FP32-only tree is **not** treated as ready (runtime is INT8-only).
#[cfg(feature = "net")]
pub async fn ensure_prequantized_model_variant(
    requested: Option<ModelVariant>,
    model_dir: &str,
) -> Result<ModelVariant> {
    let dir = Path::new(model_dir);
    let variant = requested
        .or_else(|| ModelVariant::detect_in_dir(dir).filter(|&v| is_usable_present(v, dir)))
        .unwrap_or_default();

    if is_prequantized_present(variant, dir) {
        tracing::info!("Using existing {variant:?} INT8 model at {model_dir}");
        return Ok(variant);
    }

    std::fs::create_dir_all(dir).context("Failed to create model directory")?;

    #[cfg(unix)]
    let _lock = acquire_download_lock(dir)?;

    // Re-check after acquiring the lock in case another process finished.
    if is_prequantized_present(variant, dir) {
        tracing::info!("Pre-quantized {variant:?} model found at {model_dir} after lock");
        return Ok(variant);
    }

    tracing::info!("Downloading pre-quantized {variant:?} model from {PREQUANT_RELEASE_BASE}...");

    for file in variant.prequantized_files() {
        let final_dest = dir.join(file);
        if final_dest.exists() {
            continue;
        }
        let url = format!("{PREQUANT_RELEASE_BASE}/{file}");
        let expected = variant.prequantized_checksum(file);
        stream_to_partial_then_finalize(&url, &final_dest, expected, file).await?;
    }

    tracing::info!("Pre-quantized model download complete");
    Ok(variant)
}

/// Directory name of the unpacked `.mlpackage` for a given mel bucket.
#[cfg(feature = "ane")]
pub fn ane_package_dir_name(bucket: usize) -> String {
    format!("gigaam_v3_encoder_{bucket}.mlpackage")
}

/// Filename of the published `.tar` artifact for a given mel bucket.
#[cfg(all(feature = "net", feature = "ane"))]
pub(super) fn ane_tar_name(bucket: usize) -> String {
    format!("{}.tar", ane_package_dir_name(bucket))
}

/// Pinned `.tar` SHA-256 for `bucket`, or `None` when unreleased (the empty
/// sentinel in [`ANE_TAR_CHECKSUMS`]).
#[cfg(all(feature = "net", feature = "ane"))]
fn ane_tar_checksum(bucket: usize) -> Option<&'static str> {
    ANE_TAR_CHECKSUMS
        .iter()
        .find(|(b, _)| *b == bucket)
        .and_then(|(_, sum)| if sum.is_empty() { None } else { Some(*sum) })
}

/// Return the default ANE-model directory (`~/.gigastt/models/ane/`), a sibling
/// of [`default_model_dir`].
///
/// Holds the per-bucket palettized Core ML encoder packages the macOS ANE
/// backend runs. The packages auto-download via [`ensure_ane_packages`] when the
/// ANE path is requested (`gigastt download --ane`).
#[cfg(feature = "ane")]
pub fn default_ane_model_dir() -> String {
    home_dir()
        .map(|h| {
            h.join(".gigastt")
                .join("models")
                .join("ane")
                .to_string_lossy()
                .into_owned()
        })
        .unwrap_or_else(|| ".gigastt/models/ane".into())
}

/// True when `pkg_dir` is a fully-formed Core ML `.mlpackage` directory.
///
/// Requires every structurally-load-bearing member, not just the
/// `Manifest.json` marker: the manifest, the serialized model spec, and the
/// weights blob. A package missing any of these cannot load on the ANE, so
/// treating it as "present" would wedge the download path forever. Observed
/// layout (real published package, all three buckets identical):
///   `Manifest.json`
///   `Data/com.apple.CoreML/model.mlmodel`
///   `Data/com.apple.CoreML/weights/weight.bin`
#[cfg(feature = "ane")]
pub fn ane_package_complete(pkg_dir: &Path) -> bool {
    pkg_dir.is_dir()
        && pkg_dir.join("Manifest.json").is_file()
        && pkg_dir
            .join("Data")
            .join("com.apple.CoreML")
            .join("model.mlmodel")
            .is_file()
        && pkg_dir
            .join("Data")
            .join("com.apple.CoreML")
            .join("weights")
            .join("weight.bin")
            .is_file()
}

/// True when every bucket's unpacked `.mlpackage` is present and complete in
/// `dir` (see [`ane_package_complete`] for the structural requirements).
#[cfg(feature = "ane")]
pub fn is_ane_present(dir: &Path) -> bool {
    ANE_BUCKETS
        .iter()
        .all(|&b| ane_package_complete(&dir.join(ane_package_dir_name(b))))
}

/// Ensure the per-bucket ANE Core ML encoder packages exist in `model_dir`,
/// downloading each bucket's deterministic `.tar` from the pinned GitHub Release
/// and unpacking it to reconstruct the `.mlpackage` directory.
///
/// Each `.tar` is SHA-256-verified against [`ANE_TAR_CHECKSUMS`] (one digest =
/// content identity = download pin) before being unpacked with the `tar` crate's
/// default path-traversal guard, then the `.tar` is removed. Buckets whose
/// `.mlpackage` is already present are skipped. Reuses the same streaming
/// download + atomic-rename + lock infra as [`ensure_prequantized_model_variant`].
///
/// Bails with a clear message when the release is not yet published (sentinel
/// checksums).
#[cfg(all(feature = "net", feature = "ane"))]
pub async fn ensure_ane_packages(model_dir: &str) -> Result<()> {
    let dir = Path::new(model_dir);

    if is_ane_present(dir) {
        tracing::info!("ANE encoder packages found at {model_dir}");
        return Ok(());
    }

    std::fs::create_dir_all(dir).context("Failed to create ANE model directory")?;

    #[cfg(unix)]
    let _lock = acquire_download_lock(dir)?;

    // Re-check after acquiring the lock in case another process finished.
    if is_ane_present(dir) {
        tracing::info!("ANE encoder packages found at {model_dir} after lock");
        return Ok(());
    }

    tracing::info!("Downloading ANE encoder packages from {ANE_RELEASE_BASE}...");

    for &bucket in ANE_BUCKETS {
        let pkg_name = ane_package_dir_name(bucket);
        if ane_package_complete(&dir.join(&pkg_name)) {
            continue;
        }
        let checksum = require_ane_tar_checksum(bucket)?;
        let tar_name = ane_tar_name(bucket);
        let tar_dest = dir.join(&tar_name);
        let url = format!("{ANE_RELEASE_BASE}/{tar_name}");
        stream_to_partial_then_finalize(&url, &tar_dest, Some(checksum), &tar_name).await?;

        // Extract atomically: unpack into a unique staging dir on the SAME
        // filesystem, then rename the reconstructed package into place so the
        // present-check only ever observes a fully-formed `.mlpackage`. A torn
        // unpack (disk-full / SIGKILL) leaves only the staging dir + the `.tar`,
        // both of which we remove on every error path so a retry starts clean.
        tracing::info!("Unpacking {tar_name} into {model_dir}");
        if let Err(e) = extract_ane_tar_atomic(&tar_dest, dir, &pkg_name) {
            let _ = std::fs::remove_file(&tar_dest);
            return Err(e);
        }
        // `.tar` removed only AFTER a successful rename, so a failed run above
        // retains it for retry.
        std::fs::remove_file(&tar_dest)
            .with_context(|| format!("Failed to remove {}", tar_dest.display()))?;
    }

    tracing::info!("ANE encoder packages download complete");
    Ok(())
}

/// Unpack `tar_dest` into a unique staging dir under `dir` (same filesystem →
/// atomic rename), then move the reconstructed `<pkg_name>` package into
/// `dir/<pkg_name>` with a single `rename`. The package only ever appears at
/// its final path fully-formed.
///
/// The deterministic `.tar`'s arcnames are prefixed with `<pkg_name>/`, so the
/// reconstructed package lands at `staging/<pkg_name>`. `tar::Archive::unpack`
/// keeps its default path-traversal guard (entries escaping the target are
/// rejected). On any failure the staging dir is removed before bailing so a
/// retry starts clean; the caller removes the `.tar`.
#[cfg(all(feature = "net", feature = "ane"))]
pub(super) fn extract_ane_tar_atomic(tar_dest: &Path, dir: &Path, pkg_name: &str) -> Result<()> {
    // Unique per-process staging dir, same pid+nanos scheme as
    // `partial_path_unique`, kept under `dir` so the final rename is atomic.
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let staging = dir.join(format!(".extract.{}.{}", std::process::id(), stamp));

    // Best-effort staging cleanup before any `?`-bail.
    let cleanup_staging = || {
        let _ = std::fs::remove_dir_all(&staging);
    };

    if let Err(e) = std::fs::create_dir_all(&staging)
        .with_context(|| format!("Failed to create staging dir {}", staging.display()))
    {
        cleanup_staging();
        return Err(e);
    }

    let unpack = (|| -> Result<()> {
        let tar_file = std::fs::File::open(tar_dest)
            .with_context(|| format!("Failed to open {}", tar_dest.display()))?;
        tar::Archive::new(tar_file)
            .unpack(&staging)
            .with_context(|| format!("Failed to unpack {}", tar_dest.display()))?;

        let src = staging.join(pkg_name);
        let dest = dir.join(pkg_name);
        // A torn package from a prior aborted run (the strengthened present-check
        // now rejects it) must be cleared before rename, or `rename` fails with
        // "directory not empty".
        if dest.exists() {
            std::fs::remove_dir_all(&dest)
                .with_context(|| format!("Failed to remove stale {}", dest.display()))?;
        }
        std::fs::rename(&src, &dest)
            .with_context(|| format!("Failed to rename {} -> {}", src.display(), dest.display()))?;
        Ok(())
    })();

    cleanup_staging();
    unpack
}

/// Resolve the pinned `.tar` checksum for `bucket`, bailing with the
/// not-yet-published message when it is a sentinel. Factored out so the
/// sentinel-bail branch is unit-testable without the network / async path.
#[cfg(all(feature = "net", feature = "ane"))]
pub(super) fn require_ane_tar_checksum(bucket: usize) -> Result<&'static str> {
    ane_tar_checksum(bucket).ok_or_else(|| {
        anyhow::anyhow!(
            "ANE encoder release not yet published; run the Release ANE workflow \
             (release-ane.yml), then pin the per-bucket .tar SHA-256 from \
             SHA256SUMS.txt in ANE_TAR_CHECKSUMS"
        )
    })
}

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

/// True when every downloaded file for `variant` is present in `dir`.
///
/// Checks the *downloaded* set (FP32 encoder, decoder, joiner, vocab); the
/// locally-generated INT8 encoder is not required for presence.
pub fn is_model_present(variant: ModelVariant, dir: &Path) -> bool {
    variant
        .download_files()
        .iter()
        .all(|f| dir.join(f).exists())
}

/// True when every file in `variant`'s pre-quantized bundle (INT8 encoder,
/// decoder, joiner, vocab) is present in `dir`. The engine runs from this set
/// alone — no FP32 encoder required.
pub fn is_prequantized_present(variant: ModelVariant, dir: &Path) -> bool {
    variant
        .prequantized_files()
        .iter()
        .all(|f| dir.join(f).exists())
}

/// True when the engine can load `variant` from `dir` without a download.
///
/// **INT8 only:** the lean pre-quantized set ([`is_prequantized_present`]) or,
/// for CTC heads, the same INT8-on-disk layout as [`is_model_present`] (CTC
/// download files are already INT8). An FP32-only install is **not** usable.
pub fn is_usable_present(variant: ModelVariant, dir: &Path) -> bool {
    if variant.is_ctc() {
        // CTC download set is the INT8 encoder + vocab (no separate prequant list).
        return is_model_present(variant, dir) || is_prequantized_present(variant, dir);
    }
    is_prequantized_present(variant, dir)
}

/// Append `.partial` to a path; retained for tests that assert the legacy
/// staging name. Production download path uses `partial_path_unique`.
#[cfg(test)]
pub(super) fn partial_path(final_path: &Path) -> std::path::PathBuf {
    let mut s: std::ffi::OsString = final_path.as_os_str().to_owned();
    s.push(".partial");
    std::path::PathBuf::from(s)
}

#[cfg(feature = "net")]
pub(super) mod fetch;
#[cfg(feature = "net")]
use fetch::{download_file, stream_to_partial_then_finalize};
