//! ANE encoder package presence and download.

use anyhow::Result;
use std::path::Path;

use super::super::variant::ANE_BUCKETS;
#[cfg(feature = "net")]
use super::super::variant::{ANE_RELEASE_BASE, ANE_TAR_CHECKSUMS};

use super::home_dir;

#[cfg(all(unix, feature = "net"))]
use super::acquire_download_lock;
#[cfg(feature = "net")]
use super::fetch::stream_to_partial_then_finalize;
#[cfg(feature = "net")]
use anyhow::Context;

/// Directory name of the unpacked `.mlpackage` for a given mel bucket.
#[cfg(feature = "ane")]
pub fn ane_package_dir_name(bucket: usize) -> String {
    format!("gigaam_v3_encoder_{bucket}.mlpackage")
}

/// Filename of the published `.tar` artifact for a given mel bucket.
#[cfg(all(feature = "net", feature = "ane"))]
pub(crate) fn ane_tar_name(bucket: usize) -> String {
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
pub(crate) fn extract_ane_tar_atomic(tar_dest: &Path, dir: &Path, pkg_name: &str) -> Result<()> {
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
pub(crate) fn require_ane_tar_checksum(bucket: usize) -> Result<&'static str> {
    ane_tar_checksum(bucket).ok_or_else(|| {
        anyhow::anyhow!(
            "ANE encoder release not yet published; run the Release ANE workflow \
             (release-ane.yml), then pin the per-bucket .tar SHA-256 from \
             SHA256SUMS.txt in ANE_TAR_CHECKSUMS"
        )
    })
}
