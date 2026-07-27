//! Model-directory cache hygiene: prune stale ORT optimized graphs and
//! optional content-hash hardlink dedupe.
//!
//! ONNX Runtime writes one optimized graph per encoder stem under
//! `optimized_cache/`. Switching heads or running FP32 leaves zombie graphs
//! that reclaim **hundreds of MiB to ~1 GiB** on polluted installs without
//! changing inference accuracy.

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::ModelVariant;

/// Basename of the optimized graph ORT would open for `model_path`'s stem:
/// `{file_stem}_optimized.onnx`.
pub fn optimized_cache_basename(encoder_path: &Path) -> Option<String> {
    encoder_path
        .file_stem()
        .map(|s| format!("{}_optimized.onnx", s.to_string_lossy()))
}

/// Preferred encoder path on disk for `variant`: INT8 when present, else FP32
/// (RNN-T). CTC heads are INT8-only.
pub fn preferred_encoder_path(variant: ModelVariant, dir: &Path) -> Option<PathBuf> {
    let int8 = dir.join(variant.encoder_int8_file());
    if int8.exists() {
        return Some(int8);
    }
    let enc = variant.encoder_file();
    if !enc.is_empty() {
        let fp32 = dir.join(enc);
        if fp32.exists() {
            return Some(fp32);
        }
    }
    None
}

/// Result of pruning `model_dir/optimized_cache/`.
#[derive(Debug, Default, Clone)]
pub struct OptimizedCachePruneReport {
    /// Active optimized graph retained (0 or 1 paths).
    pub kept: Vec<PathBuf>,
    /// Paths removed (or that would be removed under `dry_run`).
    pub removed: Vec<PathBuf>,
    /// Sum of sizes of `removed` entries.
    pub freed_bytes: u64,
    pub dry_run: bool,
}

/// Result of content-hash hardlink dedupe under `model_dir`.
#[derive(Debug, Default, Clone)]
pub struct DedupeReport {
    /// Number of hash groups with 2+ members.
    pub groups: usize,
    /// Number of paths re-linked (or that would be) onto the keep copy.
    pub hardlinked: usize,
    /// Approximate space reclaimed: sum of sizes of redundant members.
    pub freed_bytes: u64,
    pub dry_run: bool,
}

/// Drop non-active optimized graphs under `model_dir/optimized_cache/`.
///
/// Keeps only `{preferred_encoder_stem}_optimized.onnx` for the variant
/// detected in `model_dir` (INT8 preferred). When no head is detected, the
/// cache is left untouched so a half-installed tree is not wiped.
///
/// With `dry_run`, reports what would be deleted without removing files.
pub fn prune_optimized_cache(model_dir: &Path, dry_run: bool) -> Result<OptimizedCachePruneReport> {
    let mut report = OptimizedCachePruneReport {
        dry_run,
        ..Default::default()
    };
    let cache_dir = model_dir.join("optimized_cache");
    if !cache_dir.is_dir() {
        return Ok(report);
    }

    let keep_name = ModelVariant::detect_in_dir(model_dir)
        .and_then(|v| preferred_encoder_path(v, model_dir))
        .and_then(|p| optimized_cache_basename(&p));

    let Some(keep_name) = keep_name else {
        tracing::info!(
            "optimized_cache prune: no usable encoder in {}; leaving cache untouched",
            model_dir.display()
        );
        return Ok(report);
    };

    for entry in std::fs::read_dir(&cache_dir)
        .with_context(|| format!("failed to read optimized_cache at {}", cache_dir.display()))?
    {
        let entry = entry.context("failed to read optimized_cache entry")?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = entry.file_name();
        let name = name.to_string_lossy();
        // Only touch ORT optimized graphs; leave unrelated files alone.
        if !name.ends_with("_optimized.onnx") {
            continue;
        }
        if name.as_ref() == keep_name {
            report.kept.push(path);
            continue;
        }
        let len = entry.metadata().map(|m| m.len()).unwrap_or(0);
        if !dry_run {
            std::fs::remove_file(&path).with_context(|| {
                format!("failed to remove stale optimized graph {}", path.display())
            })?;
        }
        report.removed.push(path);
        report.freed_bytes = report.freed_bytes.saturating_add(len);
    }

    if !report.removed.is_empty() {
        tracing::info!(
            dry_run,
            kept = report.kept.len(),
            removed = report.removed.len(),
            freed_mib = report.freed_bytes / (1024 * 1024),
            "optimized_cache prune finished"
        );
    }
    Ok(report)
}

/// Hardlink exact duplicate files under `model_dir` (content SHA-256).
///
/// For each hash group with more than one path, keeps the first path in
/// stable sort order and replaces other members with hardlinks to it. Skips
/// `.partial*` staging files and directories. Cross-device hardlink failures
/// leave the original file in place and are reported via `tracing::warn`.
///
/// With `dry_run`, only measures reclaimable bytes.
pub fn dedupe_model_dir(model_dir: &Path, dry_run: bool) -> Result<DedupeReport> {
    let mut report = DedupeReport {
        dry_run,
        ..Default::default()
    };
    if !model_dir.is_dir() {
        return Ok(report);
    }

    let mut by_hash: HashMap<String, Vec<PathBuf>> = HashMap::new();
    collect_regular_files(model_dir, &mut by_hash)?;

    for mut paths in by_hash.into_values() {
        if paths.len() < 2 {
            continue;
        }
        paths.sort();
        report.groups += 1;
        let keep = paths[0].clone();
        let size = std::fs::metadata(&keep).map(|m| m.len()).unwrap_or(0);
        for other in paths.into_iter().skip(1) {
            if same_file(&keep, &other)? {
                continue;
            }
            if dry_run {
                report.hardlinked += 1;
                report.freed_bytes = report.freed_bytes.saturating_add(size);
                continue;
            }
            // Replace the redundant copy with a hardlink to `keep`.
            let tmp = other.with_extension(format!("dedupe-tmp.{}", std::process::id()));
            // Remove destination first so hard_link can create the name.
            // Use a rename-to-tmp then hardlink then remove-tmp so a crash
            // mid-way leaves either the original or a recoverable tmp.
            std::fs::rename(&other, &tmp)
                .with_context(|| format!("failed to stage {} for hardlink", other.display()))?;
            match std::fs::hard_link(&keep, &other) {
                Ok(()) => {
                    let _ = std::fs::remove_file(&tmp);
                    report.hardlinked += 1;
                    report.freed_bytes = report.freed_bytes.saturating_add(size);
                }
                Err(e) => {
                    // Restore original name on failure (e.g. cross-device).
                    let _ = std::fs::rename(&tmp, &other);
                    tracing::warn!(
                        keep = %keep.display(),
                        other = %other.display(),
                        error = %e,
                        "hardlink dedupe skipped"
                    );
                }
            }
        }
    }

    if report.hardlinked > 0 {
        tracing::info!(
            dry_run,
            groups = report.groups,
            hardlinked = report.hardlinked,
            freed_mib = report.freed_bytes / (1024 * 1024),
            "model-dir content-hash dedupe finished"
        );
    }
    Ok(report)
}

fn collect_regular_files(root: &Path, by_hash: &mut HashMap<String, Vec<PathBuf>>) -> Result<()> {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let rd = match std::fs::read_dir(&dir) {
            Ok(rd) => rd,
            Err(e) => {
                tracing::warn!(path = %dir.display(), error = %e, "skip unreadable dir");
                continue;
            }
        };
        for entry in rd {
            let entry = entry.with_context(|| format!("read_dir {}", dir.display()))?;
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            // Skip download staging and lock files.
            if name.contains(".partial") || name.ends_with(".lock") || name.starts_with('.') {
                continue;
            }
            let ft = entry
                .file_type()
                .with_context(|| format!("file_type {}", path.display()))?;
            if ft.is_dir() {
                stack.push(path);
                continue;
            }
            if !ft.is_file() {
                continue;
            }
            // Small files rarely reclaim meaningful space; still hash so exact
            // dups of vocab-scale files hardlink when present.
            let digest =
                sha256_file_streaming(&path).with_context(|| format!("hash {}", path.display()))?;
            by_hash.entry(digest).or_default().push(path);
        }
    }
    Ok(())
}

fn sha256_file_streaming(path: &Path) -> Result<String> {
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
    Ok(hex::encode(hasher.finalize()))
}

fn same_file(a: &Path, b: &Path) -> Result<bool> {
    let ma = std::fs::metadata(a)?;
    let mb = std::fs::metadata(b)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Ok(ma.dev() == mb.dev() && ma.ino() == mb.ino())
    }
    #[cfg(not(unix))]
    {
        // Best-effort: identical size is not enough; always attempt hardlink.
        let _ = (ma, mb);
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_file(path: &Path, bytes: &[u8]) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(bytes).unwrap();
    }

    #[test]
    fn test_optimized_cache_basename_from_int8_encoder() {
        let p = Path::new("/m/v3_rnnt_encoder_int8.onnx");
        assert_eq!(
            optimized_cache_basename(p).as_deref(),
            Some("v3_rnnt_encoder_int8_optimized.onnx")
        );
    }

    #[test]
    fn test_prune_optimized_cache_keeps_active_int8_drops_zombies() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        // Active rnnt INT8 install (lean set stub).
        for f in ModelVariant::Rnnt.prequantized_files() {
            write_file(&dir.join(f), b"stub");
        }
        let cache = dir.join("optimized_cache");
        let keep = cache.join("v3_rnnt_encoder_int8_optimized.onnx");
        let fp32 = cache.join("v3_rnnt_encoder_optimized.onnx");
        let e2e = cache.join("v3_e2e_rnnt_encoder_int8_optimized.onnx");
        let legacy = cache.join("encoder_optimized.onnx");
        write_file(&keep, &[1u8; 100]);
        write_file(&fp32, &[2u8; 200]);
        write_file(&e2e, &[3u8; 300]);
        write_file(&legacy, &[4u8; 400]);
        // Unrelated file must stay.
        write_file(&cache.join("notes.txt"), b"keep me");

        let report = prune_optimized_cache(dir, false).unwrap();
        assert_eq!(report.kept, vec![keep.clone()]);
        assert_eq!(report.removed.len(), 3);
        assert!(keep.exists());
        assert!(!fp32.exists());
        assert!(!e2e.exists());
        assert!(!legacy.exists());
        assert!(cache.join("notes.txt").exists());
        assert_eq!(report.freed_bytes, 200 + 300 + 400);
    }

    #[test]
    fn test_prune_optimized_cache_dry_run_does_not_delete() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        for f in ModelVariant::Rnnt.prequantized_files() {
            write_file(&dir.join(f), b"stub");
        }
        let cache = dir.join("optimized_cache");
        let zombie = cache.join("v3_rnnt_encoder_optimized.onnx");
        write_file(&zombie, &[9u8; 50]);

        let report = prune_optimized_cache(dir, true).unwrap();
        assert!(report.dry_run);
        assert_eq!(report.removed.len(), 1);
        assert!(zombie.exists(), "dry_run must not delete");
        assert_eq!(report.freed_bytes, 50);
    }

    #[test]
    fn test_prune_optimized_cache_no_head_leaves_cache() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        let cache = dir.join("optimized_cache");
        let zombie = cache.join("v3_rnnt_encoder_optimized.onnx");
        write_file(&zombie, b"x");

        let report = prune_optimized_cache(dir, false).unwrap();
        assert!(report.kept.is_empty());
        assert!(report.removed.is_empty());
        assert!(zombie.exists());
    }

    #[test]
    fn test_dedupe_hardlinks_exact_copies() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        let a = dir.join("a.onnx");
        let b = dir.join("sub").join("b.onnx");
        let payload = b"identical-payload-bytes";
        write_file(&a, payload);
        write_file(&b, payload);
        // Distinct file must not be linked.
        write_file(&dir.join("c.onnx"), b"different");

        let report = dedupe_model_dir(dir, false).unwrap();
        assert_eq!(report.groups, 1);
        assert_eq!(report.hardlinked, 1);
        assert!(same_file(&a, &b).unwrap());
        assert_eq!(std::fs::read(&a).unwrap(), payload);
        assert_eq!(std::fs::read(&b).unwrap(), payload);
        assert_eq!(std::fs::read(dir.join("c.onnx")).unwrap(), b"different");
    }

    #[test]
    fn test_dedupe_dry_run_no_hardlink() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        let a = dir.join("a.bin");
        let b = dir.join("b.bin");
        write_file(&a, b"same");
        write_file(&b, b"same");

        let report = dedupe_model_dir(dir, true).unwrap();
        assert!(report.dry_run);
        assert_eq!(report.hardlinked, 1);
        assert!(!same_file(&a, &b).unwrap());
    }
}
