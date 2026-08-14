//! Audio-file discovery and export-format parsing for batch / watch.

use anyhow::Context;
use gigastt_core::export::ExportFormat;
use std::path::{Path, PathBuf};

/// Audio extensions accepted by the batch / watch walkers (case-insensitive).
/// Mirrors the symphonia-backed file support of `Engine::transcribe_file`.
pub const SUPPORTED_EXTENSIONS: &[&str] = &["wav", "mp3", "m4a", "ogg", "flac", "webm"];

/// Whether `path` carries a supported audio extension (case-insensitive).
pub fn is_audio_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| SUPPORTED_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// Recursively collect supported audio files under `root`, sorted for a
/// deterministic processing order. Symlinked directories are not followed.
pub fn collect_audio_files(root: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    collect_into(root, &mut out)?;
    out.sort();
    Ok(out)
}

fn collect_into(dir: &Path, out: &mut Vec<PathBuf>) -> anyhow::Result<()> {
    let entries = std::fs::read_dir(dir)
        .with_context(|| format!("failed to read directory {}", dir.display()))?;
    for entry in entries {
        let entry = entry.with_context(|| format!("failed to read entry in {}", dir.display()))?;
        let path = entry.path();
        // DirEntry::file_type does not follow symlinks, so a symlinked
        // directory can never send the walker into a loop.
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_into(&path, out)?;
        } else if path.is_file() && is_audio_file(&path) {
            out.push(path);
        }
    }
    Ok(())
}

/// Output path for one input file and format: `<output_dir>/<stem>.<ext>`.
pub fn output_path_for(input: &Path, output_dir: &Path, extension: &str) -> PathBuf {
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("transcript");
    output_dir.join(format!("{stem}.{extension}"))
}

/// Parse a comma-separated format list (`txt,json,md,srt,vtt`) into
/// [`ExportFormat`]s, de-duplicated, order preserved.
pub fn parse_formats(s: &str) -> Result<Vec<ExportFormat>, String> {
    let mut out: Vec<ExportFormat> = Vec::new();
    for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let fmt = part
            .parse::<ExportFormat>()
            .map_err(|_| format!("unsupported export format: {part}"))?;
        if !out.contains(&fmt) {
            out.push(fmt);
        }
    }
    if out.is_empty() {
        return Err("at least one export format is required".to_string());
    }
    Ok(out)
}
