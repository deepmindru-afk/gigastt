//! Model-file resolution for engine load (variant override + manifest + INT8 paths).
//!
//! Free of ONNX sessions so override / manifest / disk precedence is unit-testable
//! without model weights.

use std::path::Path;

use crate::error::GigasttError;
use crate::model::{ModelManifest, ModelVariant};

/// Resolve which recognition head the engine should load.
///
/// Precedence:
/// 1. Explicit `override_` (from `--model-variant`) always wins.
/// 2. Else `manifest.toml` architecture, when that file is present.
/// 3. Else auto-detect from on-disk encoder filenames (`rnnt` precedence).
///
/// A present-but-invalid `manifest.toml` is a hard error (not silently ignored).
pub(crate) fn resolve_load_variant(
    override_: Option<ModelVariant>,
    model_dir: &Path,
) -> anyhow::Result<Option<ModelVariant>> {
    // Always validate a present manifest so a corrupt pack fails clearly, even
    // when the CLI override selects the architecture.
    let manifest = ModelManifest::load(model_dir)?;
    if let Some(v) = override_ {
        return Ok(Some(v));
    }
    if let Some(m) = manifest {
        return Ok(Some(m.architecture));
    }
    Ok(ModelVariant::detect_in_dir(model_dir))
}

/// Like [`resolve_load_variant`], but maps missing/invalid layouts to
/// [`GigasttError::ModelLoad`] for engine load entry points.
pub(crate) fn resolve_variant_required(
    override_: Option<ModelVariant>,
    model_dir: &Path,
) -> Result<ModelVariant, GigasttError> {
    match resolve_load_variant(override_, model_dir) {
        Ok(Some(v)) => Ok(v),
        Ok(None) => Err(GigasttError::ModelLoad {
            path: model_dir.display().to_string(),
            source: None,
        }),
        Err(e) => Err(GigasttError::ModelLoad {
            path: model_dir.display().to_string(),
            source: Some(e.into()),
        }),
    }
}

/// On-disk model file paths for a load: either from `manifest.toml` or from the
/// hardcoded [`ModelVariant`] basenames when no manifest is present.
pub(crate) struct ResolvedModelFiles {
    pub encoder: std::path::PathBuf,
    pub decoder: Option<std::path::PathBuf>,
    pub joint: Option<std::path::PathBuf>,
    pub vocab: std::path::PathBuf,
    pub using_int8: bool,
}

impl ResolvedModelFiles {
    pub(crate) fn resolve(dir: &Path, variant: ModelVariant) -> anyhow::Result<Self> {
        if let Some(m) = ModelManifest::load(dir)? {
            anyhow::ensure!(
                m.prefers_int8(dir),
                "manifest resolves to a non-INT8 encoder — gigastt runs INT8 only. \
                 Install the INT8 encoder (`gigastt download`) or fix encoder_int8 in manifest.toml."
            );
            return Ok(Self {
                encoder: m.preferred_encoder_path(dir),
                decoder: m.decoder_path(dir),
                joint: m.joint_path(dir),
                vocab: m.vocab_path(dir),
                using_int8: true,
            });
        }
        Self::from_variant(dir, variant)
    }

    pub(crate) fn from_variant(dir: &Path, variant: ModelVariant) -> anyhow::Result<Self> {
        // Product policy: only the INT8 encoder is supported. FP32 ONNX is not
        // loaded (no silent fallback). Fix: `gigastt download` (lean INT8).
        let int8 = dir.join(variant.encoder_int8_file());
        anyhow::ensure!(
            int8.is_file(),
            "INT8 encoder not found at {} — gigastt runs INT8 only. \
             Run `gigastt download` (lean INT8 bundle). FP32 encoders are not supported.",
            int8.display()
        );
        if variant.is_ctc() {
            Ok(Self {
                encoder: int8,
                decoder: None,
                joint: None,
                vocab: dir.join(variant.vocab_file()),
                using_int8: true,
            })
        } else {
            Ok(Self {
                encoder: int8,
                decoder: Some(dir.join(variant.decoder_file())),
                joint: Some(dir.join(variant.joint_file())),
                vocab: dir.join(variant.vocab_file()),
                using_int8: true,
            })
        }
    }

    /// Check every resolved file that has a pinned digest. Custom
    /// `manifest.toml` names with no table entry are skipped.
    pub(crate) fn verify_pinned_checksums(
        &self,
        variant: ModelVariant,
    ) -> Result<(), GigasttError> {
        let mut files: Vec<&std::path::Path> = vec![&self.encoder, &self.vocab];
        if let Some(d) = self.decoder.as_deref() {
            files.push(d);
        }
        if let Some(j) = self.joint.as_deref() {
            files.push(j);
        }
        for path in files {
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            let Some(expected) = variant.checksum(name) else {
                continue;
            };
            crate::model::verify_pinned_checksum(path, expected)?;
        }
        Ok(())
    }
}

use crate::runtime::factory::Runtime;
use crate::runtime::tensor::{Shape, Tensor, TensorData};

use super::N_MELS;
use super::pool::SessionTriplet;
use super::sizing;

/// Path to the preferred INT8 encoder for `variant`. Honors `manifest.toml`
/// when present; falls back to the expected INT8 basename when resolve fails
/// (tests may call this without a full install).
pub(crate) fn encoder_model_path(dir: &Path, variant: ModelVariant) -> std::path::PathBuf {
    ResolvedModelFiles::resolve(dir, variant)
        .map(|files| files.encoder)
        .unwrap_or_else(|_| dir.join(variant.encoder_int8_file()))
}

/// Load up to `pool_size` session triplets in parallel through the given
/// [`Runtime`], tolerating a partial pool down to `min_size`.
pub(crate) fn load_triplets_runtime(
    runtime: &dyn Runtime,
    files: &ResolvedModelFiles,
    variant: ModelVariant,
    pool_size: usize,
    min_size: usize,
) -> anyhow::Result<Vec<SessionTriplet>> {
    let encoder_path = files.encoder.clone();
    // CTC is encoder-only: no decoder/joiner ONNX exists on disk, and the CTC
    // branch in `run_inference` returns right after the encoder run without
    // touching them. Load them only for the RNN-T heads (leaving `None` for
    // CTC avoids holding an unused, never-run session per pool triplet).
    let is_ctc = variant.is_ctc();
    let decoder_path = files.decoder.clone();
    let joiner_path = files.joint.clone();

    let results: Vec<anyhow::Result<SessionTriplet>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..pool_size)
            .map(|i| {
                let encoder_path = &encoder_path;
                let decoder_path = &decoder_path;
                let joiner_path = &joiner_path;
                s.spawn(move || {
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        tracing::info!(
                            "Loading session triplet {}/{pool_size} (shared runtime)",
                            i + 1
                        );
                        let encoder = runtime
                            .load_session(encoder_path, true)
                            .map_err(|e| anyhow::anyhow!(e))?;
                        let (decoder, joiner) = if is_ctc {
                            (None, None)
                        } else {
                            let decoder_path = decoder_path.as_ref().ok_or_else(|| {
                                anyhow::anyhow!(
                                    "decoder ONNX path missing for non-CTC architecture {}",
                                    variant.as_str()
                                )
                            })?;
                            let joiner_path = joiner_path.as_ref().ok_or_else(|| {
                                anyhow::anyhow!(
                                    "joint ONNX path missing for non-CTC architecture {}",
                                    variant.as_str()
                                )
                            })?;
                            let decoder = runtime
                                .load_session(decoder_path, false)
                                .map_err(|e| anyhow::anyhow!(e))?;
                            let joiner = runtime
                                .load_session(joiner_path, false)
                                .map_err(|e| anyhow::anyhow!(e))?;
                            (Some(decoder), Some(joiner))
                        };
                        Ok(SessionTriplet {
                            encoder,
                            decoder,
                            joiner,
                            encoder_inputs: vec![
                                Tensor::new(
                                    Shape::new(vec![1, N_MELS, 1]),
                                    TensorData::F32(vec![0.0; N_MELS]),
                                )?,
                                Tensor::new(Shape::new(vec![1]), TensorData::I64(vec![0]))?,
                            ],
                        })
                    }))
                    .map_err(|_| anyhow::anyhow!("model loading thread panicked"))?
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| match h.join() {
                Ok(r) => r,
                Err(_) => Err(anyhow::anyhow!("model loading thread panicked")),
            })
            .collect()
    });
    sizing::finalize_pool_load(results, pool_size, min_size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelVariant;

    #[test]
    fn test_resolve_load_variant_override_beats_disk_detection() {
        // A directory holding BOTH the rnnt and e2e_rnnt encoders: on-disk
        // detection returns rnnt (precedence), so without an explicit override the
        // engine would silently ignore `--model-variant e2e_rnnt`. This is the
        // regression this fix guards.
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join(ModelVariant::Rnnt.encoder_file()), b"x").unwrap();
        std::fs::write(dir.path().join(ModelVariant::E2eRnnt.encoder_file()), b"x").unwrap();

        // Sanity: bare on-disk detection prefers rnnt.
        assert_eq!(
            ModelVariant::detect_in_dir(dir.path()),
            Some(ModelVariant::Rnnt)
        );
        // No override → auto-detect (rnnt precedence): behavior is unchanged.
        assert_eq!(
            resolve_load_variant(None, dir.path()).unwrap(),
            Some(ModelVariant::Rnnt)
        );
        // Explicit override wins over the higher-precedence head on disk.
        assert_eq!(
            resolve_load_variant(Some(ModelVariant::E2eRnnt), dir.path()).unwrap(),
            Some(ModelVariant::E2eRnnt)
        );

        // The override is honored even when its files aren't present — the engine
        // load then fails with a clear ModelLoad error instead of silently loading
        // whatever else is on disk.
        let empty = tempfile::tempdir().expect("tempdir");
        assert_eq!(
            resolve_load_variant(Some(ModelVariant::E2eRnnt), empty.path()).unwrap(),
            Some(ModelVariant::E2eRnnt)
        );
        // No override + empty dir → nothing to load.
        assert_eq!(resolve_load_variant(None, empty.path()).unwrap(), None);
    }

    #[test]
    fn test_resolve_load_variant_manifest_beats_disk_detection() {
        // Disk has rnnt (higher precedence), but manifest selects e2e_rnnt.
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join(ModelVariant::Rnnt.encoder_file()), b"x").unwrap();
        std::fs::write(dir.path().join(ModelVariant::E2eRnnt.encoder_file()), b"x").unwrap();
        std::fs::write(
            dir.path().join(crate::model::MANIFEST_FILE),
            r#"
architecture = "e2e_rnnt"
[files]
encoder = "v3_e2e_rnnt_encoder.onnx"
decoder = "v3_e2e_rnnt_decoder.onnx"
joint = "v3_e2e_rnnt_joint.onnx"
vocab = "v3_e2e_rnnt_vocab.txt"
"#,
        )
        .unwrap();

        assert_eq!(
            ModelVariant::detect_in_dir(dir.path()),
            Some(ModelVariant::Rnnt)
        );
        assert_eq!(
            resolve_load_variant(None, dir.path()).unwrap(),
            Some(ModelVariant::E2eRnnt)
        );
        // CLI override still wins over the manifest architecture.
        assert_eq!(
            resolve_load_variant(Some(ModelVariant::Rnnt), dir.path()).unwrap(),
            Some(ModelVariant::Rnnt)
        );
    }

    #[test]
    fn test_resolve_load_variant_invalid_manifest_is_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join(crate::model::MANIFEST_FILE),
            "not = [valid\n",
        )
        .unwrap();
        assert!(resolve_load_variant(None, dir.path()).is_err());
        // Even with an override, a corrupt manifest is a hard error.
        assert!(resolve_load_variant(Some(ModelVariant::Rnnt), dir.path()).is_err());
    }

    #[test]
    fn test_resolved_model_files_without_manifest_match_variant_basenames() {
        // Regression: no manifest.toml → byte-identical hardcoded names.
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("v3_rnnt_encoder.onnx"), b"fp32").unwrap();
        std::fs::write(dir.path().join("v3_rnnt_encoder_int8.onnx"), b"int8").unwrap();

        let files = ResolvedModelFiles::resolve(dir.path(), ModelVariant::Rnnt).unwrap();
        assert_eq!(
            files.encoder.file_name().unwrap(),
            "v3_rnnt_encoder_int8.onnx"
        );
        assert!(files.using_int8);
        assert_eq!(
            files.decoder.as_ref().unwrap().file_name().unwrap(),
            "v3_rnnt_decoder.onnx"
        );
        assert_eq!(
            files.joint.as_ref().unwrap().file_name().unwrap(),
            "v3_rnnt_joint.onnx"
        );
        assert_eq!(files.vocab.file_name().unwrap(), "v3_vocab.txt");
    }

    #[test]
    fn test_resolved_model_files_manifest_encoder_prefers_int8() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("custom_enc.onnx"), b"fp32").unwrap();
        std::fs::write(dir.path().join("custom_enc_int8.onnx"), b"int8").unwrap();
        std::fs::write(
            dir.path().join(crate::model::MANIFEST_FILE),
            r#"
architecture = "rnnt"
[files]
encoder = "custom_enc.onnx"
encoder_int8 = "custom_enc_int8.onnx"
decoder = "custom_dec.onnx"
joint = "custom_joint.onnx"
vocab = "custom_vocab.txt"
"#,
        )
        .unwrap();

        let files = ResolvedModelFiles::resolve(dir.path(), ModelVariant::Rnnt).unwrap();
        assert_eq!(
            files.encoder.file_name().unwrap(),
            "custom_enc_int8.onnx",
            "manifest INT8 encoder must win when present on disk"
        );
        assert!(files.using_int8);
        assert_eq!(
            files.decoder.as_ref().unwrap().file_name().unwrap(),
            "custom_dec.onnx"
        );
        assert_eq!(files.vocab.file_name().unwrap(), "custom_vocab.txt");
    }

    #[test]
    fn test_encoder_model_path_prefers_int8_when_present() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("v3_e2e_rnnt_encoder.onnx"), b"fp32").unwrap();
        std::fs::write(dir.path().join("v3_e2e_rnnt_encoder_int8.onnx"), b"int8").unwrap();
        let path = encoder_model_path(dir.path(), ModelVariant::E2eRnnt);
        assert_eq!(
            path.file_name().unwrap(),
            "v3_e2e_rnnt_encoder_int8.onnx",
            "INT8 encoder must win when both files exist"
        );
    }

    #[test]
    fn test_encoder_model_path_fp32_only_is_rejected() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("v3_e2e_rnnt_encoder.onnx"), b"fp32").unwrap();
        // Without INT8, resolve fails; helper still reports the INT8 basename
        // (never the FP32 file).
        let path = encoder_model_path(dir.path(), ModelVariant::E2eRnnt);
        assert_eq!(path.file_name().unwrap(), "v3_e2e_rnnt_encoder_int8.onnx");
        assert!(
            ResolvedModelFiles::from_variant(dir.path(), ModelVariant::E2eRnnt).is_err(),
            "FP32-only install must not be loadable"
        );
    }

    #[test]
    fn test_encoder_model_path_rnnt_prefers_int8() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("v3_rnnt_encoder.onnx"), b"fp32").unwrap();
        std::fs::write(dir.path().join("v3_rnnt_encoder_int8.onnx"), b"int8").unwrap();
        let path = encoder_model_path(dir.path(), ModelVariant::Rnnt);
        assert_eq!(
            path.file_name().unwrap(),
            "v3_rnnt_encoder_int8.onnx",
            "INT8 rnnt encoder must win when both files exist"
        );
    }

    #[test]
    fn test_encoder_model_path_rnnt_fp32_only_is_rejected() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("v3_rnnt_encoder.onnx"), b"fp32").unwrap();
        assert!(
            ResolvedModelFiles::from_variant(dir.path(), ModelVariant::Rnnt).is_err(),
            "FP32-only install must not be loadable"
        );
    }

    #[test]
    fn test_encoder_model_path_uses_manifest_int8_when_present() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("pack_enc.onnx"), b"fp32").unwrap();
        std::fs::write(dir.path().join("pack_enc_int8.onnx"), b"int8").unwrap();
        std::fs::write(
            dir.path().join(crate::model::MANIFEST_FILE),
            r#"
architecture = "rnnt"
[files]
encoder = "pack_enc.onnx"
encoder_int8 = "pack_enc_int8.onnx"
decoder = "pack_dec.onnx"
joint = "pack_joint.onnx"
vocab = "pack_vocab.txt"
"#,
        )
        .unwrap();
        let path = encoder_model_path(dir.path(), ModelVariant::Rnnt);
        assert_eq!(path.file_name().unwrap(), "pack_enc_int8.onnx");
    }

    #[test]
    fn test_verify_pinned_checksums_rejects_placeholder_encoder() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("v3_rnnt_encoder_int8.onnx"), b"int8").unwrap();
        std::fs::write(dir.path().join("v3_rnnt_decoder.onnx"), b"dec").unwrap();
        std::fs::write(dir.path().join("v3_rnnt_joint.onnx"), b"joint").unwrap();
        std::fs::write(dir.path().join("v3_vocab.txt"), b"a\n").unwrap();
        let files = ResolvedModelFiles::resolve(dir.path(), ModelVariant::Rnnt).unwrap();
        let err = files
            .verify_pinned_checksums(ModelVariant::Rnnt)
            .expect_err("placeholder bytes must not match the pinned digest");
        let msg = format!("{err}");
        assert!(
            msg.contains("SHA-256 mismatch") || msg.contains("model load error"),
            "unexpected error: {msg}"
        );
    }
}
