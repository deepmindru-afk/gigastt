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
