use super::*;

#[test]
fn test_model_variant_default_is_rnnt() {
    assert_eq!(ModelVariant::default(), ModelVariant::Rnnt);
}

#[test]
fn test_model_variant_all_covers_every_variant() {
    // Compile-time guard: the match is exhaustive within this crate
    // (`#[non_exhaustive]` only binds downstream), so adding a head
    // without extending `ModelVariant::ALL` fails the build here —
    // the keep-set in `cache-gc` and `detect_in_dir` both derive from it.
    let mut count = 0;
    for v in ModelVariant::ALL {
        match v {
            ModelVariant::Rnnt
            | ModelVariant::E2eRnnt
            | ModelVariant::MlCtc
            | ModelVariant::MlCtcLarge => count += 1,
        }
    }
    assert_eq!(count, 4);
}

#[test]
fn test_model_variant_rnnt_file_mapping() {
    let v = ModelVariant::Rnnt;
    assert_eq!(v.encoder_file(), "v3_rnnt_encoder.onnx");
    assert_eq!(v.encoder_int8_file(), "v3_rnnt_encoder_int8.onnx");
    assert_eq!(v.decoder_file(), "v3_rnnt_decoder.onnx");
    assert_eq!(v.joint_file(), "v3_rnnt_joint.onnx");
    // The rnnt vocab name is asymmetric: v3_vocab.txt, NOT v3_rnnt_vocab.txt.
    assert_eq!(v.vocab_file(), "v3_vocab.txt");
    assert_eq!(
        v.download_files(),
        [
            "v3_rnnt_encoder.onnx",
            "v3_rnnt_decoder.onnx",
            "v3_rnnt_joint.onnx",
            "v3_vocab.txt",
        ]
    );
}

#[test]
fn test_model_variant_e2e_rnnt_file_mapping() {
    let v = ModelVariant::E2eRnnt;
    assert_eq!(v.encoder_file(), "v3_e2e_rnnt_encoder.onnx");
    assert_eq!(v.encoder_int8_file(), "v3_e2e_rnnt_encoder_int8.onnx");
    assert_eq!(v.decoder_file(), "v3_e2e_rnnt_decoder.onnx");
    assert_eq!(v.joint_file(), "v3_e2e_rnnt_joint.onnx");
    assert_eq!(v.vocab_file(), "v3_e2e_rnnt_vocab.txt");
    assert_eq!(
        v.download_files(),
        [
            "v3_e2e_rnnt_encoder.onnx",
            "v3_e2e_rnnt_decoder.onnx",
            "v3_e2e_rnnt_joint.onnx",
            "v3_e2e_rnnt_vocab.txt",
        ]
    );
}

#[test]
fn test_model_variant_from_str() {
    use std::str::FromStr;
    assert_eq!(ModelVariant::from_str("rnnt").unwrap(), ModelVariant::Rnnt);
    assert_eq!(
        ModelVariant::from_str("e2e_rnnt").unwrap(),
        ModelVariant::E2eRnnt
    );
    assert_eq!(
        ModelVariant::from_str("E2E-RNNT").unwrap(),
        ModelVariant::E2eRnnt
    );
    assert_eq!(
        ModelVariant::from_str(" RNNT ").unwrap(),
        ModelVariant::Rnnt
    );
    assert_eq!(
        ModelVariant::from_str("ml_ctc").unwrap(),
        ModelVariant::MlCtc
    );
    assert_eq!(
        ModelVariant::from_str("ML-CTC").unwrap(),
        ModelVariant::MlCtc
    );
    assert_eq!(
        ModelVariant::from_str("ml_ctc_large").unwrap(),
        ModelVariant::MlCtcLarge
    );
    assert_eq!(
        ModelVariant::from_str("ML-CTC-LARGE").unwrap(),
        ModelVariant::MlCtcLarge
    );
    assert!(ModelVariant::from_str("whisper").is_err());
}

#[test]
fn test_model_variant_ml_ctc_file_mapping() {
    let v = ModelVariant::MlCtc;
    // Real istupakov filenames (gigaam-multilingual-ctc-onnx).
    assert_eq!(v.encoder_file(), "multilingual_ctc.onnx");
    assert_eq!(v.encoder_int8_file(), "multilingual_ctc.int8.onnx");
    assert_eq!(v.vocab_file(), "multilingual_vocab.txt");
    // Encoder-only: no decoder/joiner ONNX exists.
    assert_eq!(v.decoder_file(), "");
    assert_eq!(v.joint_file(), "");
    // Downloads the pre-quantized INT8 encoder directly + vocab.
    assert_eq!(
        v.download_files(),
        ["multilingual_ctc.int8.onnx", "multilingual_vocab.txt"]
    );
    assert_eq!(v.hf_repo(), "istupakov/gigaam-multilingual-ctc-onnx");
    assert_eq!(v.as_str(), "ml_ctc");
    assert_eq!(v.model_id(), "gigaam-multilingual-ctc");
}

#[test]
fn test_hf_repo_per_variant() {
    assert_eq!(ModelVariant::Rnnt.hf_repo(), "istupakov/gigaam-v3-onnx");
    assert_eq!(ModelVariant::E2eRnnt.hf_repo(), "istupakov/gigaam-v3-onnx");
    assert_eq!(
        ModelVariant::MlCtc.hf_repo(),
        "istupakov/gigaam-multilingual-ctc-onnx"
    );
    assert_eq!(
        ModelVariant::MlCtcLarge.hf_repo(),
        "istupakov/gigaam-multilingual-large-ctc-onnx"
    );
}

#[test]
fn test_model_variant_ml_ctc_large_file_mapping() {
    let v = ModelVariant::MlCtcLarge;
    assert_eq!(v.encoder_file(), "multilingual_large_ctc.onnx");
    assert_eq!(v.encoder_int8_file(), "multilingual_large_ctc.int8.onnx");
    // Vocab is byte-identical to (and shares the filename with) the 220M head.
    assert_eq!(v.vocab_file(), "multilingual_vocab.txt");
    assert_eq!(v.vocab_file(), ModelVariant::MlCtc.vocab_file());
    assert_eq!(v.decoder_file(), "");
    assert_eq!(v.joint_file(), "");
    assert_eq!(
        v.download_files(),
        ["multilingual_large_ctc.int8.onnx", "multilingual_vocab.txt"]
    );
    assert_eq!(v.as_str(), "ml_ctc_large");
    assert_eq!(v.model_id(), "gigaam-multilingual-large-ctc");
    assert!(v.is_ctc());
    assert!(ModelVariant::MlCtc.is_ctc());
    assert!(!ModelVariant::Rnnt.is_ctc());
}

#[test]
fn test_detect_in_dir_ml_ctc_large_by_int8_encoder() {
    let tmp = tempfile::tempdir().expect("tempdir");
    std::fs::write(tmp.path().join("multilingual_large_ctc.int8.onnx"), b"int8").unwrap();
    assert_eq!(
        ModelVariant::detect_in_dir(tmp.path()),
        Some(ModelVariant::MlCtcLarge)
    );
}

#[test]
fn test_detect_in_dir_ml_ctc_by_int8_encoder() {
    let tmp = tempfile::tempdir().expect("tempdir");
    std::fs::write(tmp.path().join("multilingual_ctc.int8.onnx"), b"int8").unwrap();
    assert_eq!(
        ModelVariant::detect_in_dir(tmp.path()),
        Some(ModelVariant::MlCtc)
    );
}

#[test]
fn test_model_variant_checksums_are_pinned() {
    // Every downloaded file for every variant has a pinned 64-char hex
    // checksum — security parity, no placeholder slipping into a release.
    for variant in [
        ModelVariant::Rnnt,
        ModelVariant::E2eRnnt,
        ModelVariant::MlCtc,
        ModelVariant::MlCtcLarge,
    ] {
        for file in variant.download_files() {
            let sum = variant
                .checksum(file)
                .unwrap_or_else(|| panic!("{variant:?} {file} must have a pinned checksum"));
            assert_eq!(
                sum.len(),
                64,
                "{variant:?} {file} checksum must be 64 hex chars, got: {sum}"
            );
            assert!(
                sum.chars()
                    .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
                "{variant:?} {file} checksum must be lowercase hex, got: {sum}"
            );
        }
    }
}

#[test]
fn test_detect_in_dir_rnnt_by_fp32_encoder() {
    let tmp = tempfile::tempdir().expect("tempdir");
    std::fs::write(tmp.path().join("v3_rnnt_encoder.onnx"), b"fp32").unwrap();
    assert_eq!(
        ModelVariant::detect_in_dir(tmp.path()),
        Some(ModelVariant::Rnnt)
    );
}

#[test]
fn test_detect_in_dir_rnnt_by_int8_encoder() {
    let tmp = tempfile::tempdir().expect("tempdir");
    std::fs::write(tmp.path().join("v3_rnnt_encoder_int8.onnx"), b"int8").unwrap();
    assert_eq!(
        ModelVariant::detect_in_dir(tmp.path()),
        Some(ModelVariant::Rnnt)
    );
}

#[test]
fn test_detect_in_dir_e2e_by_fp32_encoder() {
    let tmp = tempfile::tempdir().expect("tempdir");
    std::fs::write(tmp.path().join("v3_e2e_rnnt_encoder.onnx"), b"fp32").unwrap();
    assert_eq!(
        ModelVariant::detect_in_dir(tmp.path()),
        Some(ModelVariant::E2eRnnt)
    );
}

#[test]
fn test_detect_in_dir_e2e_by_int8_encoder() {
    let tmp = tempfile::tempdir().expect("tempdir");
    std::fs::write(tmp.path().join("v3_e2e_rnnt_encoder_int8.onnx"), b"int8").unwrap();
    assert_eq!(
        ModelVariant::detect_in_dir(tmp.path()),
        Some(ModelVariant::E2eRnnt)
    );
}

#[test]
fn test_detect_in_dir_none_when_empty() {
    let tmp = tempfile::tempdir().expect("tempdir");
    assert_eq!(ModelVariant::detect_in_dir(tmp.path()), None);
}

#[test]
fn test_is_model_present_per_variant() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path();
    // Stage a full rnnt download set.
    for f in ModelVariant::Rnnt.download_files() {
        std::fs::write(dir.join(f), b"x").unwrap();
    }
    assert!(
        is_model_present(ModelVariant::Rnnt, dir),
        "rnnt set is complete"
    );
    assert!(
        !is_model_present(ModelVariant::E2eRnnt, dir),
        "e2e set is absent — must not be reported present"
    );
}

#[test]
fn test_is_model_present_false_when_one_file_missing() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path();
    // Stage all but the vocab.
    for f in [
        ModelVariant::Rnnt.encoder_file(),
        ModelVariant::Rnnt.decoder_file(),
        ModelVariant::Rnnt.joint_file(),
    ] {
        std::fs::write(dir.join(f), b"x").unwrap();
    }
    assert!(
        !is_model_present(ModelVariant::Rnnt, dir),
        "a missing vocab must make the set incomplete"
    );
}

// ── resolve_variant decision table ──────────────────────────────────────

#[test]
fn test_resolve_variant_none_empty_dir_downloads_default() {
    // None requested + no existing → download Rnnt (the default)
    assert_eq!(
        resolve_variant(None, None),
        VariantAction::Download(ModelVariant::Rnnt),
    );
}

#[test]
fn test_resolve_variant_none_e2e_present_uses_e2e() {
    // None requested + E2eRnnt already installed → use it, no download
    assert_eq!(
        resolve_variant(None, Some(ModelVariant::E2eRnnt)),
        VariantAction::Use(ModelVariant::E2eRnnt),
    );
}

#[test]
fn test_resolve_variant_none_rnnt_present_uses_rnnt() {
    // None requested + Rnnt already installed → use it, no download
    assert_eq!(
        resolve_variant(None, Some(ModelVariant::Rnnt)),
        VariantAction::Use(ModelVariant::Rnnt),
    );
}

#[test]
fn test_resolve_variant_some_rnnt_rnnt_present_uses_rnnt() {
    // Explicit Rnnt + Rnnt installed → no download needed
    assert_eq!(
        resolve_variant(Some(ModelVariant::Rnnt), Some(ModelVariant::Rnnt)),
        VariantAction::Use(ModelVariant::Rnnt),
    );
}

#[test]
fn test_resolve_variant_some_e2e_rnnt_present_downloads_e2e() {
    // Explicit E2eRnnt + Rnnt installed → must switch, so download E2eRnnt
    assert_eq!(
        resolve_variant(Some(ModelVariant::E2eRnnt), Some(ModelVariant::Rnnt)),
        VariantAction::Download(ModelVariant::E2eRnnt),
    );
}

#[test]
fn test_resolve_variant_some_e2e_empty_downloads_e2e() {
    // Explicit E2eRnnt + nothing installed → download E2eRnnt
    assert_eq!(
        resolve_variant(Some(ModelVariant::E2eRnnt), None),
        VariantAction::Download(ModelVariant::E2eRnnt),
    );
}

#[test]
fn test_resolve_variant_some_rnnt_e2e_present_downloads_rnnt() {
    // Explicit Rnnt + E2eRnnt installed → must switch, download Rnnt
    assert_eq!(
        resolve_variant(Some(ModelVariant::Rnnt), Some(ModelVariant::E2eRnnt)),
        VariantAction::Download(ModelVariant::Rnnt),
    );
}
