//! INT8 encoder output against the author FP32 encoder on `golos_00`.
//!
//! The golden file is the author package (`gigaam` revision `7447938`,
//! `fp16_encoder=False`, CPU) encoder activation, float32, shape `[768, 100]`,
//! time fastest. This test is ignored: it loads the ONNX model. It skips when
//! that model is not installed.

use super::*;
use crate::runtime::tensor::{Shape, TensorDataView};

#[cfg(feature = "file-decode")]
#[test]
#[ignore = "needs the installed INT8 rnnt encoder; skips when that file is absent"]
fn test_golos_00_int8_encoder_near_author_fp32() {
    let Some(home) = std::env::var_os("HOME") else {
        eprintln!("skip: HOME is unset");
        return;
    };
    let model_dir = std::path::PathBuf::from(home).join(".gigastt/models");
    let encoder = model_dir.join("v3_rnnt_encoder_int8.onnx");
    if !encoder.is_file() {
        eprintln!(
            "skip: INT8 rnnt encoder not found at {}. This gate does not fetch weights.",
            encoder.display()
        );
        return;
    }
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let wav = root.join("../gigastt/tests/fixtures/golos_00.wav");
    let author_path = root.join("../../benchmark/oracle/golos_00_author_encoder.f32");
    let samples =
        crate::inference::audio::decode_audio_file(wav.to_str().expect("fixture path is utf-8"))
            .expect("decode golos_00");
    // Do not let a persistent optimized graph stand in for the pinned source
    // model when measuring numerical agreement with the author package.
    let optimized_cache = tempfile::tempdir().expect("isolated oracle graph cache");
    let engine = Engine::load_with_execution_provider(
        model_dir.to_str().expect("model dir is utf-8"),
        Some(ModelVariant::Rnnt),
        1,
        1,
        0,
        1,
        Some(optimized_cache.path().to_path_buf()),
        crate::runtime::ExecutionProviderChoice::Cpu,
    )
    .expect("load pinned INT8 rnnt engine on CPU");
    let (features, num_frames) = engine.features.compute(&samples);
    let mut guard = engine.pool.checkout_blocking().expect("checkout encoder");
    let triplet = &mut *guard;
    triplet.encoder_inputs[0].resize_to(Shape::new(vec![1, N_MELS, num_frames]));
    triplet.encoder_inputs[0]
        .as_f32_mut()
        .expect("encoder signal is f32")
        .copy_from_slice(&features);
    triplet.encoder_inputs[1]
        .as_i64_mut()
        .expect("encoder length is i64")[0] = i64::try_from(num_frames).expect("frame count");
    let outputs = triplet
        .encoder
        .run(&triplet.encoder_inputs)
        .expect("encoder run");
    let enc_len = match outputs[1].view().data() {
        TensorDataView::I32(v) => usize::try_from(v[0]).expect("encoder length"),
        TensorDataView::I64(v) => usize::try_from(v[0]).expect("encoder length"),
        _ => panic!("unexpected encoder length tensor"),
    };
    let ours = outputs[0]
        .view()
        .data()
        .as_f32()
        .expect("encoder output is f32");
    let bytes = std::fs::read(&author_path).expect("author encoder dump");
    let (chunks, rest) = bytes.as_chunks::<4>();
    assert!(
        rest.is_empty(),
        "author encoder dump is not a whole number of f32s"
    );
    let author: Vec<f32> = chunks.iter().copied().map(f32::from_le_bytes).collect();
    assert_eq!(enc_len, 100, "author encoder length on this clip is 100");
    assert_eq!(ours.len(), author.len(), "encoder tensor length");
    let mut dot = 0.0_f64;
    let mut norm_ours = 0.0_f64;
    let mut norm_author = 0.0_f64;
    let mut max_abs = 0.0_f32;
    for (&a, &b) in ours.iter().zip(author.iter()) {
        let da = f64::from(a);
        let db = f64::from(b);
        dot += da * db;
        norm_ours += da * da;
        norm_author += db * db;
        max_abs = max_abs.max((a - b).abs());
    }
    let cosine = dot / (norm_ours.sqrt() * norm_author.sqrt());
    eprintln!(
        "rnnt encoder sha256={} cosine={cosine} max_abs={max_abs}",
        ModelVariant::Rnnt.encoder_int8_checksum()
    );
    // Ryzen: 0.9985 / 0.145; GitHub ubuntu-latest: 0.99750 / 0.22470.
    // The pinned INT8 model is verified at engine load. Keep the cosine gate
    // and allow the observed cross-host maximum (see benchmark/tolerances).
    assert!(
        cosine > 0.997,
        "int8 encoder cosine {cosine} max_abs {max_abs}"
    );
    assert!(
        max_abs < 0.25,
        "int8 encoder max abs {max_abs} cosine {cosine}"
    );
    let transcript = engine
        .transcribe_samples(&samples, triplet)
        .expect("decode oracle clip");
    assert_eq!(
        transcript.text.trim(),
        "шестьдесят тысяч тенге сколько будет стоить",
        "INT8 greedy transcript must still match the author FP32 reference"
    );
}
