use super::*;

#[test]
fn test_resolve_raw_codec_absent_is_none() {
    let params = ExportParams::default();
    assert!(resolve_raw_codec(&params).unwrap().is_none());
}

#[test]
fn test_resolve_raw_codec_valid_pairs() {
    for (name, rate) in [
        ("pcmu", 8000),
        ("ulaw", 8000),
        ("pcma", 8000),
        ("alaw", 16000),
        ("g722", 8000),
        ("g722", 16000),
    ] {
        let params = ExportParams {
            codec: Some(name.into()),
            sample_rate: Some(rate),
            ..Default::default()
        };
        assert!(
            resolve_raw_codec(&params).unwrap().is_some(),
            "{name}@{rate} must resolve"
        );
    }
}

#[tokio::test]
async fn test_resolve_raw_codec_unknown_codec_is_400() {
    let params = ExportParams {
        codec: Some("g729".into()),
        sample_rate: Some(8000),
        ..Default::default()
    };
    let resp = resolve_raw_codec(&params).unwrap_err();
    let (parts, body) = resp.into_parts();
    assert_eq!(parts.status, StatusCode::BAD_REQUEST);
    let bytes = axum::body::to_bytes(body, 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["code"], "unsupported_codec");
}

#[tokio::test]
async fn test_resolve_raw_codec_missing_sample_rate_is_400() {
    let params = ExportParams {
        codec: Some("pcmu".into()),
        sample_rate: None,
        ..Default::default()
    };
    let resp = resolve_raw_codec(&params).unwrap_err();
    let (parts, body) = resp.into_parts();
    assert_eq!(parts.status, StatusCode::BAD_REQUEST);
    let bytes = axum::body::to_bytes(body, 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["code"], "invalid_sample_rate");
}

#[tokio::test]
async fn test_resolve_raw_codec_bad_sample_rate_is_400() {
    for (name, rate) in [("pcmu", 4000), ("g722", 44100)] {
        let params = ExportParams {
            codec: Some(name.into()),
            sample_rate: Some(rate),
            ..Default::default()
        };
        let resp = resolve_raw_codec(&params).unwrap_err();
        let (parts, body) = resp.into_parts();
        assert_eq!(parts.status, StatusCode::BAD_REQUEST, "{name}@{rate}");
        let bytes = axum::body::to_bytes(body, 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["code"], "invalid_sample_rate", "{name}@{rate}");
    }
}

#[test]
fn test_raw_codec_to_wav_produces_decodable_wav() {
    // μ-law silence (0xFF ≈ 0) re-wraps into a WAV the standard pipeline
    // accepts: the full raw→16kHz-WAV transform without a model.
    let raw = vec![0xFFu8; 8000]; // 1 s of μ-law silence at 8 kHz
    let wav = raw_codec_to_wav(
        &raw,
        gigastt_core::inference::audio::TelephonyCodec::Pcmu,
        8000,
    )
    .unwrap();
    let samples = gigastt_core::inference::audio::decode_audio_bytes_shared(wav).unwrap();
    assert!(
        samples.len() > 12_000 && samples.len() <= 16_000,
        "expected ~1 s at 16 kHz, got {}",
        samples.len()
    );
    assert!(
        samples.iter().all(|s| s.abs() < 0.01),
        "μ-law silence must decode to near-silence"
    );
}

#[test]
fn test_raw_codec_to_wav_rejects_bad_input() {
    let result = raw_codec_to_wav(
        &[],
        gigastt_core::inference::audio::TelephonyCodec::Pcmu,
        8000,
    );
    assert!(result.is_err(), "empty raw payload must error");
}
