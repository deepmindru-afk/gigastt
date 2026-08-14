use super::*;

#[test]
fn test_sse_data_payload_preserves_error_codes() {
    // Per-variant code is preserved (not collapsed to a generic string),
    // including the distinct inference_panic / inference_timeout events.
    for code in [
        "invalid_audio",
        "inference_error",
        "inference_panic",
        "inference_timeout",
    ] {
        let payload = sse_data_payload(&Err(StreamError {
            code,
            message: "sanitized".into(),
        }));
        let v: serde_json::Value = serde_json::from_str(&payload).unwrap();
        assert_eq!(v["type"], "error");
        assert_eq!(v["code"], code);
        assert_eq!(v["message"], "sanitized");
    }
}

#[test]
fn test_sse_data_payload_segment_framing() {
    // A final segment renders as type "final"; a non-final one as "partial".
    let seg = gigastt_core::inference::TranscriptSegment::empty_final();
    let final_payload = sse_data_payload(&Ok(seg));
    let v: serde_json::Value = serde_json::from_str(&final_payload).unwrap();
    assert_eq!(v["type"], "final");

    let mut partial = gigastt_core::inference::TranscriptSegment::empty_final();
    partial.is_final = false;
    let partial_payload = sse_data_payload(&Ok(partial));
    let v: serde_json::Value = serde_json::from_str(&partial_payload).unwrap();
    assert_eq!(v["type"], "partial");
}

#[test]
fn test_sse_data_payload_confidence_present_only_when_some() {
    // A segment with words carries the aggregate; an empty one omits the
    // key entirely, matching the WS payload contract.
    let mut seg = gigastt_core::inference::TranscriptSegment::empty_final();
    seg.confidence = Some(0.85);
    let payload = sse_data_payload(&Ok(seg));
    let v: serde_json::Value = serde_json::from_str(&payload).unwrap();
    let c = v["confidence"].as_f64().expect("numeric confidence");
    assert!((c - 0.85).abs() < 1e-6, "got {c}");

    let empty = gigastt_core::inference::TranscriptSegment::empty_final();
    let payload = sse_data_payload(&Ok(empty));
    let v: serde_json::Value = serde_json::from_str(&payload).unwrap();
    assert!(v.get("confidence").is_none());
}

#[test]
fn test_sse_data_payload_includes_words_and_timestamp() {
    // A successful segment carries text, timestamp and words through
    // unchanged so SSE clients can render word-level UI.
    use gigastt_core::inference::WordInfo;
    let mut seg = gigastt_core::inference::TranscriptSegment::empty_final();
    seg.text = "привет".into();
    seg.timestamp = 1.25;
    seg.words = vec![WordInfo::new("привет", 0.0, 0.5, 0.99, Some(0))];
    let payload = sse_data_payload(&Ok(seg));
    let v: serde_json::Value = serde_json::from_str(&payload).unwrap();
    assert_eq!(v["type"], "final");
    assert_eq!(v["text"], "привет");
    assert_eq!(v["timestamp"], 1.25);
    assert_eq!(v["words"][0]["word"], "привет");
}
