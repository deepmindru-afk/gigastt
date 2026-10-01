use super::*;

#[test]
fn test_sse_text_parts_match_websocket_and_legacy_text() {
    use gigastt_core::inference::{TranscriptAssembler, WordInfo};
    let mut assembler = TranscriptAssembler::new();
    assembler.append(vec![WordInfo::new("привет", 0.0, 0.5, 1.0, None)]);
    assembler.commit_live();
    assembler.append(vec![WordInfo::new("мир", 0.5, 1.0, 1.0, None)]);
    for segment in [assembler.partial(1.0), assembler.finalize(2.0)] {
        let ws = if segment.is_final {
            gigastt_core::protocol::ServerMessage::Final(segment.clone())
        } else {
            gigastt_core::protocol::ServerMessage::Partial(segment.clone())
        };
        let ws = serde_json::to_value(ws).unwrap();
        let sse: serde_json::Value = serde_json::from_str(&sse_data_payload(&Ok(segment))).unwrap();
        for field in ["committed", "tentative", "text"] {
            assert_eq!(sse[field], ws[field]);
        }
        assert_eq!(sse["text"], "привет мир");
        assert_eq!(
            sse["text"].as_str().unwrap(),
            sse["committed"].as_str().unwrap().to_owned() + sse["tentative"].as_str().unwrap()
        );
    }
}

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
    seg.committed = seg.text.clone();
    seg.timestamp = 1.25;
    seg.words = vec![WordInfo::new("привет", 0.0, 0.5, 0.99, Some(0))];
    let payload = sse_data_payload(&Ok(seg));
    let v: serde_json::Value = serde_json::from_str(&payload).unwrap();
    assert_eq!(v["type"], "final");
    assert_eq!(v["text"], "привет");
    assert_eq!(v["timestamp"], 1.25);
    assert_eq!(v["words"][0]["word"], "привет");
}

#[test]
fn test_sse_truncated_final_keeps_text_and_omits_the_flag_when_false() {
    let mut cut = gigastt_core::inference::TranscriptSegment::empty_final();
    cut.text = "привет мир".into();
    cut.committed.clear();
    cut.tentative = cut.text.clone();
    let cut = cut.into_truncated_final();
    let v: serde_json::Value = serde_json::from_str(&sse_data_payload(&Ok(cut))).unwrap();
    assert_eq!(v["type"], "final");
    assert_eq!(v["truncated"], true);
    assert_eq!(v["text"], "привет мир");
    assert_eq!(v["committed"], "привет мир");
    assert_eq!(v["tentative"], "");

    let plain = gigastt_core::inference::TranscriptSegment::empty_final();
    let v: serde_json::Value = serde_json::from_str(&sse_data_payload(&Ok(plain))).unwrap();
    assert!(v.get("truncated").is_none());
}

#[tokio::test(start_paused = true)]
async fn test_full_stream_queue_times_out_once_and_aborts_request() {
    use crate::server::http::stream::{STREAM_SEND_TIMEOUT, send_stream_item};
    use std::sync::atomic::{AtomicBool, Ordering};
    let (tx, mut rx) = tokio::sync::mpsc::channel(1);
    tx.try_send("queued").unwrap();
    let server = tokio_util::sync::CancellationToken::new();
    let cancel = server.child_token();
    let abort = AtomicBool::new(false);
    let start = tokio::time::Instant::now();
    assert!(
        !tokio::time::timeout(
            STREAM_SEND_TIMEOUT * 2,
            send_stream_item(&tx, "blocked", &cancel, &abort)
        )
        .await
        .expect("bounded send")
    );
    assert_eq!(start.elapsed(), STREAM_SEND_TIMEOUT);
    assert!(abort.load(Ordering::Relaxed));
    assert!(cancel.is_cancelled());
    assert!(!server.is_cancelled());
    assert!(!send_stream_item(&tx, "terminal", &cancel, &abort).await);
    assert_eq!(start.elapsed(), STREAM_SEND_TIMEOUT);
    assert_eq!(rx.recv().await, Some("queued"));
    assert!(rx.try_recv().is_err());
}

#[tokio::test(start_paused = true)]
async fn test_full_stream_queue_shutdown_interrupts_pending_send() {
    use crate::server::http::stream::send_stream_item;
    use std::sync::atomic::{AtomicBool, Ordering};
    let (tx, _rx) = tokio::sync::mpsc::channel(1);
    tx.try_send("queued").unwrap();
    let cancel = tokio_util::sync::CancellationToken::new();
    let abort = AtomicBool::new(false);
    let send = send_stream_item(&tx, "blocked", &cancel, &abort);
    tokio::pin!(send);
    assert!(futures_util::poll!(&mut send).is_pending());
    cancel.cancel();
    assert!(
        !tokio::time::timeout(std::time::Duration::from_secs(1), send)
            .await
            .expect("shutdown interrupts send")
    );
    assert!(abort.load(Ordering::Relaxed));
}

#[tokio::test(start_paused = true)]
async fn test_full_stream_queue_disconnect_interrupts_pending_send() {
    use crate::server::http::stream::send_stream_item;
    use std::sync::atomic::{AtomicBool, Ordering};
    let (tx, rx) = tokio::sync::mpsc::channel(1);
    tx.try_send("queued").unwrap();
    let cancel = tokio_util::sync::CancellationToken::new();
    let abort = AtomicBool::new(false);
    let send = send_stream_item(&tx, "blocked", &cancel, &abort);
    tokio::pin!(send);
    assert!(futures_util::poll!(&mut send).is_pending());
    drop(rx);
    assert!(!send.await);
    assert!(abort.load(Ordering::Relaxed));
}

#[tokio::test(start_paused = true)]
async fn test_slow_stream_reader_preserves_fifo_when_capacity_returns_before_deadline() {
    use crate::server::http::stream::{STREAM_SEND_TIMEOUT, send_stream_item};
    use std::sync::atomic::{AtomicBool, Ordering};
    let (tx, mut rx) = tokio::sync::mpsc::channel(1);
    tx.try_send("committed partial").unwrap();
    let cancel = tokio_util::sync::CancellationToken::new();
    let abort = AtomicBool::new(false);
    let send = send_stream_item(&tx, "final", &cancel, &abort);
    tokio::pin!(send);
    assert!(futures_util::poll!(&mut send).is_pending());
    tokio::time::advance(STREAM_SEND_TIMEOUT - std::time::Duration::from_millis(1)).await;
    assert_eq!(rx.recv().await, Some("committed partial"));
    assert!(send.await);
    assert_eq!(rx.recv().await, Some("final"));
    assert!(!abort.load(Ordering::Relaxed));
    assert!(!cancel.is_cancelled());
}
