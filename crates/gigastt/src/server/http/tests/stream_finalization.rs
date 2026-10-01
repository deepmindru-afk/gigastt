use super::*;
use axum::body::{Body, to_bytes};
use axum::extract::{FromRequest, Multipart};
use axum::http::Request;
use axum::response::IntoResponse;

#[tokio::test]
async fn test_file_stream_tail_errors_never_emit_success() {
    for openai in [false, true] {
        for prior in [false, true] {
            for joiner in [false, true] {
                for panic in [false, true] {
                    let tmp = tempfile::tempdir().unwrap();
                    gigastt_core::test_support::write_rnnt_layout(tmp.path()).unwrap();
                    let factory =
                        gigastt_core::test_support::FailingStreamFactory::new(joiner, panic);
                    let engine = Arc::new(
                        gigastt_core::test_support::load_rnnt_engine_with_factory(
                            tmp.path(),
                            1,
                            Box::new(factory.clone()),
                        )
                        .unwrap(),
                    );
                    factory.arm(if prior { 2 } else { 1 });
                    let state = Arc::new(AppState {
                        engine: engine_swap(engine.clone()),
                        limits: Arc::new(ArcSwap::from_pointee(RuntimeLimits::default())),
                        metrics_registry: None,
                        engine_builder: None,
                        reload_lock: Arc::new(tokio::sync::Mutex::new(())),
                        shutdown: tokio_util::sync::CancellationToken::new(),
                        tracker: tokio_util::task::TaskTracker::new(),
                        jobs: None,
                    });
                    let wav = gigastt_core::test_support::pcm16_wav(
                        &vec![0; if prior { 17_600 } else { 1_600 }],
                        16_000,
                    );
                    let response = if openai {
                        let mut body = b"--test\r\nContent-Disposition: form-data; name=\"stream\"\r\n\r\ntrue\r\n--test\r\nContent-Disposition: form-data; name=\"file\"; filename=\"test.wav\"\r\nContent-Type: audio/wav\r\n\r\n".to_vec();
                        body.extend(wav);
                        body.extend(b"\r\n--test--\r\n");
                        let request = Request::post("/")
                            .header("content-type", "multipart/form-data; boundary=test")
                            .body(Body::from(body))
                            .unwrap();
                        let multipart = Multipart::from_request(request, &()).await.unwrap();
                        openai_transcriptions(State(state.clone()), multipart)
                            .await
                            .unwrap()
                    } else {
                        transcribe_stream(
                            State(state.clone()),
                            Query(Default::default()),
                            Bytes::from(wav),
                        )
                        .await
                        .unwrap()
                        .into_response()
                    };
                    let bytes = tokio::time::timeout(
                        std::time::Duration::from_secs(5),
                        to_bytes(response.into_body(), 1_000_000),
                    )
                    .await
                    .unwrap()
                    .unwrap();
                    let body = String::from_utf8(bytes.to_vec()).unwrap();
                    assert!(
                        body.contains(if panic {
                            "inference_panic"
                        } else {
                            "inference_error"
                        }),
                        "{body}"
                    );
                    assert!(!body.contains("\"type\":\"final\""), "{body}");
                    assert!(!body.contains("transcript.text.done"), "{body}");
                    assert!(!body.contains("[DONE]"), "{body}");
                    if prior {
                        assert!(body.contains("hi"), "{body}");
                    }
                    state.tracker.close();
                    state.tracker.wait().await;
                    assert_eq!(engine.pool_for_batch().available(), 1);
                }
            }
        }
    }
}
