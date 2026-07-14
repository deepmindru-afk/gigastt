//! End-to-end tests for the asynchronous `/v1/jobs` API.
//!
//! All tests require the GigaAM model to be downloaded (~850MB).
//! Run with: `cargo test --test e2e_jobs -- --ignored --test-threads=1`

mod common;

use futures_util::StreamExt;
use std::time::Duration;

/// POST /v1/jobs returns 404 when the feature is not enabled.
#[ignore = "requires model"]
#[tokio::test]
async fn test_jobs_disabled_returns_404() {
    let (port, shutdown) = common::start_server(&common::model_dir()).await;

    let resp = tokio::time::timeout(Duration::from_secs(10), async {
        reqwest::Client::new()
            .post(format!("http://127.0.0.1:{port}/v1/jobs"))
            .body(common::generate_wav(1, 16000))
            .send()
            .await
            .expect("POST /v1/jobs failed")
    })
    .await
    .expect("POST /v1/jobs timed out");

    assert_eq!(resp.status(), 404);
    let _ = shutdown.send(());
}

/// Full happy path: submit a short WAV, poll until done, then fetch the result.
#[ignore = "requires model"]
#[tokio::test]
async fn test_job_submit_poll_done_result() {
    let (port, shutdown) = common::start_server_with_jobs(&common::model_dir(), 1).await;
    let wav = common::generate_wav(2, 16000);

    let submit = tokio::time::timeout(Duration::from_secs(10), async {
        reqwest::Client::new()
            .post(format!("http://127.0.0.1:{port}/v1/jobs"))
            .body(wav)
            .send()
            .await
            .expect("POST /v1/jobs failed")
    })
    .await
    .expect("POST /v1/jobs timed out");

    assert_eq!(submit.status(), 202);
    let text = submit
        .text()
        .await
        .expect("submit response should have text");
    let body: serde_json::Value =
        serde_json::from_str(&text).expect("submit response should be JSON");
    let job_id = body["job_id"].as_str().expect("job_id should be a string");
    assert_eq!(body["status"], "queued");

    // Poll until the job finishes.
    let client = reqwest::Client::new();
    let mut final_status = serde_json::Value::Null;
    for _ in 0..120 {
        let resp = client
            .get(format!("http://127.0.0.1:{port}/v1/jobs/{job_id}"))
            .send()
            .await
            .expect("GET /v1/jobs/{id} failed");
        assert_eq!(resp.status(), 200);
        let text = resp.text().await.expect("status response should have text");
        let status: serde_json::Value = serde_json::from_str(&text).expect("status should be JSON");
        if status["status"] == "done" || status["status"] == "failed" {
            final_status = status;
            break;
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }

    assert_eq!(
        final_status["status"], "done",
        "job should finish successfully: {final_status:?}"
    );

    let result = tokio::time::timeout(Duration::from_secs(10), async {
        client
            .get(format!("http://127.0.0.1:{port}/v1/jobs/{job_id}/result"))
            .send()
            .await
            .expect("GET /v1/jobs/{id}/result failed")
    })
    .await
    .expect("GET /v1/jobs/{id}/result timed out");

    assert_eq!(result.status(), 200);
    let text = result
        .text()
        .await
        .expect("result response should have text");
    let result_body: serde_json::Value =
        serde_json::from_str(&text).expect("result should be JSON");
    assert!(
        result_body["text"].is_string(),
        "result should contain text, got: {result_body:?}"
    );

    let _ = shutdown.send(());
}

/// Cancelling a queued job lets a synchronous `/v1/transcribe` run immediately.
#[ignore = "requires model"]
#[tokio::test]
async fn test_job_cancel_queued_frees_pool() {
    let (port, shutdown) = common::start_server_with_jobs(&common::model_dir(), 1).await;
    let client = reqwest::Client::new();

    // Submit a long WAV so the job stays queued for a moment.
    let submit = client
        .post(format!("http://127.0.0.1:{port}/v1/jobs"))
        .body(common::generate_wav(30, 16000))
        .send()
        .await
        .expect("POST /v1/jobs failed");
    assert_eq!(submit.status(), 202);
    let text = submit
        .text()
        .await
        .expect("submit response should have text");
    let body: serde_json::Value =
        serde_json::from_str(&text).expect("submit response should be JSON");
    let job_id = body["job_id"].as_str().expect("job_id should be a string");

    // Cancel immediately while still queued.
    let cancel = client
        .delete(format!("http://127.0.0.1:{port}/v1/jobs/{job_id}"))
        .send()
        .await
        .expect("DELETE /v1/jobs/{id} failed");
    assert_eq!(cancel.status(), 204);

    // Synchronous transcription should succeed without waiting for the cancelled job.
    let sync = tokio::time::timeout(Duration::from_secs(30), async {
        client
            .post(format!("http://127.0.0.1:{port}/v1/transcribe"))
            .body(common::generate_wav(1, 16000))
            .send()
            .await
            .expect("POST /v1/transcribe failed")
    })
    .await
    .expect("POST /v1/transcribe timed out");

    assert_eq!(sync.status(), 200);

    let _ = shutdown.send(());
}

/// SSE event stream emits progress and a terminal done event.
#[ignore = "requires model"]
#[tokio::test]
async fn test_job_sse_events() {
    let (port, shutdown) = common::start_server_with_jobs(&common::model_dir(), 1).await;
    let client = reqwest::Client::new();

    let submit = client
        .post(format!("http://127.0.0.1:{port}/v1/jobs"))
        .body(common::generate_wav(10, 16000))
        .send()
        .await
        .expect("POST /v1/jobs failed");
    assert_eq!(submit.status(), 202);
    let text = submit
        .text()
        .await
        .expect("submit response should have text");
    let body: serde_json::Value =
        serde_json::from_str(&text).expect("submit response should be JSON");
    let job_id = body["job_id"].as_str().expect("job_id should be a string");

    // Connect to the SSE stream before the job finishes.
    let resp = client
        .get(format!("http://127.0.0.1:{port}/v1/jobs/{job_id}/events"))
        .send()
        .await
        .expect("GET /v1/jobs/{id}/events failed");
    assert_eq!(resp.status(), 200);

    let mut stream = resp.bytes_stream();
    let mut saw_progress = false;
    let mut saw_done = false;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);

    while tokio::time::Instant::now() < deadline {
        match tokio::time::timeout(Duration::from_secs(1), stream.next()).await {
            Ok(Some(Ok(chunk))) => {
                let text = String::from_utf8_lossy(&chunk);
                for line in text.lines() {
                    let line = line.trim();
                    if let Some(data) = line.strip_prefix("data:") {
                        let data = data.trim();
                        if data.is_empty() {
                            continue;
                        }
                        let event: serde_json::Value =
                            serde_json::from_str(data).expect("SSE data should be JSON");
                        if event["type"] == "progress" {
                            saw_progress = true;
                        } else if event["type"] == "done" {
                            saw_done = true;
                        }
                    }
                }
            }
            Ok(Some(Err(e))) => panic!("SSE stream error: {e}"),
            Ok(None) => break,
            Err(_) => {}
        }
        if saw_done {
            break;
        }
    }

    assert!(saw_progress, "should receive at least one progress event");
    assert!(saw_done, "should receive a terminal done event");

    let _ = shutdown.send(());
}

/// A job with invalid audio eventually fails with a sanitized, client-safe error.
#[ignore = "requires model"]
#[tokio::test]
async fn test_job_invalid_audio_fails_with_sanitized_error() {
    let (port, shutdown) = common::start_server_with_jobs(&common::model_dir(), 1).await;
    let client = reqwest::Client::new();

    let submit = client
        .post(format!("http://127.0.0.1:{port}/v1/jobs"))
        .body(b"not-a-wav-file".to_vec())
        .send()
        .await
        .expect("POST /v1/jobs failed");
    assert_eq!(submit.status(), 202);
    let text = submit
        .text()
        .await
        .expect("submit response should have text");
    let body: serde_json::Value =
        serde_json::from_str(&text).expect("submit response should be JSON");
    let job_id = body["job_id"].as_str().expect("job_id should be a string");

    let mut final_status = serde_json::Value::Null;
    for _ in 0..60 {
        let resp = client
            .get(format!("http://127.0.0.1:{port}/v1/jobs/{job_id}"))
            .send()
            .await
            .expect("GET /v1/jobs/{id} failed");
        assert_eq!(resp.status(), 200);
        let text = resp.text().await.expect("status response should have text");
        let status: serde_json::Value = serde_json::from_str(&text).expect("status should be JSON");
        if status["status"] == "failed" {
            final_status = status;
            break;
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }

    assert_eq!(final_status["status"], "failed");
    assert!(
        final_status["error"].as_str().unwrap().contains("decode"),
        "error should be sanitized for clients: {final_status:?}"
    );

    let _ = shutdown.send(());
}
