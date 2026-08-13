//! Unit tests for model download / variant / progress.

use super::download::*;
use super::progress::*;
use super::variant::*;
use crate::sha256::{Sha256, hex_lower};
use std::io::Write;
use std::path::Path;
#[cfg(feature = "net")]
use tokio::io::AsyncWriteExt;

#[test]
fn test_home_dir_returns_some() {
    // On any CI or developer machine HOME / USERPROFILE should be set.
    assert!(
        home_dir().is_some(),
        "home_dir() must return Some on this platform"
    );
}

#[test]
fn test_hash_file_sha256_and_verify() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("blob.bin");
    std::fs::write(&path, b"hello").unwrap();
    let digest = hash_file_sha256(&path).expect("hash");
    assert_eq!(digest.len(), 64);
    verify_pinned_checksum(&path, &digest).expect("matching digest");
    let wrong = "00".repeat(32);
    let err = verify_pinned_checksum(&path, &wrong).expect_err("wrong digest");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("SHA-256 mismatch") || msg.contains("model load error"),
        "unexpected: {msg}"
    );
}

#[cfg(feature = "net")]
#[test]
fn test_reject_if_over_download_cap() {
    assert!(reject_if_over_download_cap(0, MAX_DOWNLOAD_BYTES).is_ok());
    assert!(reject_if_over_download_cap(0, MAX_DOWNLOAD_BYTES + 1).is_err());
    assert!(reject_if_over_download_cap(MAX_DOWNLOAD_BYTES, 1).is_err());
    assert!(reject_if_over_download_cap(MAX_DOWNLOAD_BYTES - 10, 10).is_ok());
}

#[test]
fn test_default_model_dir_contains_gigastt() {
    let dir = default_model_dir();
    assert!(
        dir.contains(".gigastt"),
        "default_model_dir() should contain \".gigastt\", got: {dir}"
    );
}

#[test]
fn test_download_progress_basic() {
    let sink = ProgressSink::human();
    let mut progress = DownloadProgress::new(1_000_000);
    // Should not panic on normal update.
    progress.update(500_000, &sink, "model.onnx");
    assert_eq!(progress.current, 500_000);
    assert_eq!(progress.last_percent, 50);
    progress.finish(&sink, "model.onnx");
}

#[test]
fn test_download_progress_zero_total() {
    let sink = ProgressSink::human();
    let mut progress = DownloadProgress::new(0);
    // Must not divide by zero.
    progress.update(100, &sink, "model.onnx");
    assert_eq!(progress.last_percent, 0);
    progress.finish(&sink, "model.onnx");
}

// ── progress events (NDJSON / human sink) ───────────────────────────────

#[test]
fn test_progress_mode_from_str() {
    use std::str::FromStr;
    assert_eq!(
        ProgressMode::from_str("human").unwrap(),
        ProgressMode::Human
    );
    assert_eq!(ProgressMode::from_str("json").unwrap(), ProgressMode::Json);
    assert_eq!(
        ProgressMode::from_str(" JSON ").unwrap(),
        ProgressMode::Json
    );
    assert!(ProgressMode::from_str("xml").is_err());
    assert_eq!(ProgressMode::default(), ProgressMode::Human);
    assert_eq!(ProgressMode::Json.as_str(), "json");
}

/// The NDJSON wire shape is the integrator contract: one line per event,
/// `phase` as the discriminator, exactly the fields sidecars match on.
#[test]
fn test_progress_event_ndjson_schema() {
    let cases: Vec<(ProgressEvent, &str)> = vec![
        (
            ProgressEvent::Download {
                file: "v3_rnnt_encoder.onnx".to_string(),
                bytes_done: 50,
                bytes_total: 100,
            },
            "{\"phase\":\"download\",\"file\":\"v3_rnnt_encoder.onnx\",\"bytes_done\":50,\"bytes_total\":100}",
        ),
        (
            ProgressEvent::Quantize {
                file: "v3_rnnt_encoder.onnx".to_string(),
            },
            "{\"phase\":\"quantize\",\"file\":\"v3_rnnt_encoder.onnx\"}",
        ),
        (
            ProgressEvent::Verify {
                file: "v3_vocab.txt".to_string(),
            },
            "{\"phase\":\"verify\",\"file\":\"v3_vocab.txt\"}",
        ),
        (
            ProgressEvent::Done {
                model_dir: "/home/u/.gigastt/models".to_string(),
            },
            "{\"phase\":\"done\",\"model_dir\":\"/home/u/.gigastt/models\"}",
        ),
        (
            ProgressEvent::Error {
                kind: ProgressErrorKind::Network,
                message: "connection refused".to_string(),
            },
            "{\"phase\":\"error\",\"kind\":\"network\",\"message\":\"connection refused\"}",
        ),
        (
            ProgressEvent::Error {
                kind: ProgressErrorKind::Interrupted,
                message: "SIGINT".to_string(),
            },
            "{\"phase\":\"error\",\"kind\":\"interrupted\",\"message\":\"SIGINT\"}",
        ),
    ];
    for (event, want) in cases {
        let line = event.to_ndjson();
        assert_eq!(line, want);
        // Every line must round-trip as a JSON object with a `phase` tag.
        let parsed: serde_json::Value =
            serde_json::from_str(&line).expect("NDJSON line must parse");
        assert!(parsed.get("phase").is_some(), "phase tag missing: {line}");
    }
}

#[test]
fn test_progress_error_kind_exit_codes_keep_nonzero_contract() {
    assert_eq!(ProgressErrorKind::Other.exit_code(), 1);
    assert_eq!(ProgressErrorKind::Network.exit_code(), 69);
    assert_eq!(ProgressErrorKind::Disk.exit_code(), 74);
    assert_eq!(ProgressErrorKind::Checksum.exit_code(), 65);
    assert_eq!(ProgressErrorKind::Interrupted.exit_code(), 130);
    // 2 stays reserved for clap usage errors: a misconfigured invocation
    // must never look like a transient (retryable) download failure.
    for kind in [
        ProgressErrorKind::Other,
        ProgressErrorKind::Network,
        ProgressErrorKind::Disk,
        ProgressErrorKind::Checksum,
        ProgressErrorKind::Interrupted,
    ] {
        assert_ne!(kind.exit_code(), 2, "{kind:?} must not collide with clap");
    }
    for kind in [
        ProgressErrorKind::Network,
        ProgressErrorKind::Disk,
        ProgressErrorKind::Checksum,
        ProgressErrorKind::Interrupted,
        ProgressErrorKind::Other,
    ] {
        assert_ne!(kind.exit_code(), 0, "{kind:?} must keep != 0");
    }
}

/// Human mode must stay byte-for-byte identical to the legacy `\r`
/// reporter: same format strings, same trailing-space padding.
#[test]
fn test_download_progress_human_render_matches_legacy() {
    let mut progress = DownloadProgress::new(10 * 1_048_576);
    // current=0, percent 0 == last_percent 0 -> no redraw.
    assert_eq!(progress.human_tick(), None);
    progress.current = 5 * 1_048_576;
    assert_eq!(
        progress.human_tick().as_deref(),
        Some("\rDownloading... 50% (5.0MB / 10.0MB)")
    );
    // Same percentage -> throttled (no redraw).
    assert_eq!(progress.human_tick(), None);
    progress.current = 10 * 1_048_576;
    assert_eq!(
        progress.human_tick().as_deref(),
        Some("\rDownloading... 100% (10.0MB / 10.0MB)")
    );
    assert_eq!(
        progress.human_finish(),
        "\rDownload complete (10.0MB)                    "
    );
}

/// Json mode: first chunk emits immediately, rapid chunks within the 200 ms
/// window are throttled, and 100% always emits exactly once.
#[test]
fn test_download_progress_json_first_throttled_then_final() {
    let (sink, log) = ProgressSink::capturing();
    let mut progress = DownloadProgress::new(1_000);
    // 100 chunks of 10 bytes, all well inside the throttle window.
    for _ in 0..100 {
        progress.update(10, &sink, "model.onnx");
    }
    let events = log.lock().unwrap();
    assert_eq!(
        events.as_slice(),
        [
            ProgressEvent::Download {
                file: "model.onnx".to_string(),
                bytes_done: 10,
                bytes_total: 1_000,
            },
            ProgressEvent::Download {
                file: "model.onnx".to_string(),
                bytes_done: 1_000,
                bytes_total: 1_000,
            },
        ]
    );
    // finish() must not duplicate the already-emitted 100% event.
    drop(events);
    progress.finish(&sink, "model.onnx");
    assert_eq!(log.lock().unwrap().len(), 2);
}

/// Json mode: once the throttle window has elapsed, mid-file progress
/// emits again (integrators get a steady cadence on long downloads).
#[test]
fn test_download_progress_json_emits_after_throttle_window() {
    let (sink, log) = ProgressSink::capturing();
    let mut progress = DownloadProgress::new(1_000);
    progress.update(10, &sink, "model.onnx");
    // Backdate the last emission past the throttle window.
    progress.last_json_emit = Some(
        std::time::Instant::now() - JSON_PROGRESS_THROTTLE - std::time::Duration::from_millis(50),
    );
    progress.update(10, &sink, "model.onnx");
    let events = log.lock().unwrap();
    assert_eq!(events.len(), 2, "elapsed window must re-emit: {events:?}");
    assert_eq!(
        events[1],
        ProgressEvent::Download {
            file: "model.onnx".to_string(),
            bytes_done: 20,
            bytes_total: 1_000,
        }
    );
}

/// Json mode with an unknown (chunked) total: throttled events carry
/// `bytes_total: 0`, and `finish` closes the file with one final event.
#[test]
fn test_download_progress_json_zero_total_emits_final_on_finish() {
    let (sink, log) = ProgressSink::capturing();
    let mut progress = DownloadProgress::new(0);
    progress.update(512, &sink, "model.onnx");
    progress.finish(&sink, "model.onnx");
    let events = log.lock().unwrap();
    assert_eq!(
        events.as_slice(),
        [
            ProgressEvent::Download {
                file: "model.onnx".to_string(),
                bytes_done: 512,
                bytes_total: 0,
            },
            ProgressEvent::Download {
                file: "model.onnx".to_string(),
                bytes_done: 512,
                bytes_total: 0,
            },
        ]
    );
}

/// Human mode never routes events to the sink (the `\r` line is the whole
/// output), so an integrator never sees a stray JSON line.
#[test]
fn test_download_progress_human_sink_emits_no_events() {
    let (sink, log) = ProgressSink::capturing();
    let sink = ProgressSink {
        mode: ProgressMode::Human,
        ..sink
    };
    let mut progress = DownloadProgress::new(1_000);
    progress.update(1_000, &sink, "model.onnx");
    progress.finish(&sink, "model.onnx");
    assert!(log.lock().unwrap().is_empty());
}

#[test]
fn test_classify_download_error_checksum_message() {
    let err = anyhow::anyhow!("SHA-256 mismatch for encoder.onnx: expected aa, got bb");
    assert_eq!(classify_download_error(&err), ProgressErrorKind::Checksum);
}

#[test]
fn test_classify_download_error_http_status_is_network() {
    let err = anyhow::anyhow!("Download failed for model.onnx: HTTP 404");
    assert_eq!(classify_download_error(&err), ProgressErrorKind::Network);
}

#[test]
fn test_classify_download_error_io_root_cause_is_disk() {
    let io = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
    let err = anyhow::Error::from(io).context("Failed to create partial model file");
    assert_eq!(classify_download_error(&err), ProgressErrorKind::Disk);
}

#[test]
fn test_classify_download_error_other() {
    let err = anyhow::anyhow!("Failed to acquire download lock (another process is downloading)");
    assert_eq!(classify_download_error(&err), ProgressErrorKind::Other);
}

/// A real reqwest connection failure classifies as network via the typed
/// root cause, not message matching.
#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_classify_download_error_reqwest_is_network() {
    let err = reqwest::Client::new()
        .get("http://127.0.0.1:9/unreachable")
        .send()
        .await
        .expect_err("port 9 must refuse");
    let err = anyhow::Error::from(err).context("HTTP request failed");
    assert_eq!(classify_download_error(&err), ProgressErrorKind::Network);
}

/// Compute the SHA-256 of a byte slice as a lowercase hex digest.
fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex_lower(&hasher.finalize())
}

/// Helper to stage a `.partial` file with arbitrary bytes, mimicking
/// the state of a fully streamed download prior to verification.
fn stage_partial(final_path: &Path, bytes: &[u8]) -> std::path::PathBuf {
    let partial = partial_path(final_path);
    let mut f = std::fs::File::create(&partial).expect("create partial");
    f.write_all(bytes).expect("write partial");
    f.sync_all().expect("sync partial");
    partial
}

#[test]
fn test_partial_path_appends_suffix() {
    let p = partial_path(Path::new("/tmp/gigastt/encoder.onnx"));
    assert_eq!(
        p,
        std::path::PathBuf::from("/tmp/gigastt/encoder.onnx.partial"),
    );
}

/// On the success path, `.partial` disappears and the final path
/// appears in a single atomic step.
#[test]
fn test_download_writes_partial_then_renames() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let final_path = tmp.path().join("encoder.onnx");
    let payload = b"fake encoder weights";
    let expected = sha256_hex(payload);

    let partial = stage_partial(&final_path, payload);
    assert!(partial.exists(), "precondition: partial is present");
    assert!(!final_path.exists(), "precondition: final is absent");

    finalize_download(&partial, &final_path, Some(&expected), "encoder.onnx")
        .expect("finalize should succeed");

    assert!(
        !partial.exists(),
        "partial must be gone after atomic rename"
    );
    assert!(
        final_path.exists(),
        "final path must exist after atomic rename"
    );
    assert_eq!(std::fs::read(&final_path).unwrap(), payload);
}

/// If the process dies between the network write and the
/// SHA verification / rename, `is_model_present` must NOT see the
/// file under its final name. We simulate the crash by staging a
/// `.partial` and never calling `finalize_download`.
#[test]
fn test_download_crash_before_rename_leaves_no_final_file() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let final_path = tmp.path().join("encoder.onnx");
    let partial = stage_partial(&final_path, b"half-written junk");

    assert!(partial.exists(), "partial must exist to simulate crash");
    assert!(
        !final_path.exists(),
        "crash before rename must never leave the final artefact visible"
    );

    // No variant's files exist in this tempdir, so is_model_present must
    // refuse to short-circuit the download path.
    assert!(
        !is_model_present(ModelVariant::Rnnt, tmp.path()),
        "is_model_present must not accept a staged partial"
    );
    assert!(
        !is_model_present(ModelVariant::E2eRnnt, tmp.path()),
        "is_model_present must not accept a staged partial"
    );
}

/// SHA mismatch removes the partial and leaves the final path
/// empty, so a retry starts from a clean slate.
#[test]
fn test_download_rejects_sha256_mismatch() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let final_path = tmp.path().join("decoder.onnx");
    let payload = b"real bytes";
    // Intentionally wrong expected hash (hash of different bytes).
    let wrong_expected = sha256_hex(b"different bytes");

    let partial = stage_partial(&final_path, payload);

    let err = finalize_download(&partial, &final_path, Some(&wrong_expected), "decoder.onnx")
        .expect_err("mismatch must error");
    let msg = format!("{err}");
    assert!(msg.contains("SHA-256 mismatch"), "unexpected error: {msg}");

    assert!(!partial.exists(), "partial must be removed on SHA mismatch");
    assert!(
        !final_path.exists(),
        "final must never appear on SHA mismatch"
    );
}

/// Success path with no checksum available still renames
/// atomically (partial gone, final present, bytes preserved).
#[test]
fn test_download_atomic_on_success_without_checksum() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let final_path = tmp.path().join("vocab.txt");
    let payload = b"token0\ntoken1\n";

    let partial = stage_partial(&final_path, payload);

    finalize_download(&partial, &final_path, None, "vocab.txt")
        .expect("no-checksum finalize should succeed");

    assert!(!partial.exists(), "partial must be gone after rename");
    assert!(final_path.exists(), "final path must exist");
    assert_eq!(std::fs::read(&final_path).unwrap(), payload);
}

/// sha256_file matches the in-memory hash of the same bytes.
#[test]
fn test_sha256_file_matches_in_memory_hash() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let p = tmp.path().join("blob");
    let payload = b"gigastt-model-bytes";
    std::fs::write(&p, payload).unwrap();

    let got = sha256_file(&p).expect("sha256_file");
    let want = sha256_hex(payload);
    assert_eq!(got, want);
}

/// `SPEAKER_MODEL_SHA256` is a 64-char lowercase hex digest
/// matching the SHA-256 of the upstream `onnx/model.onnx` blob
/// (no accidental truncation / placeholder at compile time).
#[cfg(feature = "diarization")]
#[test]
fn test_speaker_model_sha256_shape() {
    assert_eq!(
        SPEAKER_MODEL_SHA256.len(),
        64,
        "SPEAKER_MODEL_SHA256 must be a 64-char hex digest"
    );
    assert!(
        SPEAKER_MODEL_SHA256
            .chars()
            .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
        "SPEAKER_MODEL_SHA256 must be lowercase hex; got: {SPEAKER_MODEL_SHA256}"
    );
}

/// Mismatching bytes against `SPEAKER_MODEL_SHA256` must delete
/// the partial and refuse to promote it — exercises the full
/// speaker-model finalize contract without touching the network.
#[cfg(feature = "diarization")]
#[test]
fn test_speaker_model_rejects_sha256_mismatch() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let final_path = tmp.path().join(SPEAKER_MODEL_FILE);
    // Definitely not the real speaker-model bytes.
    let partial = stage_partial(&final_path, b"not the real wespeaker weights");

    let err = finalize_download(
        &partial,
        &final_path,
        Some(SPEAKER_MODEL_SHA256),
        SPEAKER_MODEL_FILE,
    )
    .expect_err("speaker mismatch must error");
    assert!(
        format!("{err}").contains("SHA-256 mismatch"),
        "unexpected error: {err}"
    );

    assert!(
        !partial.exists(),
        "partial speaker model must be removed on mismatch"
    );
    assert!(
        !final_path.exists(),
        "final speaker model must never appear on mismatch"
    );
}

/// When the partial bytes DO hash to `SPEAKER_MODEL_SHA256`, the
/// finalize path promotes them atomically. Network-free: we forge a
/// "matching" partial by precomputing the hash of an arbitrary payload
/// and passing it as the expected value.
#[cfg(feature = "diarization")]
#[test]
fn test_speaker_model_partial_promoted_on_match() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let final_path = tmp.path().join(SPEAKER_MODEL_FILE);
    let payload = b"wespeaker-surrogate";
    let expected = sha256_hex(payload);

    let partial = stage_partial(&final_path, payload);

    finalize_download(&partial, &final_path, Some(&expected), SPEAKER_MODEL_FILE)
        .expect("matching partial must promote");

    assert!(!partial.exists());
    assert!(final_path.exists());
    assert_eq!(std::fs::read(&final_path).unwrap(), payload);
}

#[test]
fn test_partial_path_unique_contains_pid_and_timestamp() {
    let p = partial_path_unique(Path::new("/tmp/final.onnx"));
    let s = p.to_string_lossy();
    assert!(s.contains(".partial."));
    assert!(s.contains(&std::process::id().to_string()));
}

#[cfg(unix)]
#[test]
fn test_acquire_download_lock_creates_lock_file() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let lock = acquire_download_lock(tmp.path()).expect("acquire lock");
    assert!(tmp.path().join(".download.lock").exists());
    drop(lock);
}

#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_stream_to_partial_then_finalize_success() {
    let server = wiremock::MockServer::start().await;
    let payload = b"fake model bytes";
    wiremock::Mock::given(wiremock::matchers::method("GET"))
        .and(wiremock::matchers::path("/model.onnx"))
        .respond_with(
            wiremock::ResponseTemplate::new(200)
                .set_body_bytes(payload.as_slice())
                .insert_header("content-length", payload.len().to_string()),
        )
        .mount(&server)
        .await;

    let tmp = tempfile::tempdir().expect("tempdir");
    let final_path = tmp.path().join("model.onnx");
    let url = format!("{}/model.onnx", server.uri());

    stream_to_partial_then_finalize(&url, &final_path, None, "model.onnx")
        .await
        .expect("download should succeed");

    assert!(final_path.exists());
    assert_eq!(std::fs::read(&final_path).unwrap(), payload);
}

#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_stream_to_partial_then_finalize_http_error() {
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("GET"))
        .and(wiremock::matchers::path("/missing.onnx"))
        .respond_with(wiremock::ResponseTemplate::new(404))
        .mount(&server)
        .await;

    let tmp = tempfile::tempdir().expect("tempdir");
    let final_path = tmp.path().join("missing.onnx");
    let url = format!("{}/missing.onnx", server.uri());

    let err = stream_to_partial_then_finalize(&url, &final_path, None, "missing.onnx")
        .await
        .expect_err("404 should fail");
    let msg = format!("{err}");
    assert!(msg.contains("404"), "error should mention 404: {msg}");
}

#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_stream_to_partial_then_finalize_checksum_mismatch() {
    let server = wiremock::MockServer::start().await;
    let payload = b"wrong bytes";
    wiremock::Mock::given(wiremock::matchers::method("GET"))
        .and(wiremock::matchers::path("/model.onnx"))
        .respond_with(wiremock::ResponseTemplate::new(200).set_body_bytes(payload.as_slice()))
        .mount(&server)
        .await;

    let tmp = tempfile::tempdir().expect("tempdir");
    let final_path = tmp.path().join("model.onnx");
    let url = format!("{}/model.onnx", server.uri());
    let wrong_hash = sha256_hex(b"different bytes");

    let err = stream_to_partial_then_finalize(&url, &final_path, Some(&wrong_hash), "model.onnx")
        .await
        .expect_err("checksum mismatch should fail");
    let msg = format!("{err}");
    assert!(
        msg.contains("SHA-256 mismatch"),
        "error should mention mismatch: {msg}"
    );
}

/// The RAII guard removes the staged partial on drop (early return) but
/// leaves it alone once disarmed (successful rename).
#[test]
fn test_partial_file_guard_drop_removes_staged_file() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let partial = tmp.path().join("model.onnx.partial");
    std::fs::write(&partial, b"half-written junk").expect("write partial");

    drop(PartialFileGuard::new(partial.clone()));
    assert!(!partial.exists(), "guard drop must remove the partial");

    std::fs::write(&partial, b"half-written junk").expect("write partial");
    PartialFileGuard::new(partial.clone()).disarm();
    assert!(partial.exists(), "disarmed guard must keep the partial");
}

/// A connection cut mid-stream leaves no orphan `.partial` behind: the
/// guard cleans up the staged bytes so a retry starts clean.
#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_stream_to_partial_then_finalize_stream_error_removes_partial() {
    use tokio::io::AsyncReadExt as _;

    // Hand-rolled stub: hyper (and therefore wiremock) refuses to send a
    // body shorter than the declared content-length, so speak raw HTTP/1.1
    // instead — send the headers, a fraction of the promised body, then
    // close the socket to cut the stream.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind stub server");
    let addr = listener.local_addr().expect("local addr");
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept");
        // Drain the tiny GET request so the response does not race it.
        let mut buf = [0u8; 1024];
        let _ = socket.read(&mut buf).await;
        let head = "HTTP/1.1 200 OK\r\ncontent-length: 4096\r\nconnection: close\r\n\r\n";
        socket.write_all(head.as_bytes()).await.expect("write head");
        socket
            .write_all(b"half of the model")
            .await
            .expect("write body");
        // The socket closes on drop, truncating the promised body.
    });

    let tmp = tempfile::tempdir().expect("tempdir");
    let final_path = tmp.path().join("model.onnx");
    let url = format!("http://{addr}/model.onnx");

    let err = stream_to_partial_then_finalize(&url, &final_path, None, "model.onnx")
        .await
        .expect_err("truncated body should fail");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("Download stream error"),
        "error should come from the stream read: {msg}"
    );
    server.await.expect("stub server task");

    assert!(!final_path.exists(), "final must not appear on failure");
    let orphans: Vec<_> = std::fs::read_dir(tmp.path())
        .expect("read dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().contains(".partial"))
        .collect();
    assert!(orphans.is_empty(), "no orphan .partial files: {orphans:?}");
}

/// End-to-end NDJSON contract on a local HTTP stub: the download of one
/// file emits a 100% `download` event followed by a `verify` event, and
/// every line round-trips through a JSON parser (true NDJSON).
#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_stream_to_partial_then_finalize_json_event_sequence() {
    let server = wiremock::MockServer::start().await;
    let payload = b"fake model bytes";
    wiremock::Mock::given(wiremock::matchers::method("GET"))
        .and(wiremock::matchers::path("/model.onnx"))
        .respond_with(
            wiremock::ResponseTemplate::new(200)
                .set_body_bytes(payload.as_slice())
                .insert_header("content-length", payload.len().to_string()),
        )
        .mount(&server)
        .await;

    let tmp = tempfile::tempdir().expect("tempdir");
    let final_path = tmp.path().join("model.onnx");
    let url = format!("{}/model.onnx", server.uri());
    let expected = sha256_hex(payload);
    let (sink, log) = ProgressSink::capturing();

    stream_to_partial_then_finalize_with_sink(
        &url,
        &final_path,
        Some(&expected),
        "model.onnx",
        &sink,
    )
    .await
    .expect("download should succeed");

    let events = log.lock().unwrap();
    assert_eq!(
        events.as_slice(),
        [
            ProgressEvent::Download {
                file: "model.onnx".to_string(),
                bytes_done: payload.len() as u64,
                bytes_total: payload.len() as u64,
            },
            ProgressEvent::Verify {
                file: "model.onnx".to_string(),
            },
        ]
    );
    // The integrator view: serialize each event to a line and parse it
    // back — the stream must be well-formed NDJSON with a phase tag.
    for event in events.iter() {
        let parsed: serde_json::Value =
            serde_json::from_str(&event.to_ndjson()).expect("event must be NDJSON");
        assert!(parsed.get("phase").is_some());
    }
}

/// No checksum pinned → no `verify` event (verification did not happen).
#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_stream_to_partial_then_finalize_json_no_verify_without_checksum() {
    let server = wiremock::MockServer::start().await;
    let payload = b"fake model bytes";
    wiremock::Mock::given(wiremock::matchers::method("GET"))
        .and(wiremock::matchers::path("/model.onnx"))
        .respond_with(
            wiremock::ResponseTemplate::new(200)
                .set_body_bytes(payload.as_slice())
                .insert_header("content-length", payload.len().to_string()),
        )
        .mount(&server)
        .await;

    let tmp = tempfile::tempdir().expect("tempdir");
    let final_path = tmp.path().join("model.onnx");
    let url = format!("{}/model.onnx", server.uri());
    let (sink, log) = ProgressSink::capturing();

    stream_to_partial_then_finalize_with_sink(&url, &final_path, None, "model.onnx", &sink)
        .await
        .expect("download should succeed");

    let events = log.lock().unwrap();
    assert!(
        events
            .iter()
            .all(|e| !matches!(e, ProgressEvent::Verify { .. })),
        "no checksum -> no verify event: {events:?}"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, ProgressEvent::Download { .. })),
        "download events expected: {events:?}"
    );
}

#[test]
fn test_punct_files_checksums_are_pinned() {
    // Three files, each with a 64-char lowercase hex digest — no truncation
    // or placeholder slipping into a release.
    assert_eq!(PUNCT_FILES.len(), 3);
    for (file, sum) in PUNCT_FILES {
        assert_eq!(
            sum.len(),
            64,
            "{file} punct checksum must be 64 hex chars, got: {sum}"
        );
        assert!(
            sum.chars()
                .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
            "{file} punct checksum must be lowercase hex, got: {sum}"
        );
    }
}

/// `ensure_punct_model` short-circuits (no network, no `.partial`) when all
/// three files are already present.
#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_ensure_punct_model_present_no_download() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path();
    for (file, _) in PUNCT_FILES {
        std::fs::write(dir.join(file), b"stub").unwrap();
    }

    ensure_punct_model(dir.to_str().unwrap())
        .await
        .expect("present model must short-circuit");

    let partials: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().contains(".partial"))
        .collect();
    assert!(partials.is_empty(), "no .partial files: {partials:?}");
    for (file, _) in PUNCT_FILES {
        assert_eq!(std::fs::read(dir.join(file)).unwrap(), b"stub");
    }
}

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

/// Verify that `ensure_model(None, dir)` with a complete E2eRnnt **INT8**
/// install does NOT create any `.partial` files (no download triggered).
#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_ensure_model_none_respects_existing_e2e_install() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path();

    // Stage a full E2eRnnt prequantized INT8 set (stub bytes, no real ONNX).
    for f in ModelVariant::E2eRnnt.prequantized_files() {
        std::fs::write(dir.join(f), b"stub").unwrap();
    }

    // ensure_model_variant with None must return E2eRnnt without downloading.
    let variant = ensure_model_variant(None, dir.to_str().unwrap())
        .await
        .expect("ensure_model_variant should succeed");

    assert_eq!(
        variant,
        ModelVariant::E2eRnnt,
        "must use the installed E2eRnnt"
    );

    // No .partial files must have been created.
    let partials: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().contains(".partial"))
        .collect();
    assert!(
        partials.is_empty(),
        "no .partial files must exist: {partials:?}"
    );

    // Files must be untouched (still stub bytes).
    for f in ModelVariant::E2eRnnt.prequantized_files() {
        assert_eq!(
            std::fs::read(dir.join(f)).unwrap(),
            b"stub",
            "{f} must be unchanged"
        );
    }
}

/// FP32-only E2eRnnt is not usable: ensure falls through to default Rnnt
/// download path would run if the network were hit — we only assert the
/// presence filter rejects FP32-only (no Use short-circuit).
#[test]
fn test_fp32_only_e2e_is_not_usable() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path();
    for f in ModelVariant::E2eRnnt.download_files() {
        std::fs::write(dir.join(f), b"stub").unwrap();
    }
    assert!(
        is_model_present(ModelVariant::E2eRnnt, dir),
        "FP32 download set is complete"
    );
    assert!(
        !is_usable_present(ModelVariant::E2eRnnt, dir),
        "FP32-only must not count as a usable install"
    );
    let existing = ModelVariant::detect_in_dir(dir).filter(|&v| is_usable_present(v, dir));
    assert_eq!(existing, None, "detect+usable filter must ignore FP32-only");
}

/// The legacy public `ensure_model(dir)` wrapper delegates to
/// `ensure_model_variant(None, dir)`: with a complete INT8 install already
/// on disk it must succeed without touching the network (no `.partial` files).
#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_ensure_model_wrapper_uses_existing_install() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path();

    // Stage a complete default (Rnnt) prequantized INT8 set.
    for f in ModelVariant::Rnnt.prequantized_files() {
        std::fs::write(dir.join(f), b"stub").unwrap();
    }

    ensure_model(dir.to_str().unwrap())
        .await
        .expect("ensure_model must succeed against an existing install");

    let partials: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().contains(".partial"))
        .collect();
    assert!(
        partials.is_empty(),
        "ensure_model must not download when the set is present: {partials:?}"
    );
}

/// `ensure_model_variant(Some(Rnnt), dir)` against a matching INT8 install
/// is the `VariantAction::Use` branch: returns Rnnt with no download.
#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_ensure_model_variant_explicit_match_uses_existing() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path();

    for f in ModelVariant::Rnnt.prequantized_files() {
        std::fs::write(dir.join(f), b"stub").unwrap();
    }

    let variant = ensure_model_variant(Some(ModelVariant::Rnnt), dir.to_str().unwrap())
        .await
        .expect("explicit matching variant must short-circuit");
    assert_eq!(variant, ModelVariant::Rnnt);

    let has_partial = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .any(|e| e.file_name().to_string_lossy().contains(".partial"));
    assert!(!has_partial, "no download for an explicit matching variant");
}

/// `ensure_vad_model` short-circuits (no network, no `.partial`) when the
/// Silero ONNX file is already present in the VAD directory.
#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_ensure_vad_model_present_no_download() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path();
    std::fs::write(dir.join(crate::vad::VAD_MODEL_FILE), b"stub vad").unwrap();

    ensure_vad_model(dir.to_str().unwrap())
        .await
        .expect("present VAD model must short-circuit");

    let partials: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().contains(".partial"))
        .collect();
    assert!(partials.is_empty(), "no .partial files: {partials:?}");
    assert_eq!(
        std::fs::read(dir.join(crate::vad::VAD_MODEL_FILE)).unwrap(),
        b"stub vad",
        "existing VAD model must be left untouched"
    );
}

/// `default_punct_model_dir` and `default_vad_model_dir` are siblings of the
/// main model dir under `.gigastt/models`, with the expected leaf names.
#[test]
fn test_default_punct_and_vad_dirs_are_model_siblings() {
    let punct = default_punct_model_dir();
    let vad = default_vad_model_dir();
    assert!(
        punct.contains(".gigastt") && punct.ends_with("punct"),
        "punct dir should be under .gigastt and end with 'punct', got: {punct}"
    );
    assert!(
        vad.contains(".gigastt") && vad.ends_with("vad"),
        "vad dir should be under .gigastt and end with 'vad', got: {vad}"
    );
}

/// `VAD_MODEL_SHA256` is a 64-char lowercase hex digest (no truncation or
/// placeholder slipping into a release).
#[test]
fn test_vad_model_sha256_shape() {
    assert_eq!(
        VAD_MODEL_SHA256.len(),
        64,
        "VAD_MODEL_SHA256 must be a 64-char hex digest"
    );
    assert!(
        VAD_MODEL_SHA256
            .chars()
            .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
        "VAD_MODEL_SHA256 must be lowercase hex; got: {VAD_MODEL_SHA256}"
    );
}

/// `ensure_model_variant` tolerates a deep, freshly-created model directory:
/// a complete Rnnt **INT8** set pre-staged under a nested path is detected
/// and used as-is (early `Use(...)` return) without any network access.
#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_ensure_model_variant_uses_complete_set_in_nested_path() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let nested = tmp.path().join("a").join("b").join("models");
    std::fs::create_dir_all(&nested).unwrap();
    for f in ModelVariant::Rnnt.prequantized_files() {
        std::fs::write(nested.join(f), b"stub").unwrap();
    }

    let variant = ensure_model_variant(None, nested.to_str().unwrap())
        .await
        .expect("nested complete install must be used as-is");
    assert_eq!(variant, ModelVariant::Rnnt);
}

// ── pre-quantized bundle ────────────────────────────────────────────────

#[test]
fn test_prequantized_files_mapping() {
    assert_eq!(
        ModelVariant::Rnnt.prequantized_files(),
        [
            "v3_rnnt_encoder_int8.onnx",
            "v3_rnnt_decoder.onnx",
            "v3_rnnt_joint.onnx",
            "v3_vocab.txt",
        ]
    );
    assert_eq!(
        ModelVariant::E2eRnnt.prequantized_files(),
        [
            "v3_e2e_rnnt_encoder_int8.onnx",
            "v3_e2e_rnnt_decoder.onnx",
            "v3_e2e_rnnt_joint.onnx",
            "v3_e2e_rnnt_vocab.txt",
        ]
    );
    // CTC is encoder-only: pre-quantized set is just the INT8 encoder + vocab.
    assert_eq!(
        ModelVariant::MlCtc.prequantized_files(),
        ["multilingual_ctc.int8.onnx", "multilingual_vocab.txt"]
    );
    assert_eq!(
        ModelVariant::MlCtcLarge.prequantized_files(),
        ["multilingual_large_ctc.int8.onnx", "multilingual_vocab.txt"]
    );
}

#[test]
fn test_encoder_int8_checksums_are_pinned() {
    for variant in [
        ModelVariant::Rnnt,
        ModelVariant::E2eRnnt,
        ModelVariant::MlCtc,
        ModelVariant::MlCtcLarge,
    ] {
        let sum = variant.encoder_int8_checksum();
        assert_eq!(
            sum.len(),
            64,
            "{variant:?} INT8 checksum must be 64 hex chars, got: {sum}"
        );
        assert!(
            sum.chars()
                .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
            "{variant:?} INT8 checksum must be lowercase hex, got: {sum}"
        );
    }
}

#[test]
fn test_prequantized_checksum_int8_encoder_and_reuses_fp32_for_rest() {
    let v = ModelVariant::Rnnt;
    // Encoder → the INT8-specific checksum.
    assert_eq!(
        v.prequantized_checksum(v.encoder_int8_file()),
        Some(v.encoder_int8_checksum())
    );
    // Decoder/joiner/vocab → the same pins as the FP32 download set.
    for f in [v.decoder_file(), v.joint_file(), v.vocab_file()] {
        assert_eq!(v.prequantized_checksum(f), v.checksum(f));
        assert!(v.prequantized_checksum(f).is_some(), "{f} must be pinned");
    }
}

#[test]
fn test_is_prequantized_present() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path();
    assert!(!is_prequantized_present(ModelVariant::Rnnt, dir));
    for f in ModelVariant::Rnnt.prequantized_files() {
        std::fs::write(dir.join(f), b"x").unwrap();
    }
    assert!(is_prequantized_present(ModelVariant::Rnnt, dir));
    // The FP32 set is absent (no FP32 encoder), yet the prequantized set is
    // complete — the two presence checks are independent.
    assert!(!is_model_present(ModelVariant::Rnnt, dir));
}

/// Regression: an INT8-only (prequantized) tree must count as a usable
/// install for ensure/bootstrap — same filter as `ensure_model_variant`.
/// Without `is_usable_present`, serve would re-download the ~844 MB FP32 set.
#[test]
fn test_ensure_filter_accepts_prequantized_only_dir() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path();
    for f in ModelVariant::Rnnt.prequantized_files() {
        std::fs::write(dir.join(f), b"stub").unwrap();
    }
    assert_eq!(
        ModelVariant::detect_in_dir(dir),
        Some(ModelVariant::Rnnt),
        "encoder_int8 alone is enough for detect_in_dir"
    );
    assert!(
        is_prequantized_present(ModelVariant::Rnnt, dir),
        "prequantized set is complete"
    );
    assert!(
        !is_model_present(ModelVariant::Rnnt, dir),
        "FP32 download set must still be absent"
    );
    assert!(
        is_usable_present(ModelVariant::Rnnt, dir),
        "INT8-only set is usable without FP32"
    );
    // Same filter as ensure_model_variant:
    let existing = ModelVariant::detect_in_dir(dir).filter(|&v| is_usable_present(v, dir));
    assert_eq!(
        existing,
        Some(ModelVariant::Rnnt),
        "ensure treats prequantized-only as present"
    );
    assert_eq!(
        resolve_variant(None, existing),
        VariantAction::Use(ModelVariant::Rnnt),
        "usable existing → no download"
    );
}

/// `ensure_model_variant` short-circuits (no network, no `.partial`) when
/// only the pre-quantized INT8 set is on disk.
#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_ensure_model_variant_accepts_prequantized_only() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path();
    for f in ModelVariant::Rnnt.prequantized_files() {
        std::fs::write(dir.join(f), b"stub").unwrap();
    }

    let variant = ensure_model_variant(None, dir.to_str().unwrap())
        .await
        .expect("prequantized-only dir must short-circuit without download");
    assert_eq!(variant, ModelVariant::Rnnt);

    let has_partial = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .any(|e| e.file_name().to_string_lossy().contains(".partial"));
    assert!(
        !has_partial,
        "no download when the prequantized set is present"
    );
    // Still no FP32 encoder on disk.
    assert!(!dir.join(ModelVariant::Rnnt.encoder_file()).exists());
}

/// `ensure_prequantized_model_variant` short-circuits (no network, no
/// `.partial`) when the pre-quantized set is already present.
#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_ensure_prequantized_present_no_download() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path();
    for f in ModelVariant::Rnnt.prequantized_files() {
        std::fs::write(dir.join(f), b"stub").unwrap();
    }

    let variant = ensure_prequantized_model_variant(None, dir.to_str().unwrap())
        .await
        .expect("present prequantized set must short-circuit");
    assert_eq!(variant, ModelVariant::Rnnt);

    let has_partial = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .any(|e| e.file_name().to_string_lossy().contains(".partial"));
    assert!(
        !has_partial,
        "no download when the prequantized set is present"
    );
}

// ── ANE packages ────────────────────────────────────────────────────────

#[cfg(feature = "ane")]
#[test]
fn test_ane_buckets_ladder_pinned() {
    // Must match the convert script's --buckets default.
    assert_eq!(ANE_BUCKETS, &[512, 768, 1536, 3000]);
}

/// Every shipped bucket must clear the ANE-residency floor (~288 mel frames):
/// below it the fixed-shape graph falls off the Neural Engine onto the CPU EP
/// (measured in the conversion spike), so a too-small bucket would silently
/// regress to CPU. 512 (the smallest) clears 288; this guards future ladder
/// edits from adding a bucket below the residency floor.
#[cfg(feature = "ane")]
#[test]
fn test_ane_buckets_above_residency_floor() {
    const ANE_RESIDENCY_FLOOR: usize = 288;
    for &b in ANE_BUCKETS {
        assert!(
            b >= ANE_RESIDENCY_FLOOR,
            "ANE bucket {b} is below the {ANE_RESIDENCY_FLOOR}-mel residency floor — it would evict to CPU"
        );
    }
}

#[cfg(all(feature = "net", feature = "ane"))]
#[test]
fn test_ane_tar_checksums_shape() {
    // Exactly one entry per bucket; each entry is either the empty
    // (unreleased) sentinel or a valid 64-char lowercase-hex digest.
    assert_eq!(ANE_TAR_CHECKSUMS.len(), ANE_BUCKETS.len());
    for &b in ANE_BUCKETS {
        let entries: Vec<_> = ANE_TAR_CHECKSUMS
            .iter()
            .filter(|(bucket, _)| *bucket == b)
            .collect();
        assert_eq!(entries.len(), 1, "exactly one ANE checksum entry for {b}");
        let sum = entries[0].1;
        if sum.is_empty() {
            continue; // genuine unreleased state
        }
        assert_eq!(
            sum.len(),
            64,
            "ANE {b} checksum must be 64 hex chars: {sum}"
        );
        assert!(
            sum.chars()
                .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
            "ANE {b} checksum must be lowercase hex: {sum}"
        );
    }
}

#[cfg(feature = "ane")]
#[test]
fn test_ane_filename_helpers() {
    assert_eq!(ane_package_dir_name(768), "gigaam_v3_encoder_768.mlpackage");
    assert_eq!(ane_tar_name(768), "gigaam_v3_encoder_768.mlpackage.tar");
}

#[cfg(feature = "ane")]
#[test]
fn test_default_ane_model_dir_is_model_sibling() {
    let ane = default_ane_model_dir();
    assert!(
        ane.contains(".gigastt") && ane.ends_with("ane"),
        "ane dir should be under .gigastt and end with 'ane', got: {ane}"
    );
}

/// Stage the FULL structurally-required file set Core ML writes into a
/// `.mlpackage` (manifest + model spec + weights blob) under a bucket dir.
#[cfg(feature = "ane")]
fn stage_complete_ane_package(pkg: &Path) {
    let coreml = pkg.join("Data").join("com.apple.CoreML");
    std::fs::create_dir_all(coreml.join("weights")).unwrap();
    std::fs::write(pkg.join("Manifest.json"), b"{}").unwrap();
    std::fs::write(coreml.join("model.mlmodel"), b"spec").unwrap();
    std::fs::write(coreml.join("weights").join("weight.bin"), b"w").unwrap();
}

#[cfg(feature = "ane")]
#[test]
fn test_is_ane_present_false_on_empty_then_true_when_staged() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path();
    assert!(!is_ane_present(dir), "empty dir has no ANE packages");

    for &b in ANE_BUCKETS {
        stage_complete_ane_package(&dir.join(ane_package_dir_name(b)));
    }
    assert!(is_ane_present(dir), "all buckets fully staged → present");
}

/// A torn package (only `Manifest.json`, no model spec / weights) must NOT
/// be reported complete — otherwise the download path wedges forever.
#[cfg(feature = "ane")]
#[test]
fn test_ane_package_complete_false_when_torn() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path();
    let pkg = dir.join(ane_package_dir_name(768));
    std::fs::create_dir_all(&pkg).unwrap();
    std::fs::write(pkg.join("Manifest.json"), b"{}").unwrap();

    assert!(
        !ane_package_complete(&pkg),
        "manifest-only package is torn, not complete"
    );

    // Stage the other buckets fully; the torn 768 bucket must still drag
    // the whole-dir check to false.
    for &b in &ANE_BUCKETS[1..] {
        stage_complete_ane_package(&dir.join(ane_package_dir_name(b)));
    }
    assert!(!is_ane_present(dir), "torn bucket → not present");
}

/// Build a deterministic `.tar` (a `<pkg_name>/` dir whose arcnames are
/// prefixed with the package name) holding the full required file set,
/// written at `tar_path`. Mirrors what `release-ane.yml` publishes.
#[cfg(feature = "ane")]
fn build_ane_tar(tar_path: &Path, pkg_name: &str) {
    let tmp = tempfile::tempdir().expect("tempdir");
    let pkg = tmp.path().join(pkg_name);
    stage_complete_ane_package(&pkg);

    let file = std::fs::File::create(tar_path).unwrap();
    let mut builder = tar::Builder::new(file);
    builder.append_dir_all(pkg_name, &pkg).unwrap();
    builder.finish().unwrap();
}

/// Building a deterministic tar (a `gigaam_v3_encoder_768.mlpackage/` dir
/// with the full file set) and unpacking it with `tar::Archive` reconstructs
/// the directory + files — proves the extract step end-to-end, no network.
#[cfg(feature = "ane")]
#[test]
fn test_ane_tar_roundtrip_extract() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let tar_path = tmp.path().join("pkg.tar");
    build_ane_tar(&tar_path, "gigaam_v3_encoder_768.mlpackage");

    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let file = std::fs::File::open(&tar_path).unwrap();
    tar::Archive::new(file).unpack(&out).unwrap();

    let extracted = out.join("gigaam_v3_encoder_768.mlpackage");
    assert!(
        ane_package_complete(&extracted),
        "extracted .mlpackage must be complete"
    );
}

/// `extract_ane_tar_atomic` reconstructs the package at its final path and
/// leaves no `.extract.*` staging dir behind on success.
#[cfg(all(feature = "net", feature = "ane"))]
#[test]
fn test_extract_ane_tar_atomic_no_staging_leak() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path();
    let pkg_name = ane_package_dir_name(768);
    let tar_dest = dir.join(ane_tar_name(768));
    build_ane_tar(&tar_dest, &pkg_name);

    extract_ane_tar_atomic(&tar_dest, dir, &pkg_name).expect("atomic extract");

    assert!(
        ane_package_complete(&dir.join(&pkg_name)),
        "package must land complete at its final path"
    );
    let leaked = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .any(|e| e.file_name().to_string_lossy().starts_with(".extract."));
    assert!(
        !leaked,
        "no .extract.* staging dir may remain after success"
    );
}

/// Every shipped bucket resolves to its pinned `.tar` checksum; a bucket with
/// no pin (an empty sentinel, or one outside the ladder) surfaces the
/// actionable "not yet published" bail rather than downloading unverified.
#[cfg(all(feature = "net", feature = "ane"))]
#[test]
fn test_require_ane_tar_checksum_resolves_pinned_and_bails_unpinned() {
    // Each ladder bucket is pinned to its release `.tar` SHA-256.
    for &b in ANE_BUCKETS {
        let sum = require_ane_tar_checksum(b).expect("shipped bucket must be pinned");
        assert_eq!(sum.len(), 64, "checksum must be 64 hex chars, got: {sum}");
    }
    // A bucket with no pin (here: outside the ladder) takes the bail path.
    let err = require_ane_tar_checksum(99_999).expect_err("unpinned bucket must bail");
    assert!(
        format!("{err}").contains("not yet published"),
        "unexpected error: {err}"
    );
}

/// The offline-bundle fetch script duplicates the crate's SHA-256 pins
/// (it must run on a machine without gigastt installed). Silent drift —
/// e.g. re-quantizing the encoder and bumping only the crate constant —
/// would break release builds at tag time; this pins the two sources of
/// truth together in PR CI instead.
#[test]
fn test_fetch_offline_models_script_pins_match_crate_constants() {
    let script_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../scripts/fetch_offline_models.sh");
    let script = std::fs::read_to_string(&script_path).expect("read fetch_offline_models.sh");

    assert!(
        script.contains(PREQUANT_RELEASE_BASE),
        "script must fetch the model bundle from the release the crate pins ({PREQUANT_RELEASE_BASE})"
    );
    assert!(
        script.contains(PUNCT_HF_REPO),
        "script must fetch the punctuation model from the repo the crate pins ({PUNCT_HF_REPO})"
    );

    // Join backslash-continued lines so every `fetch "URL" "DEST" "SHA"`
    // call is a single parseable line.
    let joined = script.replace("\\\n", " ");
    let mut checked = 0usize;
    for line in joined.lines() {
        let line = line.trim();
        if !line.starts_with("fetch ") {
            continue;
        }
        let parts: Vec<&str> = line.split('"').collect();
        assert!(parts.len() >= 6, "unparseable fetch line: {line}");
        let (dest, sha) = (parts[3], parts[5]);
        let file = dest.rsplit('/').next().expect("dest basename");
        let expected = if file == ModelVariant::Rnnt.encoder_int8_file() {
            ModelVariant::Rnnt.encoder_int8_checksum()
        } else if let Some(c) = ModelVariant::Rnnt.checksum(file) {
            c
        } else if let Some((_, c)) = PUNCT_FILES.iter().find(|(f, _)| *f == file) {
            c
        } else {
            panic!("script fetches {file}, which the crate has no pin for");
        };
        assert_eq!(
            sha, expected,
            "SHA-256 pin drift for {file}: script says {sha}, crate says {expected}"
        );
        checked += 1;
    }
    assert!(
        checked >= 7,
        "expected the script to pin at least 7 files, parsed {checked}"
    );
}
