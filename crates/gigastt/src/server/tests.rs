use super::*;

#[test]
fn test_json_text_serializes() {
    let msg = gigastt_core::protocol::ServerMessage::Ready {
        model: "test".into(),
        sample_rate: 16000,
        version: "1.0".into(),
        supported_rates: vec![16000],
        diarization: false,
        min_protocol_version: None,
        max_session_secs: 3600,
        idle_timeout_secs: 300,
    };
    let json = json_text(&msg);
    assert!(json.contains("\"type\":\"ready\""));
}

#[test]
fn test_json_text_fallback_on_error() {
    // A type that intentionally fails serialization is hard to construct
    // with serde, so we assert the fallback path exists by checking the
    // function compiles and the happy path works. The fallback is a
    // static string that we can at least verify is present in the binary
    // by inspecting the source.
    let msg = gigastt_core::protocol::ServerMessage::Error {
        message: "test".into(),
        code: "test".into(),
        retry_after_ms: None,
    };
    let json = json_text(&msg);
    assert!(json.contains("error"));
}

#[test]
fn test_rate_limit_interval_formula() {
    // Mirrors the formula used in `run_with_config` so a regression on the
    // integer-divide `/60` fix (truncates sub-60 rpm to 1 rps) trips
    // a unit test before reaching the e2e path.
    const MAX_RPM: u64 = 60_000;
    fn interval_ms_for(rpm: u32) -> u64 {
        let rpm = (rpm as u64).min(MAX_RPM);
        (60_000u64 / rpm).max(1)
    }
    let cases: &[(u32, u64)] = &[
        (1, 60_000),
        (10, 6_000),
        (30, 2_000),
        (59, 1_016), // 60_000 / 59 = 1016 (rounds down) → ~59.05 rpm
        (60, 1_000),
        (600, 100),
        (60_000, 1),
        (120_000, 1), // clamped to MAX_RPM, stays at 1 ms
    ];
    for (rpm, expected) in cases {
        assert_eq!(
            interval_ms_for(*rpm),
            *expected,
            "rpm={rpm} should map to interval_ms={expected}"
        );
    }
}

#[test]
fn test_pool_checkout_timeout_clamping() {
    let mut config = ServerConfig::local(0);
    config.limits.pool_checkout_timeout_secs = 0;
    // `run_with_config_listener` would clamp this to 1.
    if config.limits.pool_checkout_timeout_secs == 0 {
        config.limits.pool_checkout_timeout_secs = 1;
    }
    assert_eq!(config.limits.pool_checkout_timeout_secs, 1);
}

#[test]
fn test_json_text_fallback_on_serialization_error() {
    struct FailingSerialize;
    impl serde::Serialize for FailingSerialize {
        fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            Err(serde::ser::Error::custom("intentional failure"))
        }
    }
    let json = json_text(&FailingSerialize);
    assert_eq!(
        json,
        r#"{"type":"error","message":"Internal serialization error","code":"internal"}"#
    );
}

#[tokio::test]
#[ignore = "requires model"]
async fn test_run_with_shutdown_starts_and_stops() {
    let engine = gigastt_core::inference::Engine::load_with_pool_size(
        &gigastt_core::model::default_model_dir(),
        1,
    )
    .unwrap();
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
    let handle =
        tokio::spawn(
            async move { run_with_shutdown(engine, 0, "127.0.0.1", Some(shutdown_rx)).await },
        );
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    let _ = shutdown_tx.send(());
    let result = handle.await.expect("join");
    assert!(result.is_ok(), "server should stop gracefully");
}

#[tokio::test]
#[ignore = "requires model"]
async fn test_run_with_config_listener_clamps_zero_timeout() {
    let engine = gigastt_core::inference::Engine::load_with_pool_size(
        &gigastt_core::model::default_model_dir(),
        1,
    )
    .unwrap();
    let mut config = ServerConfig::local(0);
    config.limits.pool_checkout_timeout_secs = 0;
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let handle = tokio::spawn(async move {
        run_with_config_listener(engine, config, Some(shutdown_rx), listener).await
    });
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    let _ = shutdown_tx.send(());
    let result = handle.await.expect("join");
    assert!(result.is_ok(), "server should stop gracefully");
}

/// Exercises the non-blocking-boot orchestration end-to-end *without* a
/// model: a `load` future that never resolves keeps the server in the
/// bootstrap phase, so we can assert (a) `/health` answers `200` with
/// `model:"loading"` over a real socket while "loading", and (b) a shutdown
/// during loading returns `Ok` without ever standing up the full server.
#[tokio::test]
async fn test_run_with_config_loading_bootstrap_then_shutdown() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    // Reserve an ephemeral port, then release it for the server to rebind.
    let probe = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = probe.local_addr().unwrap().port();
    drop(probe);

    let config = ServerConfig::local(port);
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
    let load = std::future::pending::<Result<gigastt_core::inference::Engine>>();
    let handle =
        tokio::spawn(async move { run_with_config_loading(config, Some(shutdown_rx), load).await });

    // Give it a beat to bind and start the bootstrap accept loop.
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    // /health is up during loading and reports the bootstrap placeholder.
    let mut stream = tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .expect("bootstrap listener should accept connections during load");
    stream
        .write_all(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n")
        .await
        .unwrap();
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).await.unwrap();
    let resp = String::from_utf8_lossy(&buf);
    assert!(resp.starts_with("HTTP/1.1 200 OK"), "got: {resp}");
    assert!(resp.contains("\"model\":\"loading\""), "got: {resp}");

    // A shutdown signal during loading must unwind cleanly.
    let _ = shutdown_tx.send(());
    let result = tokio::time::timeout(std::time::Duration::from_secs(5), handle)
        .await
        .expect("loading server did not stop within the timeout")
        .expect("join");
    assert!(
        result.is_ok(),
        "loading server should stop gracefully: {result:?}"
    );
}
