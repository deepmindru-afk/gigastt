use super::*;

#[test]
fn test_metric_path_label_bounds_cardinality() {
    // Known routes map to themselves.
    for known in [
        "/health",
        "/ready",
        "/v1/models",
        "/v1/transcribe",
        "/v1/transcribe/stream",
        "/v1/audio/transcriptions",
        "/v1/ws",
        "/metrics",
    ] {
        assert_eq!(metric_path_label(known), known);
    }
    // Anything else collapses to a single bounded label.
    assert_eq!(metric_path_label("/wp-login.php"), "other");
    assert_eq!(metric_path_label("/v1/transcribe/../etc"), "other");
    assert_eq!(metric_path_label("/"), "other");
    assert_eq!(metric_path_label("/v1/models/"), "other");
}

#[tokio::test]
async fn test_request_id_middleware_generates_id() {
    use axum::Router;
    use axum::routing::get;

    let app = Router::new()
        .route("/test", get(|| async { "ok" }))
        .layer(axum::middleware::from_fn(super::request_id_middleware));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://127.0.0.1:{port}/test"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let rid = resp
        .headers()
        .get("x-request-id")
        .expect("missing X-Request-Id");
    let rid_str = rid.to_str().unwrap();
    assert!(
        uuid::Uuid::parse_str(rid_str).is_ok(),
        "X-Request-Id must be valid UUID"
    );
}

#[tokio::test]
async fn test_request_id_middleware_echoes_client_id() {
    use axum::Router;
    use axum::routing::get;

    let app = Router::new()
        .route("/test", get(|| async { "ok" }))
        .layer(axum::middleware::from_fn(super::request_id_middleware));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let client_id = "my-custom-request-id-123";
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://127.0.0.1:{port}/test"))
        .header("X-Request-Id", client_id)
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.headers()
            .get("x-request-id")
            .unwrap()
            .to_str()
            .unwrap(),
        client_id
    );
}

#[tokio::test]
async fn test_origin_middleware_integration() {
    // End-to-end check of the origin_middleware layer on a minimal
    // router. Uses real axum::serve + reqwest to catch regressions that
    // unit tests on `OriginPolicy` alone would miss — e.g. the middleware
    // attaching to the wrong routes, or `/health` accidentally being
    // guarded.
    use axum::Router;
    use axum::routing::get;

    let policy = Arc::new(OriginPolicy::loopback_only());
    let origin_layer = {
        let policy = policy.clone();
        axum::middleware::from_fn(move |req, next| {
            let policy = policy.clone();
            async move { origin_middleware(policy, req, next).await }
        })
    };
    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/transcribe", get(|| async { "ok" }))
        .layer(origin_layer);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    // Allow the server to bind.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let client = reqwest::Client::new();
    let base = format!("http://127.0.0.1:{port}");

    // /health is exempt — monitoring probes work even when Origin is set.
    let r = client
        .get(format!("{base}/health"))
        .header("Origin", "https://evil.example.com")
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "/health must skip the Origin guard");

    // Cross-origin request must be denied on /v1/*.
    let r = client
        .get(format!("{base}/v1/transcribe"))
        .header("Origin", "https://evil.example.com")
        .send()
        .await
        .unwrap();
    assert_eq!(
        r.status(),
        403,
        "non-loopback Origin must receive 403 Forbidden"
    );
    let text = r.text().await.unwrap();
    let body: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(body["code"], "origin_denied");

    // Loopback origin is always allowed.
    let r = client
        .get(format!("{base}/v1/transcribe"))
        .header("Origin", "http://localhost:3000")
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "loopback Origin must be allowed");
    assert_eq!(
        r.headers()
            .get("access-control-allow-origin")
            .and_then(|v| v.to_str().ok()),
        Some("http://localhost:3000"),
        "CORS echo must mirror the incoming Origin (no wildcard by default)",
    );

    // No Origin header (curl, CLI, server-to-server) — policy allows
    // through without a CORS echo.
    let r = client
        .get(format!("{base}/v1/transcribe"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "requests without Origin must pass");

    // Attacker trying DNS continuation on the loopback prefix must be denied.
    let r = client
        .get(format!("{base}/v1/transcribe"))
        .header("Origin", "http://localhost.evil.example.com")
        .send()
        .await
        .unwrap();
    assert_eq!(
        r.status(),
        403,
        "localhost.* DNS continuation must not impersonate loopback"
    );
}
