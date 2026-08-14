use super::*;

#[test]
fn test_peer_is_loopback_guard() {
    use std::net::{Ipv4Addr, Ipv6Addr, SocketAddr};
    // IPv4 + IPv6 loopback are accepted regardless of source port.
    assert!(peer_is_loopback(&SocketAddr::from((
        Ipv4Addr::LOCALHOST,
        5000
    ))));
    assert!(peer_is_loopback(&SocketAddr::from((
        Ipv6Addr::LOCALHOST,
        5000
    ))));
    // A non-loopback peer (LAN / public) is rejected — reload must stay local
    // even under --bind-all / --cors-allow-any.
    assert!(!peer_is_loopback(&SocketAddr::from((
        Ipv4Addr::new(192, 168, 1, 10),
        9876
    ))));
    assert!(!peer_is_loopback(&SocketAddr::from((
        Ipv4Addr::new(10, 0, 0, 1),
        9876
    ))));
    assert!(!peer_is_loopback(&SocketAddr::from((
        Ipv4Addr::new(8, 8, 8, 8),
        443
    ))));
}

#[tokio::test]
async fn test_reload_rejects_non_loopback_peer() {
    // The loopback guard fires before any engine work: a non-loopback
    // ConnectInfo yields 403 `loopback_only` even with a builder present.
    // Model-gated only because `AppState` needs a concrete `Engine`; the
    // pure guard logic is covered model-free by `test_peer_is_loopback_guard`.
    use std::net::{Ipv4Addr, SocketAddr};
    let state = Arc::new(AppState {
        engine: engine_swap(test_engine()),
        engine_builder: Some(Arc::new(|| {
            anyhow::bail!("builder must not run for a rejected peer")
        })),
        reload_lock: Arc::new(tokio::sync::Mutex::new(())),
        limits: Arc::new(ArcSwap::from_pointee(RuntimeLimits::default())),
        metrics_registry: None,
        shutdown: tokio_util::sync::CancellationToken::new(),
        tracker: tokio_util::task::TaskTracker::new(),
        jobs: None,
    });
    let peer = SocketAddr::from((Ipv4Addr::new(203, 0, 113, 7), 40000));
    let resp = reload(
        axum::extract::ConnectInfo(peer),
        axum::extract::Query(super::super::admin::ReloadQuery::default()),
        State(state),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    let bytes = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["code"], "loopback_only");
}

#[tokio::test]
async fn test_reload_unsupported_when_no_builder() {
    // A loopback peer with no stored builder (the thin `run_with_shutdown` /
    // test path) gets `reload_unsupported`, not a swap.
    use std::net::{Ipv4Addr, SocketAddr};
    let state = Arc::new(AppState {
        engine: engine_swap(test_engine()),
        engine_builder: None,
        reload_lock: Arc::new(tokio::sync::Mutex::new(())),
        limits: Arc::new(ArcSwap::from_pointee(RuntimeLimits::default())),
        metrics_registry: None,
        shutdown: tokio_util::sync::CancellationToken::new(),
        tracker: tokio_util::task::TaskTracker::new(),
        jobs: None,
    });
    let peer = SocketAddr::from((Ipv4Addr::LOCALHOST, 40000));
    let resp = reload(
        axum::extract::ConnectInfo(peer),
        axum::extract::Query(super::super::admin::ReloadQuery::default()),
        State(state),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let bytes = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["code"], "reload_unsupported");
}
