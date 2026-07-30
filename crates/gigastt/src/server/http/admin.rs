//! Administrative endpoints (model hot-reload).

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use serde::Deserialize;
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::state::AppState;

/// Whether the reload endpoint should accept a request from `peer`.
///
/// Model reload is an administrative, machine-local action: it must stay
/// reachable only from the loopback interface even under `--bind-all` /
/// `--cors-allow-any`, which would otherwise widen `origin_middleware` (the only
/// other gate). Pure so the loopback decision can be unit-tested without a model
/// or a live socket.
pub(super) fn peer_is_loopback(peer: &std::net::SocketAddr) -> bool {
    peer.ip().is_loopback()
}

/// Query parameters for `POST /v1/admin/reload`.
#[derive(Debug, Default, Deserialize)]
pub struct ReloadQuery {
    /// Soft reload: swap the new engine in **before** warmup, then wait (up to
    /// a few seconds) for the previous engine's in-flight holders to release
    /// so warmup does not stack on a still-resident old engine. Peak during
    /// the build phase can still approach ~2× ready; soft targets the
    /// warm+old double stack on edge hosts.
    #[serde(default)]
    pub soft: bool,
}

/// Wait until `arc` is the only strong reference left (or `timeout` elapses).
/// Used by soft reload so warmup runs after the previous engine has been freed
/// when no in-flight sessions hold it.
fn wait_for_unique_arc<T>(arc: &Arc<T>, timeout: Duration) -> bool {
    let start = Instant::now();
    while Arc::strong_count(arc) > 1 {
        if start.elapsed() >= timeout {
            return false;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    true
}

/// POST /v1/admin/reload — rebuild the inference engine from the boot recipe and
/// atomically swap it in without a restart.
///
/// Strictly loopback-only (checked here, not just via the origin middleware),
/// serialized by a mutex so two reloads can't race, and fail-safe: a build error
/// leaves the currently-serving engine untouched.
///
/// Order is **build → swap → warm** (not warm-before-swap): once the swap
/// completes, the previous engine can drop as soon as in-flight work finishes,
/// so the warm peak is not forced to stack on a still-serving old copy.
/// `?soft=true` waits briefly for that drop before warming.
///
/// In-flight requests keep the engine they started on and finish against its
/// pool; the old engine drops when its last reference goes.
pub async fn reload(
    axum::extract::ConnectInfo(peer): axum::extract::ConnectInfo<std::net::SocketAddr>,
    Query(query): Query<ReloadQuery>,
    State(state): State<Arc<AppState>>,
) -> Response {
    // Gotcha #2: enforce loopback here so reload stays local even when
    // `origin_middleware` has been widened by `--bind-all` / `--cors-allow-any`
    // or a caller omits the Origin header.
    if !peer_is_loopback(&peer) {
        tracing::warn!(peer = %peer, "Rejecting non-loopback model reload request");
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({
                "error": "Model reload is only available from loopback",
                "code": "loopback_only",
            })),
        )
            .into_response();
    }

    let Some(builder) = state.engine_builder.clone() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": "Model reload is not available on this server",
                "code": "reload_unsupported",
            })),
        )
            .into_response();
    };

    // Serialize reloads: the loser of the race gets 409 rather than queueing, so
    // an operator hammering the endpoint can't stack up concurrent rebuilds.
    let _reload_guard = match state.reload_lock.try_lock() {
        Ok(guard) => guard,
        Err(_) => {
            return (
                StatusCode::CONFLICT,
                Json(serde_json::json!({
                    "error": "A model reload is already in progress",
                    "code": "reload_in_progress",
                })),
            )
                .into_response();
        }
    };

    tracing::info!(
        peer = %peer,
        soft = query.soft,
        "Model reload requested — rebuilding engine"
    );

    // Build the new engine off the request path (ONNX session load is blocking).
    let build = tokio::task::spawn_blocking(move || builder()).await;

    let new_engine = match build {
        Ok(Ok(engine)) => engine,
        Ok(Err(e)) => {
            // Keep the old engine untouched. Log the detail, return a sanitized
            // message (no path / model leakage) matching the internal-error policy.
            tracing::error!("Model reload build failed: {e:#}");
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": "Model reload failed; the previous model is still serving",
                    "code": "reload_failed",
                })),
            )
                .into_response();
        }
        Err(join_err) => {
            tracing::error!("Model reload build task panicked: {join_err}");
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": "Model reload failed; the previous model is still serving",
                    "code": "reload_failed",
                })),
            )
                .into_response();
        }
    };

    let variant = new_engine.variant();
    let encoder = if new_engine.is_int8() { "int8" } else { "fp32" };

    // Pin the previous engine so soft mode can wait for in-flight holders to
    // release it after ArcSwap no longer references it.
    let previous = state.engine.load_full();
    let new_arc = Arc::new(new_engine);
    state.engine.store(new_arc.clone());
    tracing::info!(
        variant = variant.as_str(),
        encoder,
        soft = query.soft,
        "Model reloaded and swapped (warmup next)"
    );

    let soft = query.soft;
    let soft_drained = tokio::task::spawn_blocking(move || {
        if soft {
            let ok = wait_for_unique_arc(&previous, Duration::from_secs(5));
            if !ok {
                tracing::warn!(
                    holders = Arc::strong_count(&previous),
                    "soft reload: previous engine still held after 5s; warming anyway"
                );
            }
            ok
        } else {
            drop(previous);
            true
        }
    })
    .await
    .unwrap_or(true);

    // Warm the live engine after swap (and after soft drain when requested).
    // A warmup failure is non-fatal (mirrors boot): the engine already fell
    // back to CPU internally when needed.
    let warm_arc = new_arc.clone();
    if let Err(join_err) = tokio::task::spawn_blocking(move || {
        if let Err(e) = warm_arc.warmup() {
            tracing::warn!("Reloaded engine warmup failed (already serving): {e:#}");
        }
    })
    .await
    {
        tracing::error!("Model reload warmup task panicked: {join_err}");
        // Engine is already swapped and serving; still report success.
    }

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "reloaded": true,
            "variant": variant.as_str(),
            "encoder": encoder,
            "soft": query.soft,
            "soft_drained": soft_drained,
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wait_for_unique_arc_already_unique() {
        let a = Arc::new(7u32);
        assert!(wait_for_unique_arc(&a, Duration::from_millis(50)));
    }

    #[test]
    fn test_wait_for_unique_arc_timeout_when_shared() {
        let a = Arc::new(7u32);
        let _b = Arc::clone(&a);
        assert!(!wait_for_unique_arc(&a, Duration::from_millis(30)));
    }
}
