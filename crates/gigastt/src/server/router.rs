//! Protected `/v1` route table (REST + WebSocket + optional jobs).

use axum::Router;
use axum::http::StatusCode;
use axum::routing::{delete, get, options, post};
use std::sync::Arc;

use super::http;
use super::ws;

/// Routes gated by origin middleware (and the per-IP limiter when enabled).
/// `/metrics` is intentionally absent — it lives on the loopback listener.
pub(crate) fn protected_v1_router(jobs_enabled: bool) -> Router<Arc<http::AppState>> {
    let protected = Router::new()
        .route("/v1/models", get(http::models))
        .route("/v1/models", options(|| async { StatusCode::NO_CONTENT }))
        .route("/v1/transcribe", post(http::transcribe))
        .route(
            "/v1/transcribe",
            options(|| async { StatusCode::NO_CONTENT }),
        )
        .route("/v1/transcribe/stream", post(http::transcribe_stream))
        .route(
            "/v1/transcribe/stream",
            options(|| async { StatusCode::NO_CONTENT }),
        )
        // OpenAI-compatible alias for clients (llama-swap, Hermes Agent, SDKs
        // with a custom base_url) that POST multipart `file` + `model`.
        .route(
            "/v1/audio/transcriptions",
            post(http::openai_transcriptions),
        )
        .route(
            "/v1/audio/transcriptions",
            options(|| async { StatusCode::NO_CONTENT }),
        )
        // /v1/ws is the canonical WebSocket path (versioned, aligned with REST).
        .route("/v1/ws", get(ws::ws_handler))
        .route("/v1/ws", options(|| async { StatusCode::NO_CONTENT }))
        // Admin: hot-reload the model without a restart. Registered inside the
        // protected router so it inherits `origin_middleware`, but the handler
        // additionally enforces a strict loopback peer check (see `http::reload`)
        // so it stays local even under `--bind-all` / `--cors-allow-any`.
        .route("/v1/admin/reload", post(http::reload))
        .route(
            "/v1/admin/reload",
            options(|| async { StatusCode::NO_CONTENT }),
        );

    // Asynchronous job API routes. Only registered when `--enable-jobs` is set;
    // without the flag the paths fall through to axum's default 404.
    if jobs_enabled {
        protected
            .route("/v1/jobs", post(http::submit_job))
            .route("/v1/jobs", options(|| async { StatusCode::NO_CONTENT }))
            .route("/v1/jobs/{id}", get(http::get_job))
            .route("/v1/jobs/{id}", delete(http::cancel_job))
            .route(
                "/v1/jobs/{id}",
                options(|| async { StatusCode::NO_CONTENT }),
            )
            .route("/v1/jobs/{id}/result", get(http::get_job_result))
            .route(
                "/v1/jobs/{id}/result",
                options(|| async { StatusCode::NO_CONTENT }),
            )
            .route("/v1/jobs/{id}/events", get(http::job_events))
    } else {
        protected
    }
}
