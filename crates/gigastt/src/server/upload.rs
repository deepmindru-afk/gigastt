//! Admission for encoded HTTP uploads, before any body extractor buffers data.

use axum::Json;
use axum::body::Bytes;
use axum::extract::{Request, State};
use axum::http::{Method, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use std::sync::Arc;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

#[derive(Clone)]
pub(crate) struct UploadAdmission {
    slots: Arc<Semaphore>,
    retry_after_secs: u64,
}

impl UploadAdmission {
    pub(crate) fn new(capacity: usize, retry_after_secs: u64) -> Self {
        Self {
            slots: Arc::new(Semaphore::new(capacity.max(1))),
            retry_after_secs,
        }
    }
}

#[derive(Clone)]
pub(crate) struct UploadPermit {
    _permit: Arc<OwnedSemaphorePermit>,
}

impl UploadPermit {
    /// Keep admission attached to every clone/slice of the retained input.
    pub(crate) fn retain(&self, data: Bytes) -> Bytes {
        struct RetainedInput {
            data: Bytes,
            _permit: UploadPermit,
        }
        impl AsRef<[u8]> for RetainedInput {
            fn as_ref(&self) -> &[u8] {
                &self.data
            }
        }
        Bytes::from_owner(RetainedInput {
            data,
            _permit: self.clone(),
        })
    }
}

pub(crate) async fn admit(
    State(admission): State<UploadAdmission>,
    mut request: Request,
    next: Next,
) -> Response {
    if request.method() != Method::POST
        || !matches!(
            request.uri().path(),
            "/v1/transcribe" | "/v1/transcribe/stream" | "/v1/audio/transcriptions" | "/v1/jobs"
        )
    {
        return next.run(request).await;
    }
    let Ok(permit) = admission.slots.clone().try_acquire_owned() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            [(header::RETRY_AFTER, admission.retry_after_secs.to_string())],
            Json(serde_json::json!({
                "error": "Upload capacity exhausted, try again later",
                "code": "upload_busy",
                "retry_after_ms": admission.retry_after_secs.saturating_mul(1000).min(u32::MAX as u64),
            })),
        )
            .into_response();
    };
    let permit = UploadPermit {
        _permit: Arc::new(permit),
    };
    request.extensions_mut().insert(permit.clone());
    let response = next.run(request).await;
    // Body extraction and jobs-store insertion are covered by this owner.
    // Transcription adapters additionally retain an owner in their input Bytes.
    drop(permit);
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::Router;
    use axum::http::StatusCode;
    use axum::routing::{get, post};
    use std::time::Duration;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    async fn server(
        admission: UploadAdmission,
    ) -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
        let mut app = Router::new();
        for path in [
            "/v1/transcribe",
            "/v1/transcribe/stream",
            "/v1/audio/transcriptions",
            "/v1/jobs",
        ] {
            app = app.route(
                path,
                post(|body: Bytes| async move { body.len().to_string() }),
            );
        }
        let app = app
            .route("/health", get(|| async { StatusCode::OK }))
            .route("/ready", get(|| async { StatusCode::OK }))
            .route_layer(axum::middleware::from_fn_with_state(admission, admit));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (address, handle)
    }

    #[tokio::test]
    async fn test_upload_admission_rejects_before_body_poll_on_every_route() {
        let admission = UploadAdmission::new(1, 30);
        let _held = admission.slots.clone().acquire_owned().await.unwrap();
        let (address, server) = server(admission).await;
        for path in [
            "/v1/transcribe",
            "/v1/transcribe/stream",
            "/v1/audio/transcriptions",
            "/v1/jobs",
        ] {
            let mut socket = tokio::net::TcpStream::connect(address).await.unwrap();
            socket.write_all(format!("POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: 1000000\r\nConnection: close\r\n\r\n").as_bytes()).await.unwrap();
            // No body is sent: a response proves admission precedes extraction.
            let mut response = vec![0; 4096];
            let bytes = tokio::time::timeout(Duration::from_secs(1), socket.read(&mut response))
                .await
                .expect("saturated admission must reject without waiting for the body")
                .unwrap();
            let response = String::from_utf8_lossy(&response[..bytes]);
            assert!(response.starts_with("HTTP/1.1 503"), "{response}");
            assert!(response.to_ascii_lowercase().contains("retry-after: 30"));
        }
        for path in ["/health", "/ready"] {
            assert_eq!(
                reqwest::get(format!("http://{address}{path}"))
                    .await
                    .unwrap()
                    .status(),
                StatusCode::OK
            );
        }
        server.abort();
    }

    async fn wait_for_slots(admission: &UploadAdmission, count: usize) {
        tokio::time::timeout(Duration::from_secs(2), async {
            while admission.slots.available_permits() != count {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_upload_admission_releases_on_disconnect_and_rejected_body() {
        let admission = UploadAdmission::new(1, 30);
        let (address, server) = server(admission.clone()).await;
        let mut socket = tokio::net::TcpStream::connect(address).await.unwrap();
        socket
            .write_all(
                b"POST /v1/transcribe HTTP/1.1\r\nHost: localhost\r\nContent-Length: 1000\r\n\r\na",
            )
            .await
            .unwrap();
        wait_for_slots(&admission, 0).await;
        drop(socket);
        wait_for_slots(&admission, 1).await;

        let response = reqwest::Client::new()
            .post(format!("http://{address}/v1/transcribe"))
            .body(vec![0u8; 3_000_000])
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        wait_for_slots(&admission, 1).await;

        let mut socket = tokio::net::TcpStream::connect(address).await.unwrap();
        socket.write_all(b"POST /v1/transcribe HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\ninvalid\r\n").await.unwrap();
        let mut response = vec![0; 4096];
        let bytes = tokio::time::timeout(Duration::from_secs(2), socket.read(&mut response))
            .await
            .unwrap()
            .unwrap();
        let response = String::from_utf8_lossy(&response[..bytes]);
        assert!(response.starts_with("HTTP/1.1 4"), "{response}");
        wait_for_slots(&admission, 1).await;
        server.abort();
    }

    #[tokio::test]
    async fn test_upload_admission_jobs_transfer_to_store_accounting() {
        let admission = UploadAdmission::new(1, 30);
        let stored = Arc::new(std::sync::Mutex::new(Vec::new()));
        let store = stored.clone();
        let app = Router::new()
            .route(
                "/v1/jobs",
                post(move |body: Bytes| {
                    let store = store.clone();
                    async move {
                        store.lock().unwrap().push(body);
                        StatusCode::ACCEPTED
                    }
                }),
            )
            .route_layer(axum::middleware::from_fn_with_state(
                admission.clone(),
                admit,
            ));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let client = reqwest::Client::new();
        for _ in 0..3 {
            assert_eq!(
                client
                    .post(format!("http://{address}/v1/jobs"))
                    .body("input")
                    .send()
                    .await
                    .unwrap()
                    .status(),
                StatusCode::ACCEPTED
            );
            wait_for_slots(&admission, 1).await;
        }
        assert_eq!(stored.lock().unwrap().len(), 3);
        server.abort();
    }

    #[tokio::test]
    async fn test_upload_admission_timeout_keeps_detached_worker_accounted() {
        let slots = Arc::new(Semaphore::new(1));
        let permit = UploadPermit {
            _permit: Arc::new(slots.clone().acquire_owned().await.unwrap()),
        };
        let body = permit.retain(Bytes::from_static(b"encoded input"));
        drop(permit);
        let (release, gate) = std::sync::mpsc::channel();
        let worker = tokio::task::spawn_blocking(move || {
            let _upload_lifetime = body.clone();
            let _prepared = body.to_vec();
            drop(body);
            let _ = gate.recv_timeout(Duration::from_secs(2));
        });
        assert!(
            tokio::time::timeout(Duration::from_millis(10), worker)
                .await
                .is_err()
        );
        assert_eq!(slots.available_permits(), 0);
        release.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(2), async {
            while slots.available_permits() == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(slots.available_permits(), 1);
    }

    #[tokio::test]
    async fn test_upload_permit_outlives_transformed_input_until_worker_exits() {
        let slots = Arc::new(Semaphore::new(1));
        let permit = UploadPermit {
            _permit: Arc::new(slots.clone().acquire_owned().await.unwrap()),
        };
        let body = permit.retain(Bytes::from_static(b"encoded input"));
        let worker_lifetime = body.clone();
        let prepared = body.to_vec();
        drop(body);
        drop(permit);
        assert_eq!(slots.available_permits(), 0);
        drop(prepared);
        assert_eq!(slots.available_permits(), 0);
        drop(worker_lifetime);
        assert_eq!(slots.available_permits(), 1);
    }
}
