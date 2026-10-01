//! Exercise the real file-stream producers with an unread HTTP response body.

use super::*;
use axum::body::Body;
use axum::extract::{FromRequest, Multipart};
use axum::http::Request;
use axum::response::IntoResponse;
use gigastt_core::runtime_api::{
    Runtime, RuntimeError, RuntimeFactory, RuntimeSession, Shape, Tensor, TensorData,
};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Each encoder window yields one word followed by enough blanks to finalize
/// an utterance. Both native SSE and OpenAI consequently publish one event per
/// window. The counter lets tests stop exactly when the next send hits a full
/// response queue, without sleeps or assumptions about inference speed.
#[derive(Clone, Default)]
struct TalkativeFactory {
    windows: Arc<AtomicUsize>,
    completed: Arc<tokio::sync::Notify>,
}

impl RuntimeFactory for TalkativeFactory {
    fn create(&self, threads: usize) -> Result<Box<dyn Runtime>, RuntimeError> {
        Ok(Box::new(TalkativeRuntime {
            inner: gigastt_core::test_support::rnnt_factory().create(threads)?,
            factory: self.clone(),
        }))
    }
    fn cpu_fallback(&self) -> Box<dyn RuntimeFactory> {
        Box::new(self.clone())
    }
    fn verify_on_disk_checksums(&self) -> bool {
        false
    }
}

struct TalkativeRuntime {
    inner: Box<dyn Runtime>,
    factory: TalkativeFactory,
}
impl Runtime for TalkativeRuntime {
    fn load_session(
        &self,
        path: &std::path::Path,
        encoder: bool,
    ) -> Result<Box<dyn RuntimeSession>, RuntimeError> {
        if encoder {
            return Ok(Box::new(Encoder));
        }
        if path.file_stem().unwrap() == "v3_rnnt_joint" {
            return Ok(Box::new(Joiner {
                factory: self.factory.clone(),
                emitted: AtomicBool::new(false),
            }));
        }
        self.inner.load_session(path, encoder)
    }
}

struct Encoder;
impl RuntimeSession for Encoder {
    fn run(&self, _inputs: &[Tensor]) -> Result<Vec<Tensor>, RuntimeError> {
        let mut frames = vec![0.0; 768 * 96];
        for (index, frame) in frames[..96].iter_mut().enumerate() {
            *frame = index as f32;
        }
        Ok(vec![
            Tensor::new(Shape::new(vec![1, 768, 96]), TensorData::F32(frames)).unwrap(),
            Tensor::new(Shape::new(vec![1]), TensorData::I64(vec![96])).unwrap(),
        ])
    }
}
struct Joiner {
    factory: TalkativeFactory,
    emitted: AtomicBool,
}
impl RuntimeSession for Joiner {
    fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, RuntimeError> {
        let frame = inputs[0].view().data().as_f32().unwrap()[0] as usize;
        let word = frame == 50 && !self.emitted.swap(true, Ordering::Relaxed);
        if frame != 50 {
            self.emitted.store(false, Ordering::Relaxed);
        }
        if frame == 95 {
            self.factory.windows.fetch_add(1, Ordering::Relaxed);
            self.factory.completed.notify_one();
        }
        Ok(vec![
            Tensor::new(
                Shape::new(vec![1, 1, 2]),
                TensorData::F32(if word {
                    vec![10.0, 0.0]
                } else {
                    vec![0.0, 10.0]
                }),
            )
            .unwrap(),
        ])
    }
}

#[tokio::test]
async fn test_unread_file_stream_responses_release_pool_on_shutdown_or_disconnect() {
    for openai in [false, true] {
        for disconnect in [false, true] {
            let tmp = tempfile::tempdir().unwrap();
            gigastt_core::test_support::write_rnnt_layout(tmp.path()).unwrap();
            let factory = TalkativeFactory::default();
            let engine = Arc::new(
                gigastt_core::test_support::load_rnnt_engine_with_factory(
                    tmp.path(),
                    1,
                    Box::new(factory.clone()),
                )
                .unwrap(),
            );
            factory.windows.store(0, Ordering::Relaxed);
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
            let wav = gigastt_core::test_support::pcm16_wav(&vec![0; 16_000 * 50], 16_000);
            let (response, capacity) = if openai {
                let mut body = b"--test\r\nContent-Disposition: form-data; name=\"stream\"\r\n\r\ntrue\r\n--test\r\nContent-Disposition: form-data; name=\"file\"; filename=\"test.wav\"\r\nContent-Type: audio/wav\r\n\r\n".to_vec();
                body.extend(wav);
                body.extend(b"\r\n--test--\r\n");
                let request = Request::post("/")
                    .header("content-type", "multipart/form-data; boundary=test")
                    .body(Body::from(body))
                    .unwrap();
                let multipart = Multipart::from_request(request, &()).await.unwrap();
                (
                    openai_transcriptions(State(state.clone()), multipart)
                        .await
                        .unwrap(),
                    32,
                )
            } else {
                (
                    transcribe_stream(
                        State(state.clone()),
                        Query(Default::default()),
                        Bytes::from(wav),
                    )
                    .await
                    .unwrap()
                    .into_response(),
                    16,
                )
            };
            assert_eq!(response.status(), StatusCode::OK);
            // Keep the body connected but unpolled, exactly at the transport's
            // backpressure boundary. One extra window attempts the blocked send.
            tokio::time::timeout(std::time::Duration::from_secs(5), async {
                while factory.windows.load(Ordering::Relaxed) < capacity + 1 {
                    factory.completed.notified().await;
                }
            })
            .await
            .expect("producer filled the response queue");
            assert_eq!(engine.pool_for_batch().available(), 0);
            let response = if disconnect {
                drop(response);
                None
            } else {
                Some(response)
            };
            if !disconnect {
                state.shutdown.cancel();
            }
            state.tracker.close();
            tokio::time::timeout(std::time::Duration::from_secs(5), state.tracker.wait())
                .await
                .expect("producer and cancellation watcher stopped");
            assert_eq!(engine.pool_for_batch().available(), 1);
            if let Some(response) = response {
                let body = axum::body::to_bytes(response.into_body(), 1 << 20)
                    .await
                    .unwrap();
                let text = std::str::from_utf8(&body).unwrap();
                assert_eq!(text.matches("data: ").count(), capacity);
                if openai {
                    assert!(!text.contains("[DONE]"));
                    assert!(!text.contains("transcript.text.done"));
                }
            }
        }
    }
}
