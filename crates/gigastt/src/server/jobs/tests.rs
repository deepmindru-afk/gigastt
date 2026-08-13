//! Unit tests for the job queue, store, and helpers.

use super::super::config::RuntimeLimits;
use super::super::http::ExportParams;
use super::queue::{is_retryable_error, sanitize_job_error};
use super::store::MAX_JOB_EVENT_SUBSCRIBERS;
use super::*;
use axum::body::Bytes;
use parking_lot::Mutex;
use std::sync::Arc;

fn test_limits() -> RuntimeLimits {
    RuntimeLimits {
        jobs_enabled: true,
        jobs_ttl_secs: 3600,
        jobs_max: 10,
        jobs_retry: 0,
        ..RuntimeLimits::default()
    }
}

#[tokio::test]
async fn test_in_memory_store_crud() {
    let store = InMemoryJobStore::new(test_limits());
    let id = store
        .create(Job::queued(
            Bytes::from_static(b"a"),
            ExportParams::default(),
        ))
        .await
        .unwrap();
    let job = store.get(&id).await.unwrap().unwrap();
    assert!(matches!(job.status, JobStatus::Queued));
    store
        .update(&id, Box::new(|j| j.status = JobStatus::Processing))
        .await
        .unwrap();
    let job = store.get(&id).await.unwrap().unwrap();
    assert!(matches!(job.status, JobStatus::Processing));
}

#[tokio::test]
async fn test_store_fifo_order() {
    let store = InMemoryJobStore::new(test_limits());
    let id1 = store
        .create(Job::queued(
            Bytes::from_static(b"1"),
            ExportParams::default(),
        ))
        .await
        .unwrap();
    let id2 = store
        .create(Job::queued(
            Bytes::from_static(b"2"),
            ExportParams::default(),
        ))
        .await
        .unwrap();
    assert_eq!(store.next_queued().await.unwrap(), Some(id1.clone()));
    // id1 is still queued in the store; next_queued returned it but did not
    // change its status. Simulate another worker trying to pop while id1
    // is still queued: it should see id1 again because status is still Queued.
    // Mark id1 processing and then next should be id2.
    store
        .update(&id1, Box::new(|j| j.status = JobStatus::Processing))
        .await
        .unwrap();
    assert_eq!(store.next_queued().await.unwrap(), Some(id2));
}

#[tokio::test]
async fn test_store_capacity_limit() {
    let limits = RuntimeLimits {
        jobs_max: 1,
        ..test_limits()
    };
    let store = InMemoryJobStore::new(limits);
    store
        .create(Job::queued(
            Bytes::from_static(b"a"),
            ExportParams::default(),
        ))
        .await
        .unwrap();
    assert!(store.is_full().await);
    let result = store
        .create(Job::queued(
            Bytes::from_static(b"b"),
            ExportParams::default(),
        ))
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_store_byte_budget_backpressures_under_count_limit() {
    // Count limit is generous (10) but the byte budget is tiny: a store
    // holding one upload already at/over the byte budget must report full
    // and reject the next submission, even though it holds 1 << 10 jobs.
    let limits = RuntimeLimits {
        jobs_max: 10,
        jobs_max_bytes: 8,
        ..test_limits()
    };
    let store = InMemoryJobStore::new(limits);
    // First upload (11 bytes) is admitted: like the count cap admitting the
    // Nth job, the budget is checked against the bytes already resident (0).
    store
        .create(Job::queued(
            Bytes::from_static(b"audio-bytes"),
            ExportParams::default(),
        ))
        .await
        .unwrap();
    // 11 resident bytes now exceed the 8-byte budget, so the store is full
    // by bytes despite being far below jobs_max, and the next create fails.
    assert!(store.is_full().await);
    let result = store
        .create(Job::queued(
            Bytes::from_static(b"more"),
            ExportParams::default(),
        ))
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_store_byte_budget_released_when_job_terminal() {
    // A terminal job releases its body, so those bytes stop counting against
    // the budget and the queue accepts new work again.
    let limits = RuntimeLimits {
        jobs_max: 10,
        jobs_max_bytes: 8,
        ..test_limits()
    };
    let store = InMemoryJobStore::new(limits);
    let id = store
        .create(Job::queued(
            Bytes::from_static(b"audio-bytes"),
            ExportParams::default(),
        ))
        .await
        .unwrap();
    assert!(store.is_full().await);
    // Release the body the way the worker does at a terminal state.
    store
        .update(
            &id,
            Box::new(|j| {
                j.status = JobStatus::Done;
                j.body = Bytes::new();
            }),
        )
        .await
        .unwrap();
    assert!(!store.is_full().await);
}

#[tokio::test]
async fn test_store_is_full_evicts_expired() {
    let limits = RuntimeLimits {
        jobs_ttl_secs: 1,
        jobs_max: 1,
        ..test_limits()
    };
    let store = InMemoryJobStore::new(limits);
    let id = store
        .create(Job::queued(
            Bytes::from_static(b"a"),
            ExportParams::default(),
        ))
        .await
        .unwrap();
    store
        .update(&id, Box::new(|j| j.status = JobStatus::Done))
        .await
        .unwrap();
    store.backdate(&id, 2.0).await;
    // is_full should evict the expired terminal job and report capacity.
    assert!(!store.is_full().await);
}

#[tokio::test]
async fn test_store_ttl_eviction() {
    let limits = RuntimeLimits {
        jobs_ttl_secs: 1,
        jobs_max: 10,
        ..test_limits()
    };
    let store = InMemoryJobStore::new(limits);
    let id = store
        .create(Job::queued(
            Bytes::from_static(b"a"),
            ExportParams::default(),
        ))
        .await
        .unwrap();
    store
        .update(&id, Box::new(|j| j.status = JobStatus::Done))
        .await
        .unwrap();
    // Backdate the job by more than the 1-second TTL.
    store.backdate(&id, 2.0).await;
    // Creating a new job should evict the expired one.
    store
        .create(Job::queued(
            Bytes::from_static(b"b"),
            ExportParams::default(),
        ))
        .await
        .unwrap();
    assert!(store.get(&id).await.unwrap().is_none());
}

#[derive(Clone)]
struct MockExecutor {
    results: Arc<Mutex<Vec<anyhow::Result<gigastt_core::inference::TranscribeResult>>>>,
    delay_ms: u64,
}

impl JobExecution for MockExecutor {
    async fn execute(
        &self,
        _id: &str,
        _store: Arc<dyn JobStore>,
        body: Bytes,
        _params: ExportParams,
    ) -> anyhow::Result<gigastt_core::inference::TranscribeResult> {
        if self.delay_ms > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(self.delay_ms)).await;
        }
        let mut results = self.results.lock();
        let result = results.remove(0);
        // Use body length to make failures deterministic per test.
        let _ = body.len();
        result
    }
}

fn ok_result() -> anyhow::Result<gigastt_core::inference::TranscribeResult> {
    Ok(gigastt_core::inference::TranscribeResult {
        text: "ok".into(),
        words: vec![],
        duration_s: 1.0,
        confidence: None,
    })
}

#[tokio::test]
async fn test_queue_runs_jobs_in_fifo_order() {
    let limits = test_limits();
    let store: Arc<dyn JobStore> = Arc::new(InMemoryJobStore::new(limits));
    let executor = MockExecutor {
        results: Arc::new(Mutex::new(vec![ok_result(), ok_result()])),
        delay_ms: 0,
    };
    let queue = JobQueue::new(
        store.clone(),
        2,
        0,
        tokio_util::sync::CancellationToken::new(),
    );
    let tracker = tokio_util::task::TaskTracker::new();
    queue.spawn(executor, &tracker);

    let id1 = store
        .create(Job::queued(
            Bytes::from_static(b"1"),
            ExportParams::default(),
        ))
        .await
        .unwrap();
    let id2 = store
        .create(Job::queued(
            Bytes::from_static(b"2"),
            ExportParams::default(),
        ))
        .await
        .unwrap();

    // Wait for both to finish.
    for _ in 0..50 {
        let j1 = store.get(&id1).await.unwrap().unwrap();
        let j2 = store.get(&id2).await.unwrap().unwrap();
        if matches!(j1.status, JobStatus::Done) && matches!(j2.status, JobStatus::Done) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }

    assert!(matches!(
        store.get(&id1).await.unwrap().unwrap().status,
        JobStatus::Done
    ));
    assert!(matches!(
        store.get(&id2).await.unwrap().unwrap().status,
        JobStatus::Done
    ));
}

#[tokio::test]
async fn test_queue_retry_then_fail() {
    let limits = RuntimeLimits {
        jobs_retry: 2,
        ..test_limits()
    };
    let store: Arc<dyn JobStore> = Arc::new(InMemoryJobStore::new(limits));
    let executor = MockExecutor {
        // Panic is the retryable failure class (inference_timeout is not),
        // so the retry path is exercised with a panic marker.
        results: Arc::new(Mutex::new(vec![
            Err(anyhow::anyhow!("worker thread panicked")),
            Err(anyhow::anyhow!("worker thread panicked")),
            Err(anyhow::anyhow!("worker thread panicked")),
        ])),
        delay_ms: 0,
    };
    let queue = JobQueue::new(
        store.clone(),
        1,
        2,
        tokio_util::sync::CancellationToken::new(),
    );
    let tracker = tokio_util::task::TaskTracker::new();
    queue.spawn(executor, &tracker);

    let id = store
        .create(Job::queued(
            Bytes::from_static(b"x"),
            ExportParams::default(),
        ))
        .await
        .unwrap();

    for _ in 0..50 {
        let job = store.get(&id).await.unwrap().unwrap();
        if matches!(job.status, JobStatus::Failed) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }

    let job = store.get(&id).await.unwrap().unwrap();
    assert!(matches!(job.status, JobStatus::Failed));
    assert_eq!(job.attempts, 3);
}

#[tokio::test]
async fn test_queue_cancellation_discards_result() {
    let store: Arc<dyn JobStore> = Arc::new(InMemoryJobStore::new(test_limits()));
    let executor = MockExecutor {
        results: Arc::new(Mutex::new(vec![ok_result()])),
        delay_ms: 300,
    };
    let queue = JobQueue::new(
        store.clone(),
        1,
        0,
        tokio_util::sync::CancellationToken::new(),
    );
    let tracker = tokio_util::task::TaskTracker::new();
    queue.spawn(executor, &tracker);

    let id = store
        .create(Job::queued(
            Bytes::from_static(b"x"),
            ExportParams::default(),
        ))
        .await
        .unwrap();

    // Wait until processing, then cancel while the executor is still running.
    for _ in 0..50 {
        let job = store.get(&id).await.unwrap().unwrap();
        if matches!(job.status, JobStatus::Processing) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    store
        .update(&id, Box::new(|j| j.status = JobStatus::Cancelled))
        .await
        .unwrap();

    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    let job = store.get(&id).await.unwrap().unwrap();
    assert!(matches!(job.status, JobStatus::Cancelled));
    assert!(job.result.is_none());
}

#[tokio::test]
async fn test_queue_cancel_all_queued_on_shutdown() {
    let store: Arc<dyn JobStore> = Arc::new(InMemoryJobStore::new(test_limits()));
    let token = tokio_util::sync::CancellationToken::new();
    let queue = JobQueue::new(store.clone(), 1, 0, token.clone());
    // Don't spawn workers so jobs stay queued.

    let id = store
        .create(Job::queued(
            Bytes::from_static(b"x"),
            ExportParams::default(),
        ))
        .await
        .unwrap();

    token.cancel();
    queue.cancel_all_queued().await;

    let job = store.get(&id).await.unwrap().unwrap();
    assert!(matches!(job.status, JobStatus::Cancelled));
}

#[test]
fn test_sanitize_job_error_maps_known_errors() {
    assert_eq!(
        sanitize_job_error(&anyhow::anyhow!("inference_timeout")),
        "Inference timed out."
    );
    assert_eq!(
        sanitize_job_error(&anyhow::anyhow!("Invalid audio: unsupported format")),
        "Failed to decode audio file. Check format."
    );
    assert_eq!(
        sanitize_job_error(&anyhow::anyhow!("some internal onnx path /foo/bar")),
        "Transcription failed."
    );
    // A typed AudioTooLong (arrives via `anyhow::Error::from`) surfaces the
    // observed/limit seconds so a client can tell "too long" from "corrupt".
    let too_long: anyhow::Error = gigastt_core::error::GigasttError::AudioTooLong {
        observed_secs: 4000.0,
        limit_secs: 1800.0,
    }
    .into();
    assert_eq!(
        sanitize_job_error(&too_long),
        "Audio too long: 4000s exceeds the maximum of 1800s."
    );
}

#[test]
fn test_is_retryable_error_recognizes_transient_failures() {
    // A panic is transient — a fresh worker may succeed. An inference_timeout
    // is deterministic, so it is NOT retryable. A decode error never is.
    assert!(is_retryable_error(&anyhow::anyhow!(
        "worker thread panicked"
    )));
    assert!(!is_retryable_error(&anyhow::anyhow!("inference_timeout")));
    assert!(!is_retryable_error(&anyhow::anyhow!("Invalid audio")));
}

#[tokio::test]
async fn test_store_get_missing_returns_none() {
    let store = InMemoryJobStore::new(test_limits());
    assert!(store.get("no-such-id").await.unwrap().is_none());
}

#[tokio::test]
async fn test_store_update_missing_returns_error() {
    let store = InMemoryJobStore::new(test_limits());
    assert!(
        store
            .update("no-such-id", Box::new(|j| j.status = JobStatus::Done))
            .await
            .is_err()
    );
}

#[tokio::test]
async fn test_broadcast_event_prunes_dead_channels() {
    let store: Arc<dyn JobStore> = Arc::new(InMemoryJobStore::new(test_limits()));
    let id = store
        .create(Job::queued(
            Bytes::from_static(b"x"),
            ExportParams::default(),
        ))
        .await
        .unwrap();

    // Add a live channel and a channel that will be dropped before broadcast.
    let (live_tx, mut live_rx) = tokio::sync::mpsc::unbounded_channel::<JobEvent>();
    {
        let (dead_tx, _dead_rx) = tokio::sync::mpsc::unbounded_channel::<JobEvent>();
        store
            .update(
                &id,
                Box::new(move |j| {
                    j.event_channels.push(live_tx);
                    j.event_channels.push(dead_tx);
                }),
            )
            .await
            .unwrap();
    }

    broadcast_event(
        &*store,
        &id,
        JobEvent::Progress {
            percent: 50,
            processed_seconds: 1.0,
        },
    )
    .await;

    let job = store.get(&id).await.unwrap().unwrap();
    assert_eq!(job.event_channels.len(), 1);
    assert!(matches!(live_rx.try_recv(), Ok(JobEvent::Progress { .. })));
}

#[tokio::test]
async fn test_queue_concurrency_clamped_to_one() {
    let store: Arc<dyn JobStore> = Arc::new(InMemoryJobStore::new(test_limits()));
    let executor = MockExecutor {
        results: Arc::new(Mutex::new(vec![ok_result(), ok_result()])),
        delay_ms: 0,
    };
    // Pass 0; the queue must still spawn at least one worker.
    let queue = JobQueue::new(
        store.clone(),
        0,
        0,
        tokio_util::sync::CancellationToken::new(),
    );
    let tracker = tokio_util::task::TaskTracker::new();
    queue.spawn(executor, &tracker);

    let id = store
        .create(Job::queued(
            Bytes::from_static(b"x"),
            ExportParams::default(),
        ))
        .await
        .unwrap();

    for _ in 0..50 {
        let job = store.get(&id).await.unwrap().unwrap();
        if matches!(job.status, JobStatus::Done) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }

    assert!(matches!(
        store.get(&id).await.unwrap().unwrap().status,
        JobStatus::Done
    ));
}

#[tokio::test]
async fn test_queue_non_retryable_error_fails_immediately() {
    let store: Arc<dyn JobStore> = Arc::new(InMemoryJobStore::new(test_limits()));
    let executor = MockExecutor {
        results: Arc::new(Mutex::new(vec![Err(anyhow::anyhow!(
            "Invalid audio: bad header"
        ))])),
        delay_ms: 0,
    };
    let queue = JobQueue::new(
        store.clone(),
        1,
        3,
        tokio_util::sync::CancellationToken::new(),
    );
    let tracker = tokio_util::task::TaskTracker::new();
    queue.spawn(executor, &tracker);

    let id = store
        .create(Job::queued(
            Bytes::from_static(b"x"),
            ExportParams::default(),
        ))
        .await
        .unwrap();

    for _ in 0..50 {
        let job = store.get(&id).await.unwrap().unwrap();
        if matches!(job.status, JobStatus::Failed) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }

    let job = store.get(&id).await.unwrap().unwrap();
    assert!(matches!(job.status, JobStatus::Failed));
    assert_eq!(job.attempts, 1);
}

#[tokio::test]
async fn test_queue_timeout_fails_once() {
    // A deterministic inference_timeout must fail on its first and only
    // attempt — even with retry budget to spare — so a too-slow file can't
    // burn (max_retries + 1) × the timeout while the queue waits.
    let store: Arc<dyn JobStore> = Arc::new(InMemoryJobStore::new(test_limits()));
    let executor = MockExecutor {
        // Second result is queued but must never be consumed: no retry.
        results: Arc::new(Mutex::new(vec![
            Err(anyhow::anyhow!("inference_timeout")),
            ok_result(),
        ])),
        delay_ms: 0,
    };
    // Generous retry budget (3) — the point is that timeout ignores it.
    let queue = JobQueue::new(
        store.clone(),
        1,
        3,
        tokio_util::sync::CancellationToken::new(),
    );
    let tracker = tokio_util::task::TaskTracker::new();
    queue.spawn(executor, &tracker);

    let id = store
        .create(Job::queued(
            Bytes::from_static(b"x"),
            ExportParams::default(),
        ))
        .await
        .unwrap();

    for _ in 0..50 {
        let job = store.get(&id).await.unwrap().unwrap();
        if matches!(job.status, JobStatus::Failed) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }

    let job = store.get(&id).await.unwrap().unwrap();
    assert!(matches!(job.status, JobStatus::Failed));
    assert_eq!(
        job.attempts, 1,
        "inference_timeout must fail once, not retry"
    );
}

#[tokio::test]
async fn test_queue_retry_boundary_exhausted_after_max_retries() {
    // max_retries=1 means one retry is allowed: attempts 1 (retry), 2 (fail).
    let limits = RuntimeLimits {
        jobs_retry: 1,
        ..test_limits()
    };
    let store: Arc<dyn JobStore> = Arc::new(InMemoryJobStore::new(limits));
    let executor = MockExecutor {
        // Retry path is exercised with a panic marker (the retryable class);
        // inference_timeout is deterministic and fails once.
        results: Arc::new(Mutex::new(vec![
            Err(anyhow::anyhow!("worker thread panicked")),
            Err(anyhow::anyhow!("worker thread panicked")),
        ])),
        delay_ms: 0,
    };
    let queue = JobQueue::new(
        store.clone(),
        1,
        1,
        tokio_util::sync::CancellationToken::new(),
    );
    let tracker = tokio_util::task::TaskTracker::new();
    queue.spawn(executor, &tracker);

    let id = store
        .create(Job::queued(
            Bytes::from_static(b"x"),
            ExportParams::default(),
        ))
        .await
        .unwrap();

    for _ in 0..50 {
        let job = store.get(&id).await.unwrap().unwrap();
        if matches!(job.status, JobStatus::Failed) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }

    let job = store.get(&id).await.unwrap().unwrap();
    assert!(matches!(job.status, JobStatus::Failed));
    assert_eq!(job.attempts, 2);
}

#[test]
fn test_job_status_response_percent() {
    let mut job = Job::queued(Bytes::new(), ExportParams::default());
    job.id = "test".into();
    job.total_seconds = 10.0;
    job.processed_seconds = 3.5;
    job.status = JobStatus::Processing;
    let resp = job_status_response(&job);
    assert_eq!(resp.percent, 35);
    assert_eq!(resp.processed_seconds, 3.5);
}

#[tokio::test]
async fn test_queue_done_job_releases_body() {
    let store: Arc<dyn JobStore> = Arc::new(InMemoryJobStore::new(test_limits()));
    let executor = MockExecutor {
        results: Arc::new(Mutex::new(vec![ok_result()])),
        delay_ms: 0,
    };
    let queue = JobQueue::new(
        store.clone(),
        1,
        0,
        tokio_util::sync::CancellationToken::new(),
    );
    let tracker = tokio_util::task::TaskTracker::new();
    queue.spawn(executor, &tracker);

    let id = store
        .create(Job::queued(
            Bytes::from_static(b"audio-bytes"),
            ExportParams::default(),
        ))
        .await
        .unwrap();

    for _ in 0..50 {
        let job = store.get(&id).await.unwrap().unwrap();
        if matches!(job.status, JobStatus::Done) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }

    let job = store.get(&id).await.unwrap().unwrap();
    assert!(matches!(job.status, JobStatus::Done));
    assert!(job.body.is_empty());
}

/// Records the body length seen by each attempt so tests can verify the
/// upload survives retries and is only released at a terminal state.
#[derive(Clone)]
struct BodyLenRecorder {
    lens: Arc<Mutex<Vec<usize>>>,
}

impl JobExecution for BodyLenRecorder {
    async fn execute(
        &self,
        _id: &str,
        _store: Arc<dyn JobStore>,
        body: Bytes,
        _params: ExportParams,
    ) -> anyhow::Result<gigastt_core::inference::TranscribeResult> {
        self.lens.lock().push(body.len());
        // Retryable failure class, so the body must survive to the next
        // attempt and only be released at the terminal state.
        Err(anyhow::anyhow!("worker thread panicked"))
    }
}

#[tokio::test]
async fn test_queue_failed_job_releases_body_after_retries() {
    // max_retries=1 → two attempts, then terminal Failed.
    let limits = RuntimeLimits {
        jobs_retry: 1,
        ..test_limits()
    };
    let store: Arc<dyn JobStore> = Arc::new(InMemoryJobStore::new(limits));
    let lens = Arc::new(Mutex::new(Vec::new()));
    let executor = BodyLenRecorder { lens: lens.clone() };
    let queue = JobQueue::new(
        store.clone(),
        1,
        1,
        tokio_util::sync::CancellationToken::new(),
    );
    let tracker = tokio_util::task::TaskTracker::new();
    queue.spawn(executor, &tracker);

    let id = store
        .create(Job::queued(
            Bytes::from_static(b"audio-bytes"),
            ExportParams::default(),
        ))
        .await
        .unwrap();

    for _ in 0..50 {
        let job = store.get(&id).await.unwrap().unwrap();
        if matches!(job.status, JobStatus::Failed) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }

    let job = store.get(&id).await.unwrap().unwrap();
    assert!(matches!(job.status, JobStatus::Failed));
    // Both attempts received the full upload; the body is released only
    // once the job reaches a terminal state.
    assert_eq!(*lens.lock(), vec![11, 11]);
    assert!(job.body.is_empty());
}

#[test]
fn test_subscribe_prunes_closed_channels() {
    let mut job = Job::queued(Bytes::new(), ExportParams::default());
    let (dead_tx, dead_rx) = tokio::sync::mpsc::unbounded_channel::<JobEvent>();
    job.event_channels.push(dead_tx);
    drop(dead_rx);

    let (live_tx, _live_rx) = tokio::sync::mpsc::unbounded_channel::<JobEvent>();
    job.subscribe(live_tx);

    assert_eq!(job.event_channels.len(), 1);
}

#[test]
fn test_subscribe_evicts_oldest_at_cap() {
    let mut job = Job::queued(Bytes::new(), ExportParams::default());
    let mut rxs = Vec::new();
    for _ in 0..MAX_JOB_EVENT_SUBSCRIBERS {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<JobEvent>();
        job.subscribe(tx);
        rxs.push(rx);
    }
    assert_eq!(job.event_channels.len(), MAX_JOB_EVENT_SUBSCRIBERS);

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<JobEvent>();
    job.subscribe(tx);
    rxs.push(rx);

    // The list stays capped; the oldest listener was evicted, so its
    // stream is disconnected.
    assert_eq!(job.event_channels.len(), MAX_JOB_EVENT_SUBSCRIBERS);
    assert!(matches!(
        rxs[0].try_recv(),
        Err(tokio::sync::mpsc::error::TryRecvError::Disconnected)
    ));
}
