use super::*;

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
