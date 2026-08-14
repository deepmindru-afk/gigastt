use super::*;

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
