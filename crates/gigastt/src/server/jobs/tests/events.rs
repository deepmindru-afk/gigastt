use super::*;

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
