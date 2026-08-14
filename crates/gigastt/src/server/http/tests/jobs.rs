//! Job HTTP handlers — model-free via the mock engine + in-memory store.

use super::*;
use crate::server::http::{
    JobServerState, cancel_job, get_job, get_job_result, job_events, submit_job,
};
use crate::server::jobs::{InMemoryJobStore, Job, JobQueue, JobStatus, JobStore};
use axum::extract::Path;

fn jobs_state(engine: Arc<Engine>, limits: RuntimeLimits) -> Arc<AppState> {
    let store: Arc<dyn crate::server::jobs::JobStore> =
        Arc::new(InMemoryJobStore::new(limits.clone()));
    let shutdown = tokio_util::sync::CancellationToken::new();
    let queue = JobQueue::new(store.clone(), 1, 0, shutdown.clone());
    Arc::new(AppState {
        engine: engine_swap(engine),
        limits: Arc::new(ArcSwap::from_pointee(limits)),
        metrics_registry: None,
        engine_builder: None,
        reload_lock: Arc::new(tokio::sync::Mutex::new(())),
        shutdown,
        tracker: tokio_util::task::TaskTracker::new(),
        jobs: Some(JobServerState { store, queue }),
    })
}

#[tokio::test]
async fn test_submit_job_disabled_is_404() {
    let state = Arc::new(AppState {
        engine: engine_swap(test_engine()),
        limits: Arc::new(ArcSwap::from_pointee(RuntimeLimits::default())),
        metrics_registry: None,
        engine_builder: None,
        reload_lock: Arc::new(tokio::sync::Mutex::new(())),
        shutdown: tokio_util::sync::CancellationToken::new(),
        tracker: tokio_util::task::TaskTracker::new(),
        jobs: None,
    });
    let err = submit_job(State(state), Query(ExportParams::default()), short_wav())
        .await
        .expect_err("jobs disabled");
    assert_eq!(err.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_submit_job_empty_body_is_400() {
    let state = jobs_state(test_engine(), RuntimeLimits::default());
    let err = submit_job(State(state), Query(ExportParams::default()), Bytes::new())
        .await
        .expect_err("empty");
    assert_eq!(err.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_submit_job_payload_too_large() {
    let state = jobs_state(
        test_engine(),
        RuntimeLimits {
            body_limit_bytes: 4,
            ..RuntimeLimits::default()
        },
    );
    let err = submit_job(State(state), Query(ExportParams::default()), short_wav())
        .await
        .expect_err("too large");
    assert_eq!(err.status(), StatusCode::PAYLOAD_TOO_LARGE);
}

#[tokio::test]
async fn test_submit_job_conflict_split_and_diarization() {
    let state = jobs_state(test_engine(), RuntimeLimits::default());
    let params = ExportParams {
        channels: Some("split".into()),
        diarization: Some(true),
        ..ExportParams::default()
    };
    let err = submit_job(State(state), Query(params), short_wav())
        .await
        .expect_err("conflict");
    assert_eq!(err.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_submit_get_cancel_job_round_trip() {
    let state = jobs_state(test_engine(), RuntimeLimits::default());
    let resp = submit_job(
        State(state.clone()),
        Query(ExportParams::default()),
        short_wav(),
    )
    .await
    .expect("submit");
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
    let bytes = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let id = v["job_id"].as_str().unwrap().to_string();

    let resp = get_job(State(state.clone()), Path(id.clone()))
        .await
        .expect("get");
    assert_eq!(resp.status(), StatusCode::OK);

    let err = get_job_result(State(state.clone()), Path(id.clone()))
        .await
        .expect_err("not finished");
    assert_eq!(err.status(), StatusCode::CONFLICT);

    let resp = cancel_job(State(state.clone()), Path(id.clone()))
        .await
        .expect("cancel");
    assert_eq!(resp.status(), StatusCode::NO_CONTENT);

    let err = get_job(State(state.clone()), Path("missing".into()))
        .await
        .expect_err("missing");
    assert_eq!(err.status(), StatusCode::NOT_FOUND);

    let events = job_events(State(state), Path(id)).await.expect("events");
    drop(events);
}

#[tokio::test]
async fn test_get_job_result_returns_json_when_done() {
    let limits = RuntimeLimits::default();
    let store = Arc::new(InMemoryJobStore::new(limits.clone()));
    let mut job = Job::queued(short_wav(), ExportParams::default());
    job.status = JobStatus::Done;
    job.result = Some(sample_export_result());
    let id = store.create(job).await.expect("create");

    let shutdown = tokio_util::sync::CancellationToken::new();
    let queue = JobQueue::new(store.clone(), 1, 0, shutdown.clone());
    let state = Arc::new(AppState {
        engine: engine_swap(test_engine()),
        limits: Arc::new(ArcSwap::from_pointee(limits)),
        metrics_registry: None,
        engine_builder: None,
        reload_lock: Arc::new(tokio::sync::Mutex::new(())),
        shutdown,
        tracker: tokio_util::task::TaskTracker::new(),
        jobs: Some(JobServerState { store, queue }),
    });
    let resp = get_job_result(State(state), Path(id))
        .await
        .expect("result");
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 16)
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["text"], "привет мир");
}
