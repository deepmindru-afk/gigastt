//! Asynchronous job queue for long-file and batch transcription.
//!
//! The queue is intentionally decoupled from the HTTP surface via the
//! [`JobStore`] trait so a persistent backend (e.g. SQLite) can plug in
//! later without touching the handlers.

use axum::body::Bytes;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

use parking_lot::Mutex;

use super::config::RuntimeLimits;
use super::http::ExportParams;

/// Lifecycle status of a transcription job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    /// Waiting for a worker slot.
    Queued,
    /// Currently holding a triplet and transcribing.
    Processing,
    /// Finished successfully; result is available.
    Done,
    /// Failed after exhausting retries.
    Failed,
    /// Cancelled by the client or by shutdown.
    Cancelled,
}

impl JobStatus {
    /// Whether the job has reached a terminal state.
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            JobStatus::Done | JobStatus::Failed | JobStatus::Cancelled
        )
    }
}

/// Server-sent event emitted by `GET /v1/jobs/{id}/events`.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum JobEvent {
    /// Progress estimate while processing.
    Progress {
        /// Approximate fraction complete, 0–100.
        percent: u32,
        /// Seconds of audio considered processed so far.
        processed_seconds: f64,
    },
    /// Job completed successfully.
    Done,
    /// Job failed.
    Failed {
        /// Sanitized error message (no paths or model internals).
        error: String,
    },
    /// Job was cancelled.
    Cancelled,
}

impl JobEvent {
    /// Whether this event ends the stream.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            JobEvent::Done | JobEvent::Failed { .. } | JobEvent::Cancelled
        )
    }
}

/// Public status response for `GET /v1/jobs/{id}`.
#[derive(Debug, Clone, Serialize)]
pub struct JobStatusResponse {
    pub job_id: String,
    pub status: JobStatus,
    pub processed_seconds: f64,
    pub percent: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// A transcription job.
#[derive(Debug, Clone)]
pub struct Job {
    pub id: String,
    pub status: JobStatus,
    /// Raw uploaded audio bytes.
    pub body: Bytes,
    /// Export / post-processing parameters captured at submission.
    pub params: ExportParams,
    pub created_at: f64,
    pub updated_at: f64,
    pub processed_seconds: f64,
    /// Total audio duration in seconds, set once the body is decoded.
    pub total_seconds: f64,
    /// Number of execution attempts made so far.
    pub attempts: u32,
    /// Populated when status becomes `Done`.
    pub result: Option<gigastt_core::inference::TranscribeResult>,
    /// Populated when status becomes `Failed`.
    pub error: Option<String>,
    /// Active SSE listeners.
    pub event_channels: Vec<tokio::sync::mpsc::UnboundedSender<JobEvent>>,
}

impl Job {
    /// Create a new queued job.
    pub fn queued(body: Bytes, params: ExportParams) -> Self {
        let now = gigastt_core::inference::now_timestamp();
        Self {
            id: uuid::Uuid::now_v7().to_string(),
            status: JobStatus::Queued,
            body,
            params,
            created_at: now,
            updated_at: now,
            processed_seconds: 0.0,
            total_seconds: 0.0,
            attempts: 0,
            result: None,
            error: None,
            event_channels: Vec::new(),
        }
    }
}

/// Persistence boundary for jobs. Handlers talk to this trait; the in-memory
/// implementation is the default, but a SQLite-backed store can be dropped in.
pub trait JobStore: Send + Sync {
    /// Persist a new job and return its id.
    fn create(&self, job: Job) -> impl std::future::Future<Output = anyhow::Result<String>> + Send;
    /// Return a clone of the job, if it exists.
    fn get(
        &self,
        id: &str,
    ) -> impl std::future::Future<Output = anyhow::Result<Option<Job>>> + Send;
    /// Apply an in-place mutation.
    fn update(
        &self,
        id: &str,
        f: Box<dyn FnOnce(&mut Job) + Send>,
    ) -> impl std::future::Future<Output = anyhow::Result<()>> + Send;
    /// Pop the oldest queued job id whose status is still `Queued`.
    fn next_queued(
        &self,
    ) -> impl std::future::Future<Output = anyhow::Result<Option<String>>> + Send;
    /// Whether the store has reached its capacity limit.
    fn is_full(&self) -> impl std::future::Future<Output = bool> + Send;
}

/// In-memory FIFO job store with TTL eviction.
pub struct InMemoryJobStore {
    limits: RuntimeLimits,
    jobs: Mutex<HashMap<String, Job>>,
    queue: Mutex<VecDeque<String>>,
}

impl InMemoryJobStore {
    /// Create a new store with the given limits.
    pub fn new(limits: RuntimeLimits) -> Self {
        Self {
            limits,
            jobs: Mutex::new(HashMap::new()),
            queue: Mutex::new(VecDeque::new()),
        }
    }

    /// Evict terminal jobs whose TTL has expired. Must be called with both locks held.
    fn evict_expired_locked(&self, jobs: &mut HashMap<String, Job>, queue: &mut VecDeque<String>) {
        let ttl = self.limits.jobs_ttl_secs;
        if ttl == 0 {
            return;
        }
        let now = gigastt_core::inference::now_timestamp();
        let expired: Vec<String> = jobs
            .iter()
            .filter(|(_, j)| j.status.is_terminal() && now - j.updated_at > ttl as f64)
            .map(|(id, _)| id.clone())
            .collect();
        for id in expired {
            jobs.remove(&id);
            queue.retain(|x| x != &id);
        }
    }
}

impl JobStore for InMemoryJobStore {
    async fn create(&self, job: Job) -> anyhow::Result<String> {
        let mut jobs = self.jobs.lock();
        let mut queue = self.queue.lock();
        self.evict_expired_locked(&mut jobs, &mut queue);
        if jobs.len() >= self.limits.jobs_max {
            return Err(anyhow::anyhow!("job store is full"));
        }
        let id = job.id.clone();
        jobs.insert(id.clone(), job);
        queue.push_back(id.clone());
        Ok(id)
    }

    async fn get(&self, id: &str) -> anyhow::Result<Option<Job>> {
        let jobs = self.jobs.lock();
        Ok(jobs.get(id).cloned())
    }

    async fn update(&self, id: &str, f: Box<dyn FnOnce(&mut Job) + Send>) -> anyhow::Result<()> {
        let mut jobs = self.jobs.lock();
        let Some(job) = jobs.get_mut(id) else {
            return Err(anyhow::anyhow!("job not found"));
        };
        f(job);
        job.updated_at = gigastt_core::inference::now_timestamp();
        Ok(())
    }

    async fn next_queued(&self) -> anyhow::Result<Option<String>> {
        let mut jobs = self.jobs.lock();
        let mut queue = self.queue.lock();
        self.evict_expired_locked(&mut jobs, &mut queue);
        while let Some(id) = queue.pop_front() {
            if let Some(job) = jobs.get(&id)
                && matches!(job.status, JobStatus::Queued)
            {
                return Ok(Some(id));
            }
        }
        Ok(None)
    }

    async fn is_full(&self) -> bool {
        let jobs = self.jobs.lock();
        jobs.len() >= self.limits.jobs_max
    }
}

#[cfg(test)]
impl InMemoryJobStore {
    /// Shift a job's `updated_at` back in time for TTL eviction tests.
    pub async fn backdate(&self, id: &str, seconds: f64) {
        let mut jobs = self.jobs.lock();
        if let Some(job) = jobs.get_mut(id) {
            job.updated_at -= seconds;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
