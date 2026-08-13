//! Asynchronous job queue for long-file and batch transcription.
//!
//! The queue is intentionally decoupled from the HTTP surface via the
//! [`JobStore`] trait so a persistent backend (e.g. SQLite) can plug in
//! later without touching the handlers.

mod executor;
mod queue;
mod store;

pub use executor::RealJobExecutor;
pub use queue::{JobExecution, JobQueue};
pub use store::{
    InMemoryJobStore, Job, JobEvent, JobStatus, JobStatusResponse, JobStore, JobStoreFuture,
};

pub(crate) use queue::broadcast_event;
pub(crate) use store::job_status_response;

#[cfg(test)]
mod tests;
