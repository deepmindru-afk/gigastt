//! Allocation regression for completed-job polling; no model is required.
use axum::body::Bytes;
use gigastt::server::{
    RuntimeLimits,
    http::ExportParams,
    jobs::{InMemoryJobStore, Job, JobStore},
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct CountingAllocator;
thread_local! {
    static BYTES: Cell<Option<(usize, usize)>> = const { Cell::new(None) };
}
// SAFETY: all allocation operations delegate unchanged to System; counting is thread-local.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        BYTES.with(|bytes| {
            if let Some(n) = bytes.get() {
                bytes.set(Some((n.0 + 1, n.1 + layout.size())));
            }
        });
        // SAFETY: forward the caller's valid layout to the system allocator.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the pointer and layout came from this system allocator.
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[tokio::test(flavor = "current_thread")]
async fn test_status_allocation_does_not_scale_with_completed_transcript() {
    let store = InMemoryJobStore::new(RuntimeLimits::default());
    let mut job = Job::queued(Bytes::new(), ExportParams::default());
    job.status = gigastt::server::jobs::JobStatus::Done;
    job.result = Some(gigastt::inference::TranscribeResult {
        text: "x".repeat(8 * 1024 * 1024),
        words: vec![],
        duration_s: 1.0,
        confidence: None,
    });
    let id = store.create(job).await.unwrap();
    BYTES.with(|bytes| bytes.set(Some((0, 0))));
    let status = store.status(&id).await.unwrap().unwrap();
    let allocated = BYTES.with(|bytes| bytes.replace(None).unwrap());
    assert_eq!(status.status, gigastt::server::jobs::JobStatus::Done);
    assert!(
        allocated.1 < 4096,
        "status allocated {allocated:?} (allocations, bytes)"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "synthetic concurrent polling benchmark; run explicitly with --nocapture"]
async fn benchmark_completed_status_polling() {
    use std::sync::Arc;
    let store = Arc::new(InMemoryJobStore::new(RuntimeLimits::default()));
    let mut job = Job::queued(Bytes::new(), ExportParams::default());
    job.status = gigastt::server::jobs::JobStatus::Done;
    job.result = Some(gigastt::inference::TranscribeResult {
        text: "x".repeat(8 * 1024 * 1024),
        words: vec![],
        duration_s: 1.0,
        confidence: None,
    });
    let id = store.create(job).await.unwrap();
    for legacy in [true, false] {
        let mut tasks = Vec::new();
        for _ in 0..4 {
            let store = store.clone();
            let id = id.clone();
            tasks.push(tokio::spawn(async move {
                let mut samples = Vec::with_capacity(200);
                let mut allocations = (0, 0);
                for _ in 0..200 {
                    BYTES.with(|bytes| bytes.set(Some((0, 0))));
                    let started = std::time::Instant::now();
                    if legacy {
                        let job = store.get(&id).await.unwrap().unwrap();
                        // Match the old HTTP projection, including its owned id.
                        std::hint::black_box((
                            job.id.clone(),
                            job.status,
                            job.error.clone(),
                            job.partial.as_ref().and_then(|partial| partial.get()),
                        ));
                    } else {
                        std::hint::black_box(store.status(&id).await.unwrap().unwrap());
                    }
                    let sample = started.elapsed().as_nanos();
                    let allocated = BYTES.with(|bytes| bytes.replace(None).unwrap());
                    allocations.0 += allocated.0;
                    allocations.1 += allocated.1;
                    samples.push(sample);
                    tokio::task::yield_now().await;
                }
                (samples, allocations)
            }));
        }
        let mut samples = Vec::new();
        let mut allocations = (0, 0);
        for task in tasks {
            let (batch, allocated) = task.await.unwrap();
            samples.extend(batch);
            allocations.0 += allocated.0;
            allocations.1 += allocated.1;
        }
        samples.sort_unstable();
        eprintln!(
            "legacy={legacy} allocations/poll={} bytes/poll={} p50_ns={} p99_ns={}",
            allocations.0 / samples.len(),
            allocations.1 / samples.len(),
            samples[samples.len() / 2],
            samples[samples.len() * 99 / 100]
        );
    }
}
