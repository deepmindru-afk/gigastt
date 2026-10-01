//! Opt-in test-only timing of existing sidecar synchronization boundaries.
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

static ENABLED: AtomicBool = AtomicBool::new(false);
static RECORDS: parking_lot::Mutex<Vec<Observation>> = parking_lot::Mutex::new(Vec::new());

#[derive(serde::Serialize)]
struct Observation {
    stage: &'static str,
    wait_ns: Option<u128>,
    execution_ns: u128,
}

pub(crate) struct Probe {
    stage: &'static str,
    start: Option<Instant>,
    acquired: Option<Instant>,
}

impl Probe {
    pub(crate) fn new(stage: &'static str) -> Self {
        Self {
            stage,
            start: ENABLED.load(Ordering::Relaxed).then(Instant::now),
            acquired: None,
        }
    }
    pub(crate) fn acquired(&mut self) {
        if self.start.is_some() {
            self.acquired = Some(Instant::now());
        }
    }
}

impl Drop for Probe {
    fn drop(&mut self) {
        if let Some(start) = self.start {
            let end = Instant::now();
            let acquired = self.acquired.unwrap_or(start);
            RECORDS.lock().push(Observation {
                stage: self.stage,
                wait_ns: self
                    .acquired
                    .map(|time| time.duration_since(start).as_nanos()),
                execution_ns: end.duration_since(acquired).as_nanos(),
            });
        }
    }
}

#[cfg(all(feature = "file-decode", feature = "diarization", target_os = "linux"))]
mod experiment;

#[test]
fn test_probe_records_separate_wait_and_execution_intervals() {
    let start = Instant::now();
    let probe = Probe {
        stage: "synthetic_boundary",
        start: Some(start),
        acquired: Some(start + std::time::Duration::from_nanos(1)),
    };
    drop(probe);
    let mut records = RECORDS.lock();
    let position = records
        .iter()
        .position(|r| r.stage == "synthetic_boundary")
        .unwrap();
    let record = records.remove(position);
    assert_eq!(record.wait_ns, Some(1));
}
