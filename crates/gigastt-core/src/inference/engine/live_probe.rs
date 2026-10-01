//! Opt-in test telemetry. No fields, overrides or timers enter production builds.
use super::StreamingState;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::time::Duration;

#[derive(Default, serde::Serialize)]
struct Probe {
    stride_samples: usize,
    stages_ns: BTreeMap<&'static str, u64>,
    windows: Vec<Window>,
    process_thread_cpu_ns: BTreeMap<&'static str, u64>,
    #[cfg(target_os = "linux")]
    #[serde(skip)]
    cpu_enabled: bool,
    #[cfg(target_os = "linux")]
    #[serde(skip)]
    cpu_start: Option<u64>,
}
#[derive(serde::Serialize)]
struct Window {
    start: usize,
    samples: usize,
    pending: usize,
    context: usize,
}
thread_local! { static PROBE: RefCell<Option<Probe>> = const { RefCell::new(None) }; }

pub(super) fn stride(default: usize) -> usize {
    PROBE.with(|probe| {
        probe
            .borrow()
            .as_ref()
            .map_or(default, |p| p.stride_samples)
    })
}
// Sum of process thread runtimes, including background workers during the stage.
#[cfg(target_os = "linux")]
fn process_thread_cpu() -> u64 {
    std::fs::read_dir("/proc/self/task")
        .unwrap()
        .map(
            |entry| match std::fs::read_to_string(entry.unwrap().path().join("schedstat")) {
                Ok(stat) => stat
                    .split_whitespace()
                    .next()
                    .unwrap()
                    .parse::<u64>()
                    .unwrap(),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
                Err(error) => panic!("cannot sample thread CPU: {error}"),
            },
        )
        .sum()
}
pub(super) fn begin_stage() {
    #[cfg(target_os = "linux")]
    PROBE.with(|probe| {
        if let Some(probe) = probe.borrow_mut().as_mut().filter(|p| p.cpu_enabled) {
            probe.cpu_start = Some(process_thread_cpu());
        }
    });
}
pub(super) fn stage(name: &'static str, duration: Duration) {
    PROBE.with(|probe| {
        if let Some(probe) = probe.borrow_mut().as_mut() {
            *probe.stages_ns.entry(name).or_default() += duration.as_nanos() as u64;
            #[cfg(target_os = "linux")]
            if let Some(start) = probe.cpu_start.take() {
                *probe.process_thread_cpu_ns.entry(name).or_default() +=
                    process_thread_cpu().saturating_sub(start);
            }
        }
    });
}
pub(super) fn window(state: &StreamingState) {
    PROBE.with(|probe| {
        if let Some(probe) = probe.borrow_mut().as_mut() {
            probe.windows.push(Window {
                start: state.window_start_samples,
                samples: state.audio_buffer.len(),
                pending: state.pending_samples,
                context: state.context_samples,
            });
        }
    });
}

#[cfg(all(feature = "file-decode", target_os = "linux"))]
mod experiment;

#[test]
fn test_live_probe_stride_override_is_thread_local_and_inactive_by_default() {
    assert_eq!(stride(12800), 12800);
    std::thread::spawn(|| {
        PROBE.with(|probe| {
            *probe.borrow_mut() = Some(Probe {
                stride_samples: 6400,
                ..Probe::default()
            })
        });
        assert_eq!(stride(12800), 6400);
    })
    .join()
    .unwrap();
    assert_eq!(stride(12800), 12800);
}
