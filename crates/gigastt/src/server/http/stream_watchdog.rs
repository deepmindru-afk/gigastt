//! File-stream liveness independent of the blocking producer and response reader.
use parking_lot::Mutex;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

#[derive(Clone, Copy, PartialEq)]
enum Phase {
    Processing,
    Sending,
    Finishing,
    Finished,
    TimedOut,
}

pub(super) struct StreamWatchdog {
    phase: Mutex<(Phase, Instant)>,
    timed_out: CancellationToken,
    finished: CancellationToken,
}

impl StreamWatchdog {
    pub(super) fn start(
        state: &super::state::AppState,
        seconds: u64,
        abort: Arc<AtomicBool>,
        cancel: CancellationToken,
    ) -> Arc<Self> {
        let watchdog = Arc::new(Self {
            phase: Mutex::new((Phase::Processing, Instant::now())),
            timed_out: CancellationToken::new(),
            finished: CancellationToken::new(),
        });
        if seconds != 0 {
            let worker = watchdog.clone();
            let metrics = state.metrics_registry.clone();
            state.tracker.spawn(async move {
                let mut interval = tokio::time::interval(std::time::Duration::from_millis(100));
                loop {
                    tokio::select! {
                        biased;
                        _ = worker.finished.cancelled() => return,
                        _ = cancel.cancelled() => return,
                        _ = interval.tick() => {
                            let expired = {
                                let mut phase = worker.phase.lock();
                                if phase.0 == Phase::Processing && phase.1.elapsed().as_secs() >= seconds {
                                    phase.0 = Phase::TimedOut;
                                    abort.store(true, Ordering::Relaxed);
                                    // Publish under the arbitration lock so producer completion
                                    // cannot close the queue before the timeout is observable.
                                    worker.timed_out.cancel();
                                    true
                                } else { false }
                            };
                            if expired {
                                cancel.cancel();
                                if let Some(metrics) = &metrics {
                                    metrics.counter_inc("gigastt_inference_timeouts_total", &[], 1);
                                }
                                return;
                            }
                        }
                    }
                }
            });
        }
        watchdog
    }

    pub(super) fn progress(&self) {
        let mut phase = self.phase.lock();
        if phase.0 == Phase::Processing {
            phase.1 = Instant::now();
        }
    }

    /// Arbitrate terminal delivery against timeout before enqueueing a final.
    pub(super) fn claim_finish(&self) -> bool {
        let mut phase = self.phase.lock();
        match phase.0 {
            Phase::Processing | Phase::Finishing => {
                phase.0 = Phase::Finishing;
                true
            }
            _ => false,
        }
    }

    pub(super) fn completion(self: &Arc<Self>) -> Completion {
        Completion(self.clone())
    }

    pub(super) async fn send<T>(
        &self,
        tx: &tokio::sync::mpsc::Sender<T>,
        item: T,
        cancel: &CancellationToken,
        abort: &AtomicBool,
    ) -> bool {
        let previous = {
            let mut phase = self.phase.lock();
            match phase.0 {
                Phase::Processing | Phase::Finishing => {
                    let previous = phase.0;
                    phase.0 = Phase::Sending;
                    previous
                }
                _ => return false,
            }
        };
        let _sending = Sending {
            watchdog: self,
            previous,
        };
        super::stream::send_stream_item(tx, item, cancel, abort).await
    }

    /// No forwarding queue: preserve the endpoint's existing backpressure bound.
    /// Timeout wins over buffered output; late native results cannot become finals.
    pub(super) fn response<T: Send + 'static>(
        self: Arc<Self>,
        rx: tokio::sync::mpsc::Receiver<T>,
        timeout_items: impl FnOnce() -> Vec<T> + Send + 'static,
    ) -> impl futures_util::Stream<Item = T> + Send {
        let watchdog = self.clone();
        futures_util::stream::unfold(
            (rx, watchdog, Some(timeout_items), Vec::new().into_iter()),
            |(mut rx, watchdog, mut timeout_items, mut pending)| async move {
                if timeout_items.is_none() {
                    return pending
                        .next()
                        .map(|item| (item, (rx, watchdog, timeout_items, pending)));
                }
                tokio::select! {
                    biased;
                    _ = watchdog.timed_out.cancelled() => {
                        rx.close();
                        while rx.try_recv().is_ok() {}
                        pending = (timeout_items.take()?)().into_iter();
                        pending.next().map(|item| (item, (rx, watchdog, timeout_items, pending)))
                    }
                    item = rx.recv() => item.map(|item| (item, (rx, watchdog, timeout_items, pending))),
                }
            },
        )
    }
}

struct Sending<'a> {
    watchdog: &'a StreamWatchdog,
    previous: Phase,
}
impl Drop for Sending<'_> {
    fn drop(&mut self) {
        *self.watchdog.phase.lock() = (self.previous, Instant::now());
    }
}
pub(super) struct Completion(Arc<StreamWatchdog>);
impl Drop for Completion {
    fn drop(&mut self) {
        let mut phase = self.0.phase.lock();
        if phase.0 != Phase::TimedOut {
            phase.0 = Phase::Finished;
        }
        self.0.finished.cancel();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;
    use std::time::Duration;

    fn state() -> super::super::state::AppState {
        super::super::state::AppState {
            engine: super::super::tests::engine_swap(super::super::tests::test_engine()),
            limits: Arc::new(arc_swap::ArcSwap::from_pointee(
                crate::server::config::RuntimeLimits::default(),
            )),
            metrics_registry: None,
            engine_builder: None,
            reload_lock: Arc::new(tokio::sync::Mutex::new(())),
            shutdown: CancellationToken::new(),
            tracker: tokio_util::task::TaskTracker::new(),
            jobs: None,
        }
    }

    async fn advance(milliseconds: u64) {
        tokio::time::advance(Duration::from_millis(milliseconds)).await;
        tokio::task::yield_now().await;
    }

    #[tokio::test(start_paused = true)]
    async fn test_terminal_success_prevents_timeout_after_final_is_queued() {
        let state = state();
        let abort = Arc::new(AtomicBool::new(false));
        let cancel = CancellationToken::new();
        let watchdog = StreamWatchdog::start(&state, 1, abort.clone(), cancel.clone());
        let complete = watchdog.completion();
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        tokio::task::yield_now().await;
        assert!(watchdog.claim_finish());
        assert!(watchdog.send(&tx, "final", &cancel, &abort).await);
        advance(2000).await;
        assert!(!watchdog.timed_out.is_cancelled());
        assert!(!abort.load(Ordering::Relaxed));
        drop(complete);
        drop(tx);
        let response = watchdog.response(rx, || vec!["timeout"]);
        futures_util::pin_mut!(response);
        assert_eq!(response.next().await, Some("final"));
        assert_eq!(response.next().await, None);
    }

    #[tokio::test(start_paused = true)]
    async fn test_completed_progress_resets_deadline_and_timeout_rejects_late_success() {
        let state = state();
        let abort = Arc::new(AtomicBool::new(false));
        let cancel = CancellationToken::new();
        let watchdog = StreamWatchdog::start(&state, 1, abort.clone(), cancel.clone());
        let _complete = watchdog.completion();
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        tx.try_send("buffered partial").unwrap();
        tokio::task::yield_now().await;
        advance(900).await;
        watchdog.progress();
        advance(900).await;
        assert!(!watchdog.timed_out.is_cancelled());
        advance(200).await;
        assert!(watchdog.timed_out.is_cancelled());
        assert!(abort.load(Ordering::Relaxed));
        watchdog.progress();
        assert!(!watchdog.claim_finish());
        assert!(!watchdog.send(&tx, "late final", &cancel, &abort).await);
        let response = watchdog.response(rx, || vec!["timeout"]);
        futures_util::pin_mut!(response);
        assert_eq!(response.next().await, Some("timeout"));
        // The sender is still alive: response closure cannot depend on worker exit.
        assert_eq!(response.next().await, None);
        assert!(tx.is_closed());
    }

    #[tokio::test(start_paused = true)]
    async fn test_output_backpressure_suspends_inference_deadline_then_resumes() {
        let state = state();
        let abort = Arc::new(AtomicBool::new(false));
        let cancel = CancellationToken::new();
        let watchdog = StreamWatchdog::start(&state, 1, abort.clone(), cancel.clone());
        let _complete = watchdog.completion();
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        tx.try_send("first").unwrap();
        let sender = {
            let watchdog = watchdog.clone();
            let abort = abort.clone();
            let cancel = cancel.clone();
            tokio::spawn(async move { watchdog.send(&tx, "second", &cancel, &abort).await })
        };
        tokio::task::yield_now().await;
        assert!(watchdog.phase.lock().0 == Phase::Sending);
        advance(2000).await;
        assert!(!watchdog.timed_out.is_cancelled());
        assert!(!abort.load(Ordering::Relaxed));
        assert!(!sender.is_finished());
        assert_eq!(rx.recv().await, Some("first"));
        assert!(sender.await.unwrap());
        assert_eq!(rx.recv().await, Some("second"));
        advance(900).await;
        assert!(!watchdog.timed_out.is_cancelled());
        advance(200).await;
        assert!(watchdog.timed_out.is_cancelled());
    }

    #[tokio::test(start_paused = true)]
    async fn test_disabled_watchdog_does_not_expire_processing() {
        let state = state();
        let abort = Arc::new(AtomicBool::new(false));
        let cancel = CancellationToken::new();
        let watchdog = StreamWatchdog::start(&state, 0, abort.clone(), cancel.clone());
        let _complete = watchdog.completion();
        advance(3_600_000).await;
        assert!(!watchdog.timed_out.is_cancelled());
        assert!(!abort.load(Ordering::Relaxed));
        assert!(!cancel.is_cancelled());
        assert!(watchdog.claim_finish());
    }
}
