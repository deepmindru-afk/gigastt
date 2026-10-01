//! Model-free thread launch and actual engine/pool fairness measurements.
use anyhow::Context;
use gigastt_core::inference::{FeatureExtractor, TranscribeRequest, TranscribeSource, audio};
use gigastt_core::runtime_api::{
    Runtime, RuntimeError, RuntimeFactory, RuntimeSession, Shape, Tensor, TensorData, cpu_factory,
};
use gigastt_core::test_support::{load_scheduling_engine, rnnt_factory, write_rnnt_layout};
use serde_json::json;
use std::path::Path;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed},
    mpsc,
};
use std::time::{Duration, Instant};

fn launch_profile() {
    for slots in [1, 2, 4] {
        for waves in [1, 4, 16, 64] {
            let mut times = Vec::new();
            for _ in 0..21 {
                let start = Instant::now();
                for _ in 0..waves {
                    if slots == 1 {
                        std::hint::black_box(0);
                        continue;
                    }
                    std::thread::scope(|scope| {
                        let mut handles = Vec::with_capacity(slots);
                        for slot in 0..slots {
                            handles.push(scope.spawn(move || std::hint::black_box(slot)));
                        }
                        // Production joins in input order, including the primary worker.
                        let mut outputs = Vec::with_capacity(slots);
                        for handle in handles {
                            outputs.push(handle.join().expect("no-op worker panicked"));
                        }
                        std::hint::black_box(outputs);
                    });
                }
                times.push(start.elapsed().as_secs_f64() * 1000.0);
            }
            println!(
                "{}",
                json!({"kind":"launch","slots":slots,"waves":waves,"milliseconds":times,
                "cpu_budget":std::thread::available_parallelism().map_or(0,usize::from)})
            );
        }
    }
}

struct Control {
    armed: AtomicBool,
    first_batch: AtomicBool,
    release: AtomicBool,
    started: mpsc::Sender<()>,
    batch_runs: AtomicUsize,
    batch_completed: AtomicUsize,
    live_runs: AtomicUsize,
}
struct DelayFactory(Arc<Control>);
struct DelayRuntime {
    inner: Box<dyn Runtime>,
    control: Arc<Control>,
}
struct DelaySession {
    inner: Box<dyn RuntimeSession>,
    control: Arc<Control>,
}
impl RuntimeFactory for DelayFactory {
    fn create(&self, threads: usize) -> Result<Box<dyn Runtime>, RuntimeError> {
        Ok(Box::new(DelayRuntime {
            inner: rnnt_factory().create(threads)?,
            control: self.0.clone(),
        }))
    }
    fn cpu_fallback(&self) -> Box<dyn RuntimeFactory> {
        Box::new(Self(self.0.clone()))
    }
    fn verify_on_disk_checksums(&self) -> bool {
        false
    }
}
impl Runtime for DelayRuntime {
    fn load_session(
        &self,
        path: &Path,
        encoder: bool,
    ) -> Result<Box<dyn RuntimeSession>, RuntimeError> {
        let inner = self.inner.load_session(path, encoder)?;
        if encoder {
            Ok(Box::new(DelaySession {
                inner,
                control: self.control.clone(),
            }))
        } else {
            Ok(inner)
        }
    }
}
impl RuntimeSession for DelaySession {
    fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, RuntimeError> {
        let batch = self.control.armed.load(Relaxed) && inputs[0].shape().dims()[2] > 1000;
        if self.control.armed.load(Relaxed) {
            if batch {
                self.control.batch_runs.fetch_add(1, Relaxed);
                if !self.control.first_batch.swap(true, Relaxed) {
                    let _ = self.control.started.send(());
                    let deadline = Instant::now() + Duration::from_secs(10);
                    while !self.control.release.load(Relaxed) && Instant::now() < deadline {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                }
            } else {
                self.control.live_runs.fetch_add(1, Relaxed);
            }
            std::thread::sleep(Duration::from_millis(if batch { 20 } else { 2 }));
        }
        let result = self.inner.run(inputs);
        if batch {
            self.control.batch_completed.fetch_add(1, Relaxed);
        }
        result
    }
}

fn mixed_profile() -> anyhow::Result<()> {
    let layout = tempfile::tempdir()?;
    write_rnnt_layout(layout.path())?;
    for (slots, batch_slots, cap) in [(2, 0, 1), (2, 0, 2), (4, 0, 2), (4, 0, 4), (4, 3, 3)] {
        for windows in [2, 8, 16] {
            for repeat in 0..5 {
                let (tx, rx) = mpsc::channel();
                let control = Arc::new(Control {
                    armed: AtomicBool::new(false),
                    first_batch: AtomicBool::new(false),
                    release: AtomicBool::new(false),
                    started: tx,
                    batch_runs: AtomicUsize::new(0),
                    batch_completed: AtomicUsize::new(0),
                    live_runs: AtomicUsize::new(0),
                });
                let engine = load_scheduling_engine(
                    layout.path(),
                    slots,
                    batch_slots,
                    Box::new(DelayFactory(control.clone())),
                )?
                .with_file_window_concurrency(cap);
                let samples = vec![0.01; (24 + (windows - 1) * 22) * 16000];
                let batch_done = AtomicBool::new(false);
                control.armed.store(true, Relaxed);
                let (batch_ms, live) = std::thread::scope(|scope| -> anyhow::Result<_> {
                    let batch = scope.spawn(|| -> anyhow::Result<f64> {
                        let mut slot = engine.pool_for_batch().checkout_blocking()?;
                        let start = Instant::now();
                        engine.transcribe_request(
                            TranscribeRequest::new(TranscribeSource::Samples(&samples)),
                            &mut slot,
                        )?;
                        batch_done.store(true, Relaxed);
                        Ok(start.elapsed().as_secs_f64() * 1000.0)
                    });
                    rx.recv_timeout(Duration::from_secs(10))
                        .context("batch did not enter encoder")?;
                    let mut live = Vec::new();
                    for _ in 0..8 {
                        live.push(scope.spawn(|| -> anyhow::Result<_> {
                            let start = Instant::now();
                            let mut slot = engine.pool.checkout_blocking()?;
                            let checkout_ms = start.elapsed().as_secs_f64() * 1000.0;
                            let acquired_before_batch_end = !batch_done.load(Relaxed);
                            let acquired_before_last_window_completed =
                                control.batch_completed.load(Relaxed) < windows;
                            let mut state = engine.create_state(false);
                            engine.process_chunk(&vec![0.01; 40000], &mut state, &mut slot)?;
                            Ok(json!({
                                "checkout_ms": checkout_ms,
                                "completion_ms": start.elapsed().as_secs_f64() * 1000.0,
                                "acquired_before_batch_end": acquired_before_batch_end,
                                "acquired_before_last_window_completed": acquired_before_last_window_completed,
                            }))
                        }));
                    }
                    // Wait for every live caller to either acquire or queue, then release
                    // the first encoder. Do not include arbitrary startup races as fairness.
                    let deadline = Instant::now() + Duration::from_secs(2);
                    while engine.pool.waiters() + control.live_runs.load(Relaxed) < 8
                        && Instant::now() < deadline
                    {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    let coordinated = engine.pool.waiters() + control.live_runs.load(Relaxed) >= 8;
                    // Release before reporting setup failure so scoped workers can finish.
                    control.release.store(true, Relaxed);
                    anyhow::ensure!(coordinated, "live arrival coordination timed out");
                    let batch_ms = batch
                        .join()
                        .map_err(|_| anyhow::anyhow!("batch worker panicked"))??;
                    let live = live
                        .into_iter()
                        .map(|h| {
                            h.join()
                                .map_err(|_| anyhow::anyhow!("live worker panicked"))?
                        })
                        .collect::<anyhow::Result<Vec<_>>>()?;
                    Ok((batch_ms, live))
                })?;
                anyhow::ensure!(
                    control.batch_runs.load(Relaxed) == windows,
                    "unexpected window count"
                );
                anyhow::ensure!(
                    control.live_runs.load(Relaxed) == 8,
                    "live requests did not each decode once"
                );
                anyhow::ensure!(
                    engine.pool.available()
                        + engine.batch_pool.as_ref().map_or(0, |p| p.available())
                        == slots,
                    "slot leak"
                );
                println!(
                    "{}",
                    json!({"kind":"mixed","slots":slots,"batch_slots":batch_slots,"cap":cap,
                    "windows":windows,"repeat":repeat,"batch_ms":batch_ms,"live":live,
                    "batch_service_ms":20,"live_service_ms":2,
                    "cpu_budget":std::thread::available_parallelism().map_or(0,usize::from)})
                );
            }
        }
    }
    Ok(())
}

fn native_profile() -> anyhow::Result<()> {
    let models = std::env::var_os("GIGASTT_SCHEDULING_MODEL_DIR")
        .map(std::path::PathBuf::from)
        .context("set GIGASTT_SCHEDULING_MODEL_DIR to local ml_ctc model directory")?;
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../gigastt/tests/fixtures/golos_00.wav");
    let pcm = audio::decode_audio_file(fixture.to_str().context("fixture path not UTF-8")?)?;
    let samples: Vec<_> = pcm.iter().copied().cycle().take(24 * 16000).collect();
    let start = Instant::now();
    let (features, frames) = FeatureExtractor::new().compute(&samples);
    let feature_ms = start.elapsed().as_secs_f64() * 1000.0;
    let inputs = [
        Tensor::new(Shape::new(vec![1, 64, frames]), TensorData::F32(features))?,
        Tensor::new(Shape::new(vec![1]), TensorData::I64(vec![frames as i64]))?,
    ];
    let budget = std::thread::available_parallelism().map_or(1, usize::from);
    let encoder_threads = (budget / 2).max(1);
    let runtime = cpu_factory().create(encoder_threads)?;
    let sessions = [
        runtime.load_session(&models.join("multilingual_ctc.int8.onnx"), true)?,
        runtime.load_session(&models.join("multilingual_ctc.int8.onnx"), true)?,
    ];
    for session in &sessions {
        std::hint::black_box(session.run(&inputs)?);
    }
    for cap in [1, 2] {
        for windows in [2, 8] {
            let mut runs = Vec::new();
            let start = Instant::now();
            for _ in 0..windows / cap {
                if cap == 1 {
                    let start = Instant::now();
                    std::hint::black_box(sessions[0].run(&inputs)?);
                    runs.push(start.elapsed().as_secs_f64() * 1000.0);
                } else {
                    let wave = std::thread::scope(|scope| -> anyhow::Result<Vec<f64>> {
                        let handles: Vec<_> = sessions
                            .iter()
                            .map(|session| {
                                scope.spawn(|| -> anyhow::Result<f64> {
                                    let start = Instant::now();
                                    std::hint::black_box(session.run(&inputs)?);
                                    Ok(start.elapsed().as_secs_f64() * 1000.0)
                                })
                            })
                            .collect();
                        handles
                            .into_iter()
                            .map(|h| {
                                h.join()
                                    .map_err(|_| anyhow::anyhow!("encoder worker panicked"))?
                            })
                            .collect()
                    })?;
                    runs.extend(wave);
                }
            }
            println!(
                "{}",
                json!({"kind":"native_encoder","cpu_budget":budget,"encoder_threads":encoder_threads,
                "pool_slots":2,"cap":cap,"windows":windows,"wall_ms":start.elapsed().as_secs_f64()*1000.0,
                "encoder_run_ms":runs,"feature_ms":feature_ms,"samples":samples.len(),"ort":ort::info()})
            );
        }
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    match std::env::var("GIGASTT_SCHEDULING_BENCH_MODE").as_deref() {
        Ok("launch") => launch_profile(),
        Ok("mixed") => mixed_profile()?,
        Ok("native") => native_profile()?,
        _ => eprintln!(
            "Set GIGASTT_SCHEDULING_BENCH_MODE=launch, mixed or native; see docs/file-window-scheduling.md"
        ),
    }
    Ok(())
}
