//! Installed-model equivalence and timing decomposition for output workspaces.
use super::*;
use crate::runtime::{Runtime, RuntimeError, RuntimeFactory, RuntimeSession, Tensor};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Clone, Copy, Default, Debug)]
struct Timing {
    calls: usize,
    nanos: u128,
}
type Timings = Arc<Mutex<[Timing; 3]>>;

struct LegacySession(Box<dyn RuntimeSession>);
impl RuntimeSession for LegacySession {
    fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, RuntimeError> {
        self.0.run(inputs)
    }
}
struct TimedSession {
    inner: Box<dyn RuntimeSession>,
    timings: Timings,
    stage: usize,
}
impl TimedSession {
    fn record(&self, start: Instant) {
        let elapsed = start.elapsed().as_nanos();
        let mut stats = self.timings.lock().unwrap();
        stats[self.stage].calls += 1;
        stats[self.stage].nanos += elapsed;
    }
}
impl RuntimeSession for TimedSession {
    fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, RuntimeError> {
        let start = Instant::now();
        let result = self.inner.run(inputs);
        self.record(start);
        result
    }
    fn run_f32_into(
        &self,
        inputs: &[Tensor],
        outputs: &mut [&mut Vec<f32>],
    ) -> Result<(), RuntimeError> {
        let start = Instant::now();
        let result = self.inner.run_f32_into(inputs, outputs);
        self.record(start);
        result
    }
}
struct MeasuredFactory {
    legacy: bool,
    timings: Timings,
}
struct MeasuredRuntime {
    inner: Box<dyn Runtime>,
    legacy: bool,
    timings: Timings,
}
impl RuntimeFactory for MeasuredFactory {
    fn create(&self, threads: usize) -> Result<Box<dyn Runtime>, RuntimeError> {
        Ok(Box::new(MeasuredRuntime {
            inner: crate::runtime::cpu_factory().create(threads)?,
            legacy: self.legacy,
            timings: self.timings.clone(),
        }))
    }
    fn cpu_fallback(&self) -> Box<dyn RuntimeFactory> {
        Box::new(Self {
            legacy: self.legacy,
            timings: self.timings.clone(),
        })
    }
}
impl Runtime for MeasuredRuntime {
    fn load_session(
        &self,
        path: &Path,
        is_encoder: bool,
    ) -> Result<Box<dyn RuntimeSession>, RuntimeError> {
        let inner = self.inner.load_session(path, is_encoder)?;
        let inner: Box<dyn RuntimeSession> = if self.legacy {
            Box::new(LegacySession(inner))
        } else {
            inner
        };
        Ok(Box::new(TimedSession {
            inner,
            timings: self.timings.clone(),
            stage: if is_encoder {
                0
            } else if path.to_string_lossy().contains("decoder") {
                1
            } else {
                2
            },
        }))
    }
}

#[test]
#[ignore = "full pipeline equivalence and timing; skips heads without an installed INT8 encoder"]
fn test_installed_runtime_workspace_equivalence() {
    use crate::inference::{DecoderState, decode::greedy_decode};
    use crate::runtime::tensor::{Shape, TensorDataView};
    use std::sync::atomic::{AtomicUsize, Ordering};
    let model_dir = crate::model::default_model_dir();
    for variant in [
        crate::model::ModelVariant::Rnnt,
        crate::model::ModelVariant::E2eRnnt,
    ] {
        if !Path::new(&model_dir)
            .join(variant.encoder_int8_file())
            .is_file()
        {
            eprintln!(
                "skip {variant:?}: INT8 encoder is not installed; this test never downloads models"
            );
            continue;
        }
        let measurements: Vec<_> = [true, false]
            .into_iter()
            .map(|legacy| {
                let timings = Arc::new(Mutex::new([Timing::default(); 3]));
                let engine = Engine::load_with_factory(
                    Path::new(&model_dir),
                    Some(variant),
                    1,
                    1,
                    0,
                    Box::new(MeasuredFactory {
                        legacy,
                        timings: timings.clone(),
                    }),
                    4,
                    true,
                    "cpu",
                )
                .unwrap();
                (engine, timings)
            })
            .collect();
        for fixture in ["golos_00.wav", "golos_01.wav"] {
            let path = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../gigastt/tests/fixtures")
                .join(fixture);
            let path = path.to_str().unwrap();
            let mut reference = None;
            for iteration in 0..7 {
                let order = if iteration % 2 == 0 { [0, 1] } else { [1, 0] };
                for mode in order {
                    let (engine, timings) = &measurements[mode];
                    let mut guard = engine.pool.checkout_blocking().unwrap();
                    *timings.lock().unwrap() = [Timing::default(); 3];
                    let start = Instant::now();
                    let result = engine.transcribe_file(path, &mut guard).unwrap();
                    let elapsed = start.elapsed();
                    let serialized = serde_json::to_value(&result).unwrap();
                    if let Some(expected) = &reference {
                        assert_eq!(&serialized, expected);
                    } else {
                        reference = Some(serialized);
                    }
                    if iteration > 0 {
                        eprintln!(
                            "{variant:?} {fixture} legacy={} iteration={iteration} total_us={} audio_s={} stages_encoder_decoder_joiner={:?}",
                            mode == 0,
                            elapsed.as_micros(),
                            result.duration_s,
                            *timings.lock().unwrap()
                        );
                    }
                }
            }
            // Compare the exact same encoded activations through both decode
            // paths, including cancellation between runtime calls.
            let samples = crate::inference::audio::decode_audio_file(path).unwrap();
            let (mel, frames) = measurements[0].0.features.compute(&samples);
            let mut legacy = measurements[0].0.pool.checkout_blocking().unwrap();
            let optimized = measurements[1].0.pool.checkout_blocking().unwrap();
            legacy.encoder_inputs[0].resize_to(Shape::new(vec![1, 64, frames]));
            legacy.encoder_inputs[0]
                .as_f32_mut()
                .unwrap()
                .copy_from_slice(&mel);
            legacy.encoder_inputs[1].as_i64_mut().unwrap()[0] = frames as i64;
            let encoded = legacy.encoder.run(&legacy.encoder_inputs).unwrap();
            let length = match encoded[1].view().data() {
                TensorDataView::I64(v) => v[0] as usize,
                TensorDataView::I32(v) => v[0] as usize,
                _ => panic!("invalid length"),
            };
            let blank = measurements[0].0.tokenizer.blank_id();
            for abort_after in [3, usize::MAX] {
                let mut previous = None;
                for triplet in [&*legacy, &*optimized] {
                    let checks = AtomicUsize::new(0);
                    let abort = || checks.fetch_add(1, Ordering::Relaxed) >= abort_after;
                    let mut state = DecoderState::new(blank);
                    let result = greedy_decode(
                        triplet.decoder.as_deref().unwrap(),
                        triplet.joiner.as_deref().unwrap(),
                        &encoded[0].view(),
                        length,
                        blank,
                        &mut state,
                        None,
                        Some(&abort),
                    )
                    .unwrap();
                    if abort_after == usize::MAX {
                        eprintln!(
                            "{variant:?} {fixture} full_decode_emitted_tokens={}",
                            result.tokens.len()
                        );
                    }
                    let snapshot = (
                        result
                            .tokens
                            .iter()
                            .map(|t| (t.token_id, t.frame_index, t.confidence.to_bits()))
                            .collect::<Vec<_>>(),
                        result.endpoint_detected,
                        state.h,
                        state.c,
                        state.prev_token,
                        state.consecutive_blanks,
                        checks.load(Ordering::Relaxed),
                    );
                    if let Some(expected) = &previous {
                        assert_eq!(&snapshot, expected);
                    } else {
                        previous = Some(snapshot);
                    }
                }
            }
        }
    }
}
