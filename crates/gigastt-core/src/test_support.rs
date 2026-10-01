//! Model-free test engines for in-crate tests and the private `__internals`
//! feature (server / FFI unit tests).
//!
//! Not part of the stable public API. The encoder session accepts any audio
//! length and emits a single blank frame so REST / SSE / jobs / file wrappers
//! can run without ONNX weights.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::error::GigasttError;
use crate::inference::{Engine, PRED_HIDDEN};
use crate::runtime::mock::{MockFactory, MockSession};
use crate::runtime::tensor::{Shape, Tensor, TensorData};

const ENC_DIM: usize = 768;

/// Write the INT8 rnnt filenames the engine loader expects (empty ONNX bytes).
pub fn write_rnnt_layout(dir: &Path) -> std::io::Result<()> {
    std::fs::write(dir.join("v3_rnnt_encoder_int8.onnx"), b"")?;
    std::fs::write(dir.join("v3_rnnt_decoder.onnx"), b"")?;
    std::fs::write(dir.join("v3_rnnt_joint.onnx"), b"")?;
    std::fs::write(dir.join("v3_vocab.txt"), "\u{2581}hi\n<blk>\n")?;
    Ok(())
}

/// PCM16 mono WAV with a standard 44-byte header.
pub fn pcm16_wav(samples: &[i16], sample_rate: u32) -> Vec<u8> {
    let data_size = (samples.len() * 2) as u32;
    let file_size = 36 + data_size;
    let mut wav = Vec::with_capacity(44 + samples.len() * 2);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&file_size.to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_size.to_le_bytes());
    for s in samples {
        wav.extend_from_slice(&s.to_le_bytes());
    }
    wav
}

/// Scripted rnnt factory: unconstrained encoder (any T) + fixed decoder/joiner.
pub fn rnnt_factory() -> MockFactory {
    let mut sessions: HashMap<String, Arc<MockSession>> = HashMap::new();
    sessions.insert(
        "v3_rnnt_encoder_int8".into(),
        Arc::new(MockSession::unconstrained(vec![
            Tensor::new(
                Shape::new(vec![1, ENC_DIM, 1]),
                TensorData::F32(vec![0.0; ENC_DIM]),
            )
            .expect("encoder out"),
            Tensor::new(Shape::new(vec![1]), TensorData::I64(vec![1])).expect("enc len"),
        ])),
    );
    sessions.insert(
        "v3_rnnt_decoder".into(),
        Arc::new(MockSession::new(
            vec![
                Shape::new(vec![1, 1]),
                Shape::new(vec![1, 1, PRED_HIDDEN]),
                Shape::new(vec![1, 1, PRED_HIDDEN]),
            ],
            vec![
                Tensor::new(
                    Shape::new(vec![1, 1, PRED_HIDDEN]),
                    TensorData::F32(vec![0.0; PRED_HIDDEN]),
                )
                .expect("dec h"),
                Tensor::new(
                    Shape::new(vec![1, 1, PRED_HIDDEN]),
                    TensorData::F32(vec![0.0; PRED_HIDDEN]),
                )
                .expect("dec c"),
                Tensor::new(
                    Shape::new(vec![1, 1, PRED_HIDDEN]),
                    TensorData::F32(vec![0.0; PRED_HIDDEN]),
                )
                .expect("dec out"),
            ],
        )),
    );
    sessions.insert(
        "v3_rnnt_joint".into(),
        Arc::new(MockSession::new(
            vec![
                Shape::new(vec![1, ENC_DIM, 1]),
                Shape::new(vec![1, PRED_HIDDEN, 1]),
            ],
            vec![
                Tensor::new(Shape::new(vec![1, 1, 2]), TensorData::F32(vec![0.0; 2]))
                    .expect("joint"),
            ],
        )),
    );
    MockFactory::new(sessions)
}

/// Load an INT8 rnnt engine from `dir` (must already hold [`write_rnnt_layout`]).
pub fn load_rnnt_engine(dir: &Path, pool_size: usize) -> Result<Engine, GigasttError> {
    load_rnnt_engine_with_factory(dir, pool_size, Box::new(rnnt_factory()))
}

/// Load a model-free engine with custom scripted sessions for server tests.
pub fn load_rnnt_engine_with_factory(
    dir: &Path,
    pool_size: usize,
    factory: Box<dyn crate::runtime::factory::RuntimeFactory>,
) -> Result<Engine, GigasttError> {
    Engine::load_with_factory(dir, None, pool_size.max(1), 1, 0, factory, 1, true, "cpu")
}

/// Private scheduling benchmark loader with a caller-controlled mock runtime.
/// Keeps production engine loading and pool partitioning in the measured path.
pub fn load_scheduling_engine(
    dir: &Path,
    pool_size: usize,
    batch_pool_size: usize,
    factory: Box<dyn crate::runtime::RuntimeFactory>,
) -> Result<Engine, GigasttError> {
    Engine::load_with_factory(
        dir,
        None,
        pool_size,
        1,
        batch_pool_size,
        factory,
        1,
        false,
        "cpu",
    )
}

/// Convenience for this crate's own unit tests (`tempfile` is a dev-dep).
#[cfg(test)]
pub fn rnnt_engine() -> (Engine, tempfile::TempDir) {
    let tmp = tempfile::tempdir().expect("tempdir");
    write_rnnt_layout(tmp.path()).expect("rnnt layout");
    let engine = load_rnnt_engine(tmp.path(), 1).expect("mock rnnt engine");
    (engine, tmp)
}

/// Model-free benchmark driver for file-window snapshot publication.
#[derive(Default)]
pub struct SnapshotBenchPublisher {
    snapshot: crate::inference::TranscriptSnapshot,
    publisher: crate::inference::SnapshotPublisher,
}

impl SnapshotBenchPublisher {
    /// Publish the suffix changed by a synthetic window seam.
    pub fn publish(
        &mut self,
        words: &[crate::inference::WordInfo],
        retained: usize,
        channel: Option<usize>,
    ) {
        self.publisher.publish(
            &self.snapshot,
            retained,
            words[retained..].to_vec(),
            channel,
            0.0,
        );
    }

    /// Materialize the current complete owned snapshot.
    pub fn get(&self) -> Option<crate::inference::TranscriptSegment> {
        self.snapshot.get()
    }
}

/// Model-free runtime with an armed encoder/joiner failure on a chosen window.
/// Arm after loading to keep warmup independent of the failure scenario.
#[derive(Clone)]
pub struct FailingStreamFactory {
    windows: Arc<std::sync::atomic::AtomicUsize>,
    fail_at: Arc<std::sync::atomic::AtomicUsize>,
    joiner: bool,
    panic: bool,
}
impl FailingStreamFactory {
    pub fn new(joiner: bool, panic: bool) -> Self {
        Self {
            windows: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            fail_at: Arc::new(std::sync::atomic::AtomicUsize::new(usize::MAX)),
            joiner,
            panic,
        }
    }
    pub fn arm(&self, window: usize) {
        use std::sync::atomic::Ordering::SeqCst;
        self.windows.store(0, SeqCst);
        self.fail_at.store(window, SeqCst);
    }
}
impl crate::runtime::factory::RuntimeFactory for FailingStreamFactory {
    fn create(
        &self,
        threads: usize,
    ) -> Result<Box<dyn crate::runtime::factory::Runtime>, crate::runtime::error::RuntimeError>
    {
        Ok(Box::new(FailingStreamRuntime {
            inner: rnnt_factory().create(threads)?,
            factory: self.clone(),
        }))
    }
    fn cpu_fallback(&self) -> Box<dyn crate::runtime::factory::RuntimeFactory> {
        Box::new(self.clone())
    }
    fn verify_on_disk_checksums(&self) -> bool {
        false
    }
}
struct FailingStreamRuntime {
    inner: Box<dyn crate::runtime::factory::Runtime>,
    factory: FailingStreamFactory,
}
impl crate::runtime::factory::Runtime for FailingStreamRuntime {
    fn load_session(
        &self,
        path: &Path,
        encoder: bool,
    ) -> Result<Box<dyn crate::runtime::session::RuntimeSession>, crate::runtime::error::RuntimeError>
    {
        let inner = self.inner.load_session(path, encoder)?;
        let joiner = path.file_stem().is_some_and(|s| s == "v3_rnnt_joint");
        if encoder || joiner {
            Ok(Box::new(FailingStreamSession {
                inner,
                factory: self.factory.clone(),
                encoder,
            }))
        } else {
            Ok(inner)
        }
    }
}
struct FailingStreamSession {
    inner: Box<dyn crate::runtime::session::RuntimeSession>,
    factory: FailingStreamFactory,
    encoder: bool,
}
impl crate::runtime::session::RuntimeSession for FailingStreamSession {
    fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, crate::runtime::error::RuntimeError> {
        use std::sync::atomic::Ordering::SeqCst;
        if self.encoder {
            self.factory.windows.fetch_add(1, SeqCst);
        }
        if self.encoder != self.factory.joiner
            && self.factory.windows.load(SeqCst) >= self.factory.fail_at.load(SeqCst)
        {
            assert!(!self.factory.panic, "injected tail inference panic");
            return Err(crate::runtime::error::RuntimeError::InferenceFailed(
                "injected tail inference failure".into(),
            ));
        }
        if !self.encoder {
            return Ok(vec![
                Tensor::new(Shape::new(vec![1, 1, 2]), TensorData::F32(vec![10.0, 0.0]))
                    .expect("test logits"),
            ]);
        }
        self.inner.run(inputs)
    }
}
