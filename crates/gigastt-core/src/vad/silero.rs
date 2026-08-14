//! Silero v5 ONNX wrapper.

use std::path::Path;

use anyhow::{Context, Result};
use parking_lot::Mutex;

use crate::runtime::{
    factory::RuntimeFactory,
    session::RuntimeSession,
    tensor::{Shape, Tensor, TensorData},
};

use super::{VAD_FRAME_SAMPLES, VAD_SAMPLE_RATE, VAD_STATE_LEN, VadConfig, regions_from_probs};

/// Silero v5 VAD model wrapped around the shared `ort` runtime.
///
/// The ONNX session is behind a [`Mutex`] because VAD runs off the hot decode
/// loop (either once per file or once per streamed chunk) and is not worth
/// pooling. The recurrent state is owned by the caller (per stream / per call),
/// never by this struct, so a single `SileroVad` can serve many concurrent
/// streams.
pub struct SileroVad {
    session: Mutex<Box<dyn RuntimeSession>>,
    /// Reusable input tensors: `[frame [1,512], state [2,1,128], sample_rate [1]]`.
    /// Mutated in place in `run_frame` to avoid per-frame allocations.
    input_tensors: Mutex<Vec<Tensor>>,
}

impl SileroVad {
    /// Load the Silero VAD ONNX model from `model_path`.
    ///
    /// # Errors
    ///
    /// Returns an error if the file is missing or `ort` fails to build the
    /// session. The caller treats an error as "VAD unavailable" and proceeds
    /// without it — VAD is strictly optional.
    pub fn load(model_path: &Path) -> Result<Self> {
        let factory = crate::runtime::cpu_factory();
        Self::load_with_factory(model_path, factory.as_ref())
    }

    /// Like [`SileroVad::load`], but loads the ONNX session through a
    /// caller-supplied `RuntimeFactory` (e.g. a non-`ort` backend or a test
    /// mock) instead of the default CPU `ort` runtime.
    pub fn load_with_factory(model_path: &Path, factory: &dyn RuntimeFactory) -> Result<Self> {
        tracing::debug!("Loading VAD model from {}", model_path.display());
        let runtime = factory
            .cpu_fallback()
            .create(1)
            .map_err(|e| anyhow::anyhow!(e))
            .context("Failed to create runtime for VAD model")?;
        let session = runtime
            .load_session(model_path, false)
            .map_err(|e| anyhow::anyhow!(e))
            .context("Failed to load VAD model")?;
        tracing::info!("VAD model loaded from {}", model_path.display());
        Ok(Self {
            session: Mutex::new(session),
            input_tensors: Mutex::new(vec![
                Tensor::new_checked(
                    Shape::new(vec![1, VAD_FRAME_SAMPLES]),
                    TensorData::F32(vec![0.0; VAD_FRAME_SAMPLES]),
                ),
                Tensor::new_checked(
                    Shape::new(vec![2, 1, 128]),
                    TensorData::F32(vec![0.0; VAD_STATE_LEN]),
                ),
                Tensor::new_checked(Shape::new(vec![1]), TensorData::I64(vec![VAD_SAMPLE_RATE])),
            ]),
        })
    }

    /// Run one fixed 512-sample window through the model, advancing `state`
    /// (the `[2, 1, 128]` recurrent tensor, flattened to [`VAD_STATE_LEN`]).
    /// Returns the speech probability in `[0, 1]`.
    ///
    /// `frame` shorter than [`VAD_FRAME_SAMPLES`] is zero-padded; longer is
    /// truncated, matching Silero's own contract.
    pub(crate) fn run_frame(&self, frame: &[f32], state: &mut [f32; VAD_STATE_LEN]) -> Result<f32> {
        let mut input = [0.0f32; VAD_FRAME_SAMPLES];
        let n = frame.len().min(VAD_FRAME_SAMPLES);
        input[..n].copy_from_slice(&frame[..n]);

        let outputs = {
            let mut inputs = self.input_tensors.lock();
            inputs[0]
                .as_f32_mut()
                .context("VAD frame tensor is not f32")?
                .copy_from_slice(&input);
            inputs[1]
                .as_f32_mut()
                .context("VAD state tensor is not f32")?
                .copy_from_slice(state);
            // inputs[2] (sample rate) is constant and was set at construction.

            let session = self.session.lock();
            session.run(&inputs).context("VAD model inference failed")?
        };

        // Identify the state and probability outputs by shape so the code
        // does not depend on the exact output order of the Silero model.
        let mut prob = 0.0f32;
        let mut new_state = [0.0f32; VAD_STATE_LEN];
        for output in outputs {
            let view = output.view();
            if let Some(data) = view.data().as_f32() {
                if data.len() == VAD_STATE_LEN {
                    new_state.copy_from_slice(data);
                } else if data.len() == 1 {
                    prob = data[0];
                }
            }
        }
        state.copy_from_slice(&new_state);
        Ok(prob)
    }

    /// Speech probability for every non-overlapping 512-sample window of
    /// `samples` (the trailing partial window, if any, is included zero-padded).
    pub fn frame_probs(&self, samples: &[f32]) -> Result<Vec<f32>> {
        self.frame_probs_with_abort(samples, None)
    }

    /// Like [`SileroVad::frame_probs`] but polls `abort` while scanning so a
    /// VAD pass over a whole long file can bail out cooperatively (a
    /// no-progress inference watchdog or a client cancellation flips the flag).
    /// Returns an error — which the caller maps to
    /// [`GigasttError::Cancelled`](crate::error::GigasttError::Cancelled) — when
    /// `abort` fires. With `abort = None` it is byte-for-byte `frame_probs`.
    pub(crate) fn frame_probs_with_abort(
        &self,
        samples: &[f32],
        abort: Option<&dyn Fn() -> bool>,
    ) -> Result<Vec<f32>> {
        let mut state = [0.0f32; VAD_STATE_LEN];
        let mut probs = Vec::with_capacity(samples.len() / VAD_FRAME_SAMPLES + 1);
        let mut i = 0;
        let mut since_check = 0usize;
        while i < samples.len() {
            // Poll the abort flag roughly every ~2 s of audio (64 × 512
            // samples) rather than once per 32 ms frame: interruptible on a
            // multi-minute scan without an atomic load in the inner loop.
            if let Some(abort) = abort {
                since_check += 1;
                if since_check >= 64 {
                    since_check = 0;
                    if abort() {
                        anyhow::bail!("cancelled");
                    }
                }
            }
            let end = (i + VAD_FRAME_SAMPLES).min(samples.len());
            probs.push(self.run_frame(&samples[i..end], &mut state)?);
            i = end;
        }
        Ok(probs)
    }

    /// Detect the speech spans of `samples` as `[start, end)` sample ranges
    /// (inclusive start, exclusive end) on the original timeline.
    ///
    /// Empty when no frame clears `cfg.threshold`.
    pub fn speech_regions(&self, samples: &[f32], cfg: &VadConfig) -> Result<Vec<(usize, usize)>> {
        self.speech_regions_with_abort(samples, cfg, None)
    }

    /// Like [`SileroVad::speech_regions`] but threads an `abort` poll into the
    /// underlying frame scan. With `abort = None` it is byte-for-byte
    /// `speech_regions`.
    pub(crate) fn speech_regions_with_abort(
        &self,
        samples: &[f32],
        cfg: &VadConfig,
        abort: Option<&dyn Fn() -> bool>,
    ) -> Result<Vec<(usize, usize)>> {
        let probs = self.frame_probs_with_abort(samples, abort)?;
        Ok(regions_from_probs(
            &probs,
            VAD_FRAME_SAMPLES,
            samples.len(),
            cfg,
        ))
    }
}
