//! Voice activity detection (VAD) via the Silero v5 ONNX model.
//!
//! Used for two optional, opt-in features on top of the recognition engine:
//!
//! 1. **File silence skipping** — [`SileroVad::speech_regions`] returns the
//!    speech spans of a clip so the engine can decode only those, skipping long
//!    pauses. Speedup is proportional to the silence fraction.
//! 2. **Streaming endpointing** — [`VadEndpointer`] tracks trailing silence
//!    across streamed chunks and signals when an utterance has ended. When a
//!    VAD is attached it owns endpointing: the decoder's blank-run heuristic
//!    is ignored, so `min_silence_ms` fully controls finalization.
//!
//! The model is loaded through the same `ort` runtime the recognition engine
//! already uses (no extra dependency, no second ONNX Runtime). The Silero v5
//! graph (opset 16, conv + LSTM) takes a fixed 512-sample window at 16 kHz plus
//! a recurrent state tensor `[2, 1, 128]`, and returns a speech probability in
//! `[0, 1]` together with the next state.
//!
//! All of the segmentation / endpointing decision logic is split into pure
//! functions ([`regions_from_probs`], [`Hangover`]) so it can be unit-tested on
//! synthetic probability sequences without loading the model.

use std::path::Path;

use anyhow::{Context, Result};
use parking_lot::Mutex;

use crate::runtime::{
    factory::RuntimeFactory,
    session::RuntimeSession,
    tensor::{Shape, Tensor, TensorData},
};

/// Silero VAD ONNX filename on disk. Single source of truth shared with the
/// model-download path in [`crate::model`].
pub const VAD_MODEL_FILE: &str = "silero_vad.onnx";

/// Sample rate the engine (and Silero) operate at.
pub const VAD_SAMPLE_RATE: i64 = 16000;

/// Fixed Silero v5 window at 16 kHz (~32 ms). The model only accepts this size.
pub const VAD_FRAME_SAMPLES: usize = 512;

/// Length of the Silero recurrent state tensor (`[2, 1, 128]` flattened).
const VAD_STATE_LEN: usize = 2 * 128;

/// Tunable thresholds for turning a per-frame speech-probability sequence into
/// speech spans (file path) and endpoint decisions (streaming).
#[derive(Debug, Clone, Copy)]
pub struct VadConfig {
    /// Speech-probability threshold in `[0, 1]`; frames at or above are speech.
    pub threshold: f32,
    /// Minimum trailing silence before a speech region is closed / an utterance
    /// is considered ended (endpointing).
    pub min_silence_ms: u32,
    /// Speech runs shorter than this are dropped as noise (file path only).
    pub min_speech_ms: u32,
    /// Padding added on each side of a kept speech region so onsets/offsets are
    /// not clipped (file path only).
    pub speech_pad_ms: u32,
}

impl Default for VadConfig {
    fn default() -> Self {
        // Silero's own defaults, lightly adapted: 0.5 threshold, ~500 ms of
        // silence to close a turn, 250 ms minimum speech, 100 ms pad.
        Self {
            threshold: 0.5,
            min_silence_ms: 500,
            min_speech_ms: 250,
            speech_pad_ms: 100,
        }
    }
}

impl VadConfig {
    fn ms_to_samples(ms: u32) -> usize {
        (VAD_SAMPLE_RATE as usize * ms as usize) / 1000
    }
}

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
    fn run_frame(&self, frame: &[f32], state: &mut [f32; VAD_STATE_LEN]) -> Result<f32> {
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

/// Turn a per-frame speech-probability sequence into merged `[start, end)`
/// speech-sample spans. Pure (no model) so it is unit-testable on synthetic
/// probabilities.
///
/// `frame_samples` is the samples-per-probability stride ([`VAD_FRAME_SAMPLES`]
/// in production); `total_samples` clamps the final span to the real signal
/// length. Applies, in order: threshold, min-silence merge (gaps shorter than
/// `min_silence_ms` do not split a region), min-speech drop, and symmetric
/// `speech_pad_ms` padding (clamped to `[0, total_samples]`, then re-merged if
/// padding makes neighbours overlap).
pub fn regions_from_probs(
    probs: &[f32],
    frame_samples: usize,
    total_samples: usize,
    cfg: &VadConfig,
) -> Vec<(usize, usize)> {
    if probs.is_empty() || total_samples == 0 {
        return Vec::new();
    }

    let min_silence = VadConfig::ms_to_samples(cfg.min_silence_ms);
    let min_speech = VadConfig::ms_to_samples(cfg.min_speech_ms);
    let pad = VadConfig::ms_to_samples(cfg.speech_pad_ms);

    // 1. Raw speech runs from the thresholded probabilities.
    let mut regions: Vec<(usize, usize)> = Vec::new();
    let mut run_start: Option<usize> = None;
    for (i, &p) in probs.iter().enumerate() {
        let speech = p >= cfg.threshold;
        if speech && run_start.is_none() {
            run_start = Some(i * frame_samples);
        } else if !speech && let Some(s) = run_start.take() {
            regions.push((s, i * frame_samples));
        }
    }
    if let Some(s) = run_start.take() {
        regions.push((s, total_samples));
    }
    if regions.is_empty() {
        return regions;
    }

    // 2. Merge regions separated by a silence gap shorter than min_silence.
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(regions.len());
    for (s, e) in regions {
        match merged.last_mut() {
            Some(last) if s.saturating_sub(last.1) < min_silence => last.1 = e,
            _ => merged.push((s, e)),
        }
    }

    // 3. Drop regions shorter than min_speech (measured before padding).
    merged.retain(|(s, e)| e - s >= min_speech);
    if merged.is_empty() {
        return merged;
    }

    // 4. Pad each side, clamp to the signal, then re-merge any overlaps the
    //    padding introduced.
    let mut padded: Vec<(usize, usize)> = Vec::with_capacity(merged.len());
    for (s, e) in merged {
        let ps = s.saturating_sub(pad);
        let pe = (e + pad).min(total_samples);
        match padded.last_mut() {
            Some(last) if ps <= last.1 => last.1 = last.1.max(pe),
            _ => padded.push((ps, pe)),
        }
    }
    padded
}

/// Map a timestamp on the compressed (silence-removed) timeline back to the
/// original timeline, given the kept speech `regions` (original `[start, end)`
/// sample ranges, in order) and `sample_rate`. Pure — unit-tested directly.
///
/// File transcription with VAD decodes a buffer formed by concatenating the
/// speech regions, so decoded word timestamps are in compressed time; this
/// undoes that compression. A time at or past the end of all regions clamps to
/// the last region's end (guards rounding past the final frame).
pub fn remap_compressed_seconds(
    t_compressed_s: f64,
    regions: &[(usize, usize)],
    sample_rate: f64,
) -> f64 {
    if regions.is_empty() {
        return t_compressed_s;
    }
    let target = (t_compressed_s * sample_rate).max(0.0);
    let mut acc = 0.0f64; // compressed-sample offset at the current region's start
    for &(s, e) in regions {
        let len = (e - s) as f64;
        if target <= acc + len {
            let into = (target - acc).max(0.0);
            return (s as f64 + into) / sample_rate;
        }
        acc += len;
    }
    let &(_, end) = regions.last().expect("non-empty checked above");
    end as f64 / sample_rate
}

/// Causal, bounded-memory form of the file-VAD pipeline: the frame scan of
/// [`SileroVad::speech_regions`] plus the silence-free concatenation the engine
/// builds from its output.
///
/// The batch pair needs the whole 16 kHz buffer resident — once to score every
/// frame, once to copy the kept spans out — which is what pinned the VAD file
/// path to a duration ceiling. Silero is already causal (its recurrent state
/// carries frame to frame), and every decision [`regions_from_probs`] makes is
/// settled by a *bounded* look-ahead: a region can still absorb the next speech
/// run for `min_silence_ms`, can still be dropped for falling short of
/// `min_speech_ms`, and can still merge with its neighbour across
/// `2 × speech_pad_ms`. So samples go in, whole frames are scored as they
/// complete, and kept audio is released as soon as the look-ahead that decides
/// it has passed — roughly `min_speech_ms + min_silence_ms + speech_pad_ms` of
/// PCM held at a time (~850 ms at the defaults), whatever the file's length.
///
/// The result is byte-identical to the batch path, not merely equivalent:
/// [`VadSegmenter::regions`] after `finish` equals [`regions_from_probs`] over
/// the same frames, and the samples appended to `out` are exactly that
/// concatenation. Both are asserted directly, including a proptest over random
/// probability sequences and configs.
///
/// Only the file path needs it, so it is gated on `file-decode`: a lean build
/// has no file VAD at all, only [`VadEndpointer`] for streams.
#[cfg(feature = "file-decode")]
pub(crate) struct VadSegmenter {
    threshold: f32,
    min_silence: usize,
    min_speech: usize,
    pad: usize,
    /// Silero recurrent state, carried across pushes.
    state: [f32; VAD_STATE_LEN],
    /// Retained 16 kHz samples; `raw[0]` is absolute sample `raw_start`.
    raw: Vec<f32>,
    raw_start: usize,
    /// Absolute index one past the last scored frame.
    pos: usize,
    /// Inside a raw (thresholded) speech run.
    in_run: bool,
    /// Merged region under construction: `(start, end of its last speech frame)`.
    merged: Option<(usize, usize)>,
    /// Index in `regions` of the entry the open region extends, set once that
    /// region is long enough that the `min_speech_ms` drop can no longer take it.
    open_idx: Option<usize>,
    /// Kept, padded spans in order. Only the last one can still grow.
    regions: Vec<(usize, usize)>,
    /// Next `regions` entry to release samples from.
    out_idx: usize,
    /// Absolute index already copied into the caller's compressed buffer.
    copied_to: usize,
}

#[cfg(feature = "file-decode")]
impl VadSegmenter {
    /// New segmenter for `cfg`, at absolute sample 0.
    pub(crate) fn new(cfg: &VadConfig) -> Self {
        Self {
            threshold: cfg.threshold,
            min_silence: VadConfig::ms_to_samples(cfg.min_silence_ms),
            min_speech: VadConfig::ms_to_samples(cfg.min_speech_ms),
            pad: VadConfig::ms_to_samples(cfg.speech_pad_ms),
            state: [0.0; VAD_STATE_LEN],
            raw: Vec::new(),
            raw_start: 0,
            pos: 0,
            in_run: false,
            merged: None,
            open_idx: None,
            regions: Vec::new(),
            out_idx: 0,
            copied_to: 0,
        }
    }

    /// Kept spans as `[start, end)` on the **original** timeline, in order.
    /// Complete only after [`VadSegmenter::finish`]; before that the last entry
    /// can still grow.
    pub(crate) fn regions(&self) -> &[(usize, usize)] {
        &self.regions
    }

    /// Feed the next contiguous block of 16 kHz samples, appending everything
    /// the VAD has committed to keeping to `out`.
    pub(crate) fn push(
        &mut self,
        vad: &SileroVad,
        samples: &[f32],
        out: &mut Vec<f32>,
    ) -> Result<()> {
        self.push_with(samples, out, |frame, state| vad.run_frame(frame, state))
    }

    /// Close the stream at `total` absolute samples and release the rest.
    pub(crate) fn finish(
        &mut self,
        vad: &SileroVad,
        total: usize,
        out: &mut Vec<f32>,
    ) -> Result<()> {
        self.finish_with(total, out, |frame, state| vad.run_frame(frame, state))
    }

    /// [`VadSegmenter::push`] with the per-frame scorer injected, so the pure
    /// decision logic can be driven from a probability sequence in tests
    /// without loading Silero.
    fn push_with<F>(&mut self, samples: &[f32], out: &mut Vec<f32>, mut score: F) -> Result<()>
    where
        F: FnMut(&[f32], &mut [f32; VAD_STATE_LEN]) -> Result<f32>,
    {
        self.raw.extend_from_slice(samples);
        let avail = self.raw_start + self.raw.len();
        while self.pos + VAD_FRAME_SAMPLES <= avail {
            let off = self.pos - self.raw_start;
            let prob = score(&self.raw[off..off + VAD_FRAME_SAMPLES], &mut self.state)?;
            self.step(prob, self.pos + VAD_FRAME_SAMPLES);
        }
        self.flush(out);
        self.trim();
        Ok(())
    }

    /// [`VadSegmenter::finish`] with the per-frame scorer injected.
    fn finish_with<F>(&mut self, total: usize, out: &mut Vec<f32>, mut score: F) -> Result<()>
    where
        F: FnMut(&[f32], &mut [f32; VAD_STATE_LEN]) -> Result<f32>,
    {
        // `frame_probs` scores a trailing partial window zero-padded; so does
        // this. That frame's timeline stops at `total`, not at the padded frame
        // boundary — `regions_from_probs` measures the last run against
        // `total_samples` the same way, and a region released early must not be
        // credited with samples the signal does not have.
        while self.pos < total {
            let off = self.pos - self.raw_start;
            let end = (off + VAD_FRAME_SAMPLES).min(self.raw.len());
            let prob = score(&self.raw[off..end], &mut self.state)?;
            self.step(prob, (self.pos + VAD_FRAME_SAMPLES).min(total));
        }
        self.close(total);
        self.flush(out);
        Ok(())
    }

    /// Settle the tail: a run still open at EOF ends at the true signal length
    /// (not at the padded frame boundary), the open region is finalized, and
    /// every span is clamped to `total` — exactly what `regions_from_probs`
    /// does with its `total_samples` argument.
    fn close(&mut self, total: usize) {
        if self.in_run
            && let Some(m) = self.merged.as_mut()
        {
            m.1 = total;
            self.in_run = false;
        }
        if let Some((ms, me)) = self.merged.take() {
            self.finalize(ms, me);
        }
        for r in &mut self.regions {
            r.1 = r.1.min(total);
        }
        self.pos = self.pos.max(total);
    }

    /// Advance the timeline to `end` with the scored frame's speech probability
    /// `prob`. `end` is the frame boundary except for the trailing partial
    /// frame, where it is the true signal length.
    fn step(&mut self, prob: f32, end: usize) {
        let start = self.pos;
        if prob >= self.threshold {
            if !self.in_run {
                match self.merged {
                    // A gap shorter than `min_silence` does not split a region.
                    Some((_, me)) if start.saturating_sub(me) < self.min_silence => {}
                    Some((ms, me)) => {
                        self.finalize(ms, me);
                        self.merged = None;
                    }
                    None => {}
                }
                if self.merged.is_none() {
                    self.merged = Some((start, start));
                }
                self.in_run = true;
            }
            if let Some(m) = self.merged.as_mut() {
                m.1 = end;
            }
            // Once the region clears `min_speech` the drop can no longer take
            // it, so its audio is released without waiting for it to close —
            // this is what keeps an hour of unbroken speech from being buffered.
            if let Some((ms, me)) = self.merged {
                match self.open_idx {
                    Some(i) => self.regions[i].1 = self.regions[i].1.max(me),
                    None if me - ms >= self.min_speech => {
                        self.open_idx = Some(self.push_padded(ms.saturating_sub(self.pad), me));
                    }
                    None => {}
                }
            }
        } else {
            self.in_run = false;
            // The earliest a later run can start is `end`, so once that is
            // `min_silence` past the region's end nothing can merge into it.
            if let Some((ms, me)) = self.merged
                && end.saturating_sub(me) >= self.min_silence
            {
                self.finalize(ms, me);
                self.merged = None;
            }
        }
        self.pos = end;
    }

    /// Append a padded span, merging it into the previous one when the padding
    /// makes them touch. Mirrors step 4 of [`regions_from_probs`]; returns the
    /// index of the entry that now covers `[ps, pe)`.
    fn push_padded(&mut self, ps: usize, pe: usize) -> usize {
        match self.regions.last_mut() {
            Some(last) if ps <= last.1 => last.1 = last.1.max(pe),
            _ => self.regions.push((ps, pe)),
        }
        self.regions.len() - 1
    }

    /// Commit a closed merged region `[ms, me)`: dropped when shorter than
    /// `min_speech`, otherwise padded and merged into the output list.
    fn finalize(&mut self, ms: usize, me: usize) {
        let idx = self.open_idx.take();
        if me - ms < self.min_speech {
            return;
        }
        let pe = me + self.pad;
        match idx {
            Some(i) => self.regions[i].1 = self.regions[i].1.max(pe),
            None => {
                self.push_padded(ms.saturating_sub(self.pad), pe);
            }
        }
    }

    /// Absolute index up to which membership in `regions` is final.
    fn decided_to(&self) -> usize {
        let base = self.regions.last().map_or(0, |r| r.1);
        let bound = match self.merged {
            // The open region is certain to survive and `regions.last()` already
            // tracks how far it reaches.
            Some(_) if self.open_idx.is_some() => base,
            // Still undecided from its padded start on.
            Some((ms, _)) => base.max(ms.saturating_sub(self.pad)),
            // Silence: a later region's padded start is at least `pos - pad`.
            None => base.max(self.pos.saturating_sub(self.pad)),
        };
        bound.min(self.pos)
    }

    /// Copy every decided, not-yet-released kept sample into `out`.
    fn flush(&mut self, out: &mut Vec<f32>) {
        let decided = self.decided_to().min(self.raw_start + self.raw.len());
        while self.out_idx < self.regions.len() {
            let (s, e) = self.regions[self.out_idx];
            let from = self.copied_to.max(s);
            let to = e.min(decided);
            if to > from {
                // `trim` only ever drops PCM below what a still-growing span can
                // reach back to; if that ever stops holding, say so here rather
                // than underflowing into an opaque index panic.
                debug_assert!(
                    from >= self.raw_start,
                    "released PCM at {from} below the retained start {}",
                    self.raw_start
                );
                out.extend_from_slice(&self.raw[from - self.raw_start..to - self.raw_start]);
                self.copied_to = to;
            }
            if self.out_idx + 1 < self.regions.len() {
                self.out_idx += 1;
            } else {
                break;
            }
        }
    }

    /// Drop the retained PCM that can no longer be needed. Everything below the
    /// decided watermark has been released already; an undecided open region
    /// still needs its padded start.
    fn trim(&mut self) {
        let decided = self.decided_to();
        let keep_from = match self.merged {
            Some((ms, _)) if self.open_idx.is_none() => decided.min(ms.saturating_sub(self.pad)),
            _ => decided,
        };
        let drop = keep_from.saturating_sub(self.raw_start).min(self.raw.len());
        if drop > 0 {
            self.raw.drain(..drop);
            self.raw_start += drop;
        }
    }

    /// Retained PCM, in samples. Test-only: the bound on this is the whole point.
    #[cfg(test)]
    fn retained(&self) -> usize {
        self.raw.len()
    }
}

/// Streaming endpoint detector: feeds streamed audio through the VAD in fixed
/// frames, tracks trailing silence, and reports when an utterance has ended
/// (≥ `min_silence_ms` of silence *after* speech was seen).
///
/// Owns its recurrent state and a small leftover buffer so callers can push
/// arbitrary chunk sizes. The threshold/silence logic is exercised directly in
/// tests via [`Hangover`].
pub struct VadEndpointer {
    state: [f32; VAD_STATE_LEN],
    leftover: Vec<f32>,
    hangover: Hangover,
}

impl VadEndpointer {
    /// New endpointer for the given config.
    pub fn new(cfg: &VadConfig) -> Self {
        Self {
            state: [0.0f32; VAD_STATE_LEN],
            leftover: Vec::with_capacity(VAD_FRAME_SAMPLES),
            hangover: Hangover::new(cfg),
        }
    }

    /// Feed a chunk of 16 kHz samples. Returns `true` exactly once per utterance
    /// when trailing silence first crosses `min_silence_ms` after speech — the
    /// caller should finalize the current segment. Resets internally so the next
    /// speech run can trigger again.
    ///
    /// On model inference failure the chunk is treated as non-endpointing (logged
    /// by the caller) so streaming is never blocked by VAD.
    pub fn push(&mut self, vad: &SileroVad, samples: &[f32]) -> Result<bool> {
        self.leftover.extend_from_slice(samples);
        let mut endpoint = false;
        let mut off = 0;
        while off + VAD_FRAME_SAMPLES <= self.leftover.len() {
            let prob = vad.run_frame(
                &self.leftover[off..off + VAD_FRAME_SAMPLES],
                &mut self.state,
            )?;
            off += VAD_FRAME_SAMPLES;
            if self.hangover.update(prob, VAD_FRAME_SAMPLES) {
                endpoint = true;
            }
        }
        // Retain only the unprocessed tail.
        if off > 0 {
            self.leftover.drain(..off);
        }
        Ok(endpoint)
    }
}

/// Pure trailing-silence state machine shared by the streaming endpointer.
///
/// `update` is fed one frame's probability at a time and returns `true` on the
/// single frame where trailing silence first reaches `min_silence_ms` after
/// speech has been observed. After firing it disarms until speech resumes, so
/// one utterance yields exactly one endpoint.
#[derive(Debug)]
pub struct Hangover {
    threshold: f32,
    min_silence_samples: usize,
    seen_speech: bool,
    trailing_silence: usize,
    armed: bool,
}

impl Hangover {
    fn new(cfg: &VadConfig) -> Self {
        Self {
            threshold: cfg.threshold,
            min_silence_samples: VadConfig::ms_to_samples(cfg.min_silence_ms),
            seen_speech: false,
            trailing_silence: 0,
            armed: false,
        }
    }

    /// Advance by one frame of `frame_samples` samples with speech probability
    /// `prob`. Returns `true` on the endpoint-crossing frame. Thresholds are
    /// fixed at construction ([`Hangover::new`]).
    fn update(&mut self, prob: f32, frame_samples: usize) -> bool {
        if prob >= self.threshold {
            self.seen_speech = true;
            self.armed = true;
            self.trailing_silence = 0;
            return false;
        }
        if !self.seen_speech {
            return false;
        }
        self.trailing_silence += frame_samples;
        if self.armed && self.trailing_silence >= self.min_silence_samples {
            self.armed = false; // fire once until speech resumes
            return true;
        }
        false
    }
}

#[cfg(test)]
mod tests;
