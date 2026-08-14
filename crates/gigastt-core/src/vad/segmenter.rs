//! Causal, bounded-memory file-VAD segmenter.

use anyhow::Result;

use super::{SileroVad, VAD_FRAME_SAMPLES, VAD_STATE_LEN, VadConfig};

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
    pub(crate) fn push_with<F>(
        &mut self,
        samples: &[f32],
        out: &mut Vec<f32>,
        mut score: F,
    ) -> Result<()>
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
    pub(crate) fn finish_with<F>(
        &mut self,
        total: usize,
        out: &mut Vec<f32>,
        mut score: F,
    ) -> Result<()>
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
    pub(crate) fn retained(&self) -> usize {
        self.raw.len()
    }
}
