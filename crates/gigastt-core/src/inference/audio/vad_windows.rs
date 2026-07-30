//! Windowed PCM source for the VAD file path.
//!
//! The batch VAD path decodes the whole file, scans every frame for speech, and
//! copies the kept spans into a second whole-file buffer — two O(file)
//! allocations, which is why that path alone still enforced a duration ceiling
//! while the plain file path had none. [`VadWindows`] replaces both with a
//! stream: [`FileWindows`] feeds the container in fixed blocks, [`VadSegmenter`]
//! scores them causally and releases kept audio as soon as its bounded
//! look-ahead allows, and this source hands the result to the decode loop as the
//! same overlapping windows [`SliceWindows`](super::SliceWindows) would yield
//! over the fully compressed buffer.
//!
//! Peak audio memory is one decode window plus well under a second of retained
//! PCM, regardless of file length. The windows themselves are unchanged, so the
//! transcript is too.

use crate::error::GigasttError;
use crate::vad::{SileroVad, VadConfig, VadSegmenter};

use super::stream::{FileWindows, PcmWindow, PcmWindows, WindowCursor, WindowSpec};

/// Block size (@16 kHz) pulled from the container per VAD step. A multiple of
/// the encoder frame stride, large enough to amortize the per-block work and
/// small enough that the retained PCM stays negligible.
const PULL_SAMPLES: usize = 32_000; // 2 s

/// Poll the abort flag every this many pulled blocks (~30 s of audio), matching
/// the batch scan's cadence: interruptible on a multi-hour file without an
/// atomic load per block.
const ABORT_POLL_BLOCKS: usize = 16;

/// [`PcmWindows`] over the silence-free timeline of a streamed container.
pub(crate) struct VadWindows<'a> {
    raw: FileWindows,
    vad: &'a SileroVad,
    seg: VadSegmenter,
    abort: Option<&'a dyn Fn() -> bool>,
    /// Rolling compressed buffer holding `[buf_start_abs, compressed_total)`.
    buf: Vec<f32>,
    /// Absolute sample index (on the compressed timeline) of `buf[0]`.
    buf_start_abs: usize,
    /// Total compressed samples released so far.
    compressed_total: usize,
    /// True once the container is drained and the segmenter has been closed.
    eof: bool,
    /// Set when the VAD model itself failed mid-stream; the caller re-decodes
    /// without VAD rather than returning a truncated transcript.
    vad_failed: bool,
    blocks: usize,
    cursor: WindowCursor,
}

impl<'a> VadWindows<'a> {
    /// Wrap an open container in the VAD stream. `spec` is the decode geometry
    /// applied to the *compressed* timeline — the same one the no-VAD path uses.
    pub(crate) fn new(
        raw: FileWindows,
        vad: &'a SileroVad,
        cfg: &VadConfig,
        spec: WindowSpec,
        abort: Option<&'a dyn Fn() -> bool>,
    ) -> Self {
        Self {
            raw,
            vad,
            seg: VadSegmenter::new(cfg),
            abort,
            buf: Vec::new(),
            buf_start_abs: 0,
            compressed_total: 0,
            eof: false,
            vad_failed: false,
            blocks: 0,
            cursor: WindowCursor::new(spec),
        }
    }

    /// Window geometry for the flat container reads this source performs:
    /// consecutive, non-overlapping blocks covering the stream exactly once.
    pub(crate) fn pull_spec() -> WindowSpec {
        WindowSpec::new(0, PULL_SAMPLES, 0)
    }

    /// Kept speech spans on the **original** timeline, in order. Complete once
    /// the source is drained; used to map decoded word timestamps back off the
    /// compressed timeline.
    pub(crate) fn regions(&self) -> &[(usize, usize)] {
        self.seg.regions()
    }

    /// Total 16 kHz samples read from the container. Exact once drained — this
    /// is the clip's real duration, not the compressed one.
    pub(crate) fn total_16k_samples(&self) -> usize {
        self.raw.total_16k_samples()
    }

    /// True when the caller must re-decode without VAD: either the model failed
    /// mid-stream, or the scan found no speech at all in a non-empty clip (a bad
    /// threshold against continuous speech or a pure tone). Both cases fall back
    /// to the full/chunked decode rather than returning an empty transcript.
    pub(crate) fn needs_fallback(&self) -> bool {
        self.vad_failed || (self.regions().is_empty() && self.total_16k_samples() > 0)
    }

    /// Decode and score until `target` compressed samples are available (or the
    /// stream ends).
    fn fill_to(&mut self, target: usize) -> Result<(), GigasttError> {
        let Self {
            raw,
            vad,
            seg,
            abort,
            buf,
            compressed_total,
            eof,
            vad_failed,
            blocks,
            ..
        } = self;
        while !*eof && *compressed_total < target {
            if let Some(abort) = abort {
                *blocks += 1;
                if *blocks >= ABORT_POLL_BLOCKS {
                    *blocks = 0;
                    if abort() {
                        return Err(GigasttError::Cancelled);
                    }
                }
            }
            let before = buf.len();
            let scanned = match raw.next_window()? {
                Some(w) => seg.push(vad, w.samples, buf),
                None => {
                    *eof = true;
                    seg.finish(vad, raw.total_16k_samples(), buf)
                }
            };
            if let Err(e) = scanned {
                // Same policy as the batch path: a VAD failure never fails the
                // request, it drops VAD for this clip.
                tracing::warn!("VAD failed mid-stream, decoding full audio: {e:#}");
                *vad_failed = true;
                *eof = true;
                buf.truncate(before);
                return Ok(());
            }
            *compressed_total += buf.len() - before;
        }
        Ok(())
    }
}

impl PcmWindows for VadWindows<'_> {
    fn spec(&self) -> WindowSpec {
        self.cursor.spec()
    }

    fn next_window(&mut self) -> Result<Option<PcmWindow<'_>>, GigasttError> {
        if self.cursor.is_done() {
            return Ok(None);
        }

        // Reclaim the previous window's consumed prefix; windows only move
        // forward, so everything before `next_start` is dead.
        let drop = self
            .cursor
            .next_start()
            .saturating_sub(self.buf_start_abs)
            .min(self.buf.len());
        if drop > 0 {
            self.buf.drain(0..drop);
            self.buf_start_abs += drop;
        }

        self.fill_to(self.cursor.fill_target())?;

        // Nothing survived the VAD: yield no window rather than an empty one.
        // The batch path returns early on empty regions for the same reason —
        // a zero-length encoder run is not a decode.
        if self.eof && self.compressed_total == 0 {
            return Ok(None);
        }

        let Some((start, end)) = self.cursor.take(self.compressed_total, self.eof) else {
            return Ok(None);
        };
        let s = start - self.buf_start_abs;
        let e = end - self.buf_start_abs;
        Ok(Some(PcmWindow {
            start_sample: start,
            samples: &self.buf[s..e],
        }))
    }
}

/// Byte-identity tests against the batch VAD path.
///
/// The Silero session is scripted through the mock runtime, so a whole
/// probability sequence — and therefore a whole speech/silence layout — can be
/// pinned without the model on disk. That makes the comparison that matters run
/// in CI: the same clip, the same config, batch versus streaming, window for
/// window.
#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use super::*;
    use crate::inference::audio::{SliceWindows, encode_wav_pcm16};
    use crate::runtime::mock::{MockFactory, MockSession};
    use crate::runtime::tensor::{Shape, Tensor, TensorData};
    use crate::vad::{VAD_FRAME_SAMPLES, regions_from_probs};
    use bytes::Bytes;

    /// One decode window, owned: `(start on the compressed timeline, samples)`.
    type OwnedWindow = (usize, Vec<f32>);

    /// The ort long-form geometry, matching `Engine::window_spec` (CPU backend).
    fn ort_spec() -> WindowSpec {
        WindowSpec::new(16000 * 30, 16000 * 24, 16000 * 2)
    }

    /// A [`SileroVad`] whose session replays `probs`, one per frame, so the VAD
    /// file path runs without the model. Shapes match the real graph, so the
    /// output-by-shape identification in `run_frame` is exercised too.
    fn scripted_vad(probs: &[f32]) -> SileroVad {
        let script: Vec<Vec<Tensor>> = probs
            .iter()
            .map(|&p| {
                vec![
                    Tensor::new_checked(Shape::new(vec![1]), TensorData::F32(vec![p])),
                    Tensor::new_checked(
                        Shape::new(vec![2, 1, 128]),
                        TensorData::F32(vec![0.0; 2 * 128]),
                    ),
                ]
            })
            .collect();
        let session = MockSession::new(
            vec![
                Shape::new(vec![1, VAD_FRAME_SAMPLES]),
                Shape::new(vec![2, 1, 128]),
                Shape::new(vec![1]),
            ],
            Vec::new(),
        )
        .with_script(script);
        let factory = MockFactory::new(HashMap::from([(
            "silero_vad".to_string(),
            Arc::new(session),
        )]));
        SileroVad::load_with_factory(std::path::Path::new("silero_vad.onnx"), &factory)
            .expect("scripted vad")
    }

    /// Deterministic signal in [-1, 1) that survives PCM16 quantization.
    fn signal(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| 0.4 * ((i as f32 * 0.017).sin() + 0.5 * (i as f32 * 0.0031).sin()))
            .collect()
    }

    /// `(start, samples)` pairs from any window source.
    fn drain(src: &mut dyn PcmWindows) -> Vec<OwnedWindow> {
        let mut out = Vec::new();
        while let Some(w) = src.next_window().expect("window") {
            out.push((w.start_sample, w.samples.to_vec()));
        }
        out
    }

    /// What the batch path feeds the decoder: `speech_regions` over the whole
    /// buffer, the kept spans concatenated, then the standard geometry (one
    /// whole-buffer window under the single-pass ceiling).
    fn batch(
        samples: &[f32],
        probs: &[f32],
        cfg: &VadConfig,
    ) -> (Vec<(usize, usize)>, Vec<OwnedWindow>) {
        let regions = regions_from_probs(probs, VAD_FRAME_SAMPLES, samples.len(), cfg);
        let compressed: Vec<f32> = regions
            .iter()
            .flat_map(|&(s, e)| samples[s..e].iter().copied())
            .collect();
        let spec = ort_spec();
        let windows = if compressed.is_empty() {
            // The batch path returns before touching the decoder when no region
            // survives, and falls back to a full decode instead.
            Vec::new()
        } else if compressed.len() <= spec.single_pass_max() {
            vec![(0, compressed.clone())]
        } else {
            drain(&mut SliceWindows::new(&compressed, spec))
        };
        (regions, windows)
    }

    /// Probabilities covering `n` samples: `on_frames` speech then `off_frames`
    /// silence, repeating.
    fn alternating(n: usize, on_frames: usize, off_frames: usize) -> Vec<f32> {
        let period = on_frames + off_frames;
        (0..n.div_ceil(VAD_FRAME_SAMPLES))
            .map(|i| if i % period < on_frames { 0.9 } else { 0.1 })
            .collect()
    }

    fn assert_matches_batch(n: usize, probs: &[f32], cfg: &VadConfig) {
        let samples = signal(n);
        let wav = Bytes::from(encode_wav_pcm16(&samples, 16000));
        // 16 kHz PCM16 is the passthrough path, so the decoded buffer is the
        // quantized `samples` — take it from the decoder to compare like for like.
        let decoded = FileWindows::from_bytes(wav.clone(), WindowSpec::flat(), None)
            .expect("open flat")
            .drain_to_vec()
            .expect("drain");
        let (want_regions, want_windows) = batch(&decoded, probs, cfg);

        let vad = scripted_vad(probs);
        let mut src = VadWindows::new(
            FileWindows::from_bytes(wav, VadWindows::pull_spec(), None).expect("open pull"),
            &vad,
            cfg,
            ort_spec(),
            None,
        );
        let got_windows = drain(&mut src);
        assert_eq!(src.regions(), want_regions, "regions diverged at n={n}");
        assert_eq!(got_windows, want_windows, "windows diverged at n={n}");
        assert_eq!(src.total_16k_samples(), decoded.len());
        // Both paths hand the clip back to the full decode on an empty result.
        assert_eq!(src.needs_fallback(), want_regions.is_empty());
    }

    #[test]
    fn test_vad_windows_match_batch_below_single_pass() {
        // 20 s: the compressed buffer stays under the single-pass ceiling, so
        // both paths must yield exactly one whole-buffer window.
        let n = 16000 * 20;
        assert_matches_batch(n, &alternating(n, 20, 40), &VadConfig::default());
    }

    #[test]
    fn test_vad_windows_match_batch_in_the_chunked_regime() {
        // 90 s with ~2/3 speech: the compressed buffer is well past the ceiling,
        // so the overlapping geometry is exercised over compressed time.
        let n = 16000 * 90;
        assert_matches_batch(n, &alternating(n, 60, 30), &VadConfig::default());
    }

    #[test]
    fn test_vad_windows_match_batch_on_sparse_speech() {
        // Long silences between short bursts: many regions, heavy compression.
        let n = 16000 * 75;
        assert_matches_batch(n, &alternating(n, 12, 100), &VadConfig::default());
    }

    #[test]
    fn test_vad_windows_match_batch_across_configs() {
        let n = 16000 * 45;
        let probs = alternating(n, 25, 25);
        for cfg in [
            VadConfig {
                threshold: 0.5,
                min_silence_ms: 0,
                min_speech_ms: 0,
                speech_pad_ms: 0,
            },
            VadConfig {
                threshold: 0.5,
                min_silence_ms: 100,
                min_speech_ms: 1000,
                speech_pad_ms: 300,
            },
            VadConfig {
                threshold: 0.5,
                min_silence_ms: 2000,
                min_speech_ms: 100,
                speech_pad_ms: 50,
            },
        ] {
            assert_matches_batch(n, &probs, &cfg);
        }
    }

    #[test]
    fn test_vad_windows_all_speech_is_the_plain_stream() {
        // Every frame speech: the compressed timeline is the raw one, so the
        // windows must equal what `FileWindows` alone would yield.
        let n = 16000 * 70;
        let samples = signal(n);
        let wav = Bytes::from(encode_wav_pcm16(&samples, 16000));
        let probs = vec![0.9f32; n.div_ceil(VAD_FRAME_SAMPLES)];
        let vad = scripted_vad(&probs);
        let cfg = VadConfig::default();
        let mut src = VadWindows::new(
            FileWindows::from_bytes(wav.clone(), VadWindows::pull_spec(), None).expect("pull"),
            &vad,
            &cfg,
            ort_spec(),
            None,
        );
        let got = drain(&mut src);
        let want = drain(&mut FileWindows::from_bytes(wav, ort_spec(), None).expect("plain"));
        assert_eq!(src.regions(), &[(0, n)]);
        assert_eq!(got, want);
        assert!(!src.needs_fallback());
    }

    #[test]
    fn test_vad_windows_no_speech_asks_for_fallback() {
        let n = 16000 * 40;
        let samples = signal(n);
        let wav = Bytes::from(encode_wav_pcm16(&samples, 16000));
        let probs = vec![0.1f32; n.div_ceil(VAD_FRAME_SAMPLES)];
        let vad = scripted_vad(&probs);
        let cfg = VadConfig::default();
        let mut src = VadWindows::new(
            FileWindows::from_bytes(wav, VadWindows::pull_spec(), None).expect("pull"),
            &vad,
            &cfg,
            ort_spec(),
            None,
        );
        assert!(drain(&mut src).is_empty());
        assert!(src.regions().is_empty());
        assert!(
            src.needs_fallback(),
            "a clip with no detected speech must fall back, not transcribe to nothing"
        );
    }

    #[test]
    fn test_vad_windows_empty_clip_needs_no_fallback() {
        let wav = Bytes::from(encode_wav_pcm16(&[], 16000));
        let vad = scripted_vad(&[0.1]);
        let cfg = VadConfig::default();
        let mut src = VadWindows::new(
            FileWindows::from_bytes(wav, VadWindows::pull_spec(), None).expect("pull"),
            &vad,
            &cfg,
            ort_spec(),
            None,
        );
        drain(&mut src);
        assert_eq!(src.total_16k_samples(), 0);
        assert!(!src.needs_fallback());
    }

    #[test]
    fn test_vad_windows_cancellation_stops_the_scan() {
        let n = 16000 * 600; // 10 min: many pull blocks, so the poll is reached
        let samples = signal(n);
        let wav = Bytes::from(encode_wav_pcm16(&samples, 16000));
        let probs = vec![0.9f32; n.div_ceil(VAD_FRAME_SAMPLES)];
        let vad = scripted_vad(&probs);
        let cfg = VadConfig::default();
        let flag = || true;
        let mut src = VadWindows::new(
            FileWindows::from_bytes(wav, VadWindows::pull_spec(), None).expect("pull"),
            &vad,
            &cfg,
            ort_spec(),
            Some(&flag),
        );
        assert!(matches!(
            src.next_window(),
            Err(crate::error::GigasttError::Cancelled)
        ));
    }
}
