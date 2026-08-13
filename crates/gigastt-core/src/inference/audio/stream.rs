//! Windowed PCM source for long-form file decode.
//!
//! Long-form decode used to own its `while start < total { … }` loop directly
//! over one `&[f32]` holding the whole file. This module puts that window
//! geometry behind a small source trait so the decode loop no longer assumes
//! the PCM is fully materialized: [`SliceWindows`] is the buffer-backed source
//! used today, and a decoder-backed source can be added later without touching
//! the loop.
//!
//! [`PcmWindows::next_window`] **lends** its window — the returned
//! [`PcmWindow`] borrows the source — so a source can hand out a view into a
//! decoder's own scratch buffer instead of copying. That borrow shape is why
//! this is not an [`Iterator`].

use crate::error::GigasttError;
use crate::inference::{ENCODER_SUBSAMPLING, HOP_LENGTH};

#[cfg(feature = "file-decode")]
use anyhow::{Context, Result};
#[cfg(feature = "file-decode")]
use bytes::Bytes;
#[cfg(feature = "file-decode")]
use symphonia::core::codecs::audio::well_known::CODEC_ID_OPUS;
#[cfg(feature = "file-decode")]
use symphonia::core::codecs::audio::{AudioDecoder, AudioDecoderOptions};
#[cfg(feature = "file-decode")]
use symphonia::core::formats::probe::Hint;
#[cfg(feature = "file-decode")]
use symphonia::core::formats::{FormatOptions, FormatReader, TrackType};
#[cfg(feature = "file-decode")]
use symphonia::core::io::MediaSourceStream;
#[cfg(feature = "file-decode")]
use symphonia::core::meta::MetadataOptions;

#[cfg(feature = "file-decode")]
use super::decode::BytesMediaSource;
#[cfg(feature = "file-decode")]
use super::opus::{OPUS_DECODE_RATE, OpusStream, next_demux_packet};
#[cfg(feature = "file-decode")]
use super::resample::{RESAMPLE_STAGING_FRAMES, ResampleTo16k, SampleRate};
#[cfg(feature = "file-decode")]
use super::telephony::{sniffs_as_g722_wav, try_decode_g722_wav};
#[cfg(feature = "file-decode")]
use super::{MAX_SAMPLE_RATE, audio_too_long_err, decode_error, resolve_budget};

/// Samples per encoder output frame (`HOP_LENGTH * ENCODER_SUBSAMPLING`,
/// 640 @16 kHz). Window starts are multiples of this so each window's frame
/// offset is integral.
const FRAME_SAMPLES: usize = HOP_LENGTH * ENCODER_SUBSAMPLING;

/// Long-form window geometry, all in samples @16 kHz.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WindowSpec {
    single_pass_max: usize,
    window: usize,
    stride: usize,
}

impl WindowSpec {
    /// Build a spec from the single-pass ceiling, the window length, and the
    /// overlap retained between consecutive windows.
    ///
    /// The stride (`window - overlap`) is aligned **down** to an encoder-frame
    /// boundary so every window start maps to an integral frame offset;
    /// otherwise the offset would drift by a sub-frame each hop. It is clamped
    /// to one frame so a mis-specified `overlap >= window` cannot produce a
    /// zero-stride (non-advancing) source.
    pub(crate) fn new(single_pass_max: usize, window: usize, overlap: usize) -> Self {
        let stride =
            (window.saturating_sub(overlap) / FRAME_SAMPLES * FRAME_SAMPLES).max(FRAME_SAMPLES);
        Self {
            single_pass_max,
            window,
            stride,
        }
    }

    /// Window length in samples.
    pub(crate) fn window(&self) -> usize {
        self.window
    }

    /// Frame-aligned distance between consecutive window starts.
    pub(crate) fn stride(&self) -> usize {
        self.stride
    }

    /// Largest total (samples @16 kHz) that stays on the single-pass branch — one
    /// encoder Run over the whole buffer rather than overlapping windows.
    pub(crate) fn single_pass_max(&self) -> usize {
        self.single_pass_max
    }

    /// Samples shared by two consecutive windows (`window - stride`). Equals the
    /// requested overlap whenever that overlap is already frame-aligned.
    pub(crate) fn overlap(&self) -> usize {
        self.window.saturating_sub(self.stride)
    }

    /// True when `total` samples are short enough for the single-pass (one
    /// encoder Run over the whole buffer) branch.
    pub(crate) fn is_single_pass(&self, total: usize) -> bool {
        total <= self.single_pass_max
    }

    /// Sentinel geometry for flat (`drain_to_vec`) decode, which never windows.
    /// Its window/stride are never read; only [`FileWindows::drain_to_vec`] uses
    /// a `FileWindows` built with it.
    #[cfg(feature = "file-decode")]
    pub(crate) fn flat() -> Self {
        Self::new(usize::MAX, usize::MAX, 0)
    }
}

/// Window-cursor arithmetic for a source that decodes as it goes.
///
/// Pure and shared, so every streaming source yields the one window sequence
/// [`SliceWindows`] would over the same fully-decoded buffer:
/// [`WindowCursor::fill_target`] says how far the source must decode before the
/// next window can be decided, and [`WindowCursor::take`] turns "here is what
/// is available" into that window.
#[cfg(feature = "file-decode")]
pub(crate) struct WindowCursor {
    spec: WindowSpec,
    /// Absolute start (@16 kHz) of the next window to yield.
    next_start: usize,
    /// True until the first window's single-pass-vs-windowed decision is made.
    first: bool,
    done: bool,
}

#[cfg(feature = "file-decode")]
impl WindowCursor {
    pub(crate) fn new(spec: WindowSpec) -> Self {
        Self {
            spec,
            next_start: 0,
            first: true,
            done: false,
        }
    }

    pub(crate) fn is_done(&self) -> bool {
        self.done
    }

    pub(crate) fn next_start(&self) -> usize {
        self.next_start
    }

    pub(crate) fn spec(&self) -> WindowSpec {
        self.spec
    }

    /// Decode at least this many samples before calling [`WindowCursor::take`]:
    /// one past the window end, so `end == total` is distinguishable from a
    /// mid-stream boundary, and enough for the first window to decide
    /// single-pass vs windowed.
    pub(crate) fn fill_target(&self) -> usize {
        if self.first {
            self.spec
                .single_pass_max()
                .saturating_add(1)
                .max(self.spec.window().saturating_add(1))
        } else {
            self.next_start + self.spec.window() + 1
        }
    }

    /// The next `[start, end)` window given the samples decoded so far
    /// (`avail_end`) and whether the stream is exhausted, or `None` once every
    /// window has been yielded.
    pub(crate) fn take(&mut self, avail_end: usize, eof: bool) -> Option<(usize, usize)> {
        if self.done {
            return None;
        }
        let start = self.next_start;
        if self.first {
            self.first = false;
            if eof && avail_end <= self.spec.single_pass_max() {
                // The whole stream fits the single-pass ceiling: one window over
                // all of it. `decode_words_streaming` then runs one encoder pass
                // with frame offset 0 and stitches onto an empty list —
                // byte-identical to `decode_words`' non-windowed branch.
                self.done = true;
                return Some((start, avail_end));
            }
        }
        if start >= avail_end {
            // Reachable only at EOF, once the last window has been yielded.
            self.done = true;
            return None;
        }
        // Because the caller decoded one sample past `start + window` (or hit
        // EOF), `end == avail_end` holds only when this window reaches the end.
        let end = (start + self.spec.window()).min(avail_end);
        if eof && end == avail_end {
            self.done = true;
        } else {
            self.next_start = start + self.spec.stride();
        }
        Some((start, end))
    }
}

/// Which channel a windowed decode yields.
///
/// The default file pipeline wants the mono mix; `channels=split` wants each
/// channel on its own, and needs it *streamed* — materializing every channel of
/// the whole file is what pinned that path to a duration ceiling.
#[cfg(feature = "file-decode")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChannelSelect {
    /// Mean of every channel present in the packet.
    Mono,
    /// A single channel, by zero-based index.
    One(usize),
}

/// One decode window lent by a [`PcmWindows`] source.
pub(crate) struct PcmWindow<'a> {
    /// Absolute offset of `samples[0]` in the stream, in samples @16 kHz.
    pub(crate) start_sample: usize,
    /// The window's PCM.
    pub(crate) samples: &'a [f32],
}

/// A source of overlapping decode windows.
pub(crate) trait PcmWindows {
    /// The window geometry this source yields.
    fn spec(&self) -> WindowSpec;

    /// Lend the next window, or `Ok(None)` once the stream is exhausted.
    fn next_window(&mut self) -> Result<Option<PcmWindow<'_>>, GigasttError>;
}

/// [`PcmWindows`] over a fully materialized buffer.
///
/// Yields exactly the `(start, end)` sequence of the loop it replaced:
/// `while start < total { end = min(start + window, total); …; if end == total
/// { break } start += stride }`.
pub(crate) struct SliceWindows<'a> {
    samples: &'a [f32],
    spec: WindowSpec,
    next_start: usize,
    done: bool,
}

impl<'a> SliceWindows<'a> {
    pub(crate) fn new(samples: &'a [f32], spec: WindowSpec) -> Self {
        Self {
            samples,
            spec,
            next_start: 0,
            done: false,
        }
    }
}

impl PcmWindows for SliceWindows<'_> {
    fn spec(&self) -> WindowSpec {
        self.spec
    }

    fn next_window(&mut self) -> Result<Option<PcmWindow<'_>>, GigasttError> {
        let total = self.samples.len();
        if self.done || self.next_start >= total {
            return Ok(None);
        }
        let start = self.next_start;
        let end = (start + self.spec.window()).min(total);
        // The replaced loop stopped on `end == total` rather than on the next
        // start passing `total`, so a window that reaches the end is the last
        // one even when `start + stride` is still short of `total`.
        if end == total {
            self.done = true;
        } else {
            self.next_start = start + self.spec.stride();
        }
        Ok(Some(PcmWindow {
            start_sample: start,
            samples: &self.samples[start..end],
        }))
    }
}

/// The decode engine behind [`FileWindows`]: a streaming symphonia loop, or an
/// already-materialized buffer for the formats that cannot stream.
#[cfg(feature = "file-decode")]
enum Source {
    /// Container decoded packet-by-packet, resampled to 16 kHz as it goes.
    Streaming {
        format: Box<dyn FormatReader>,
        decoder: Box<dyn AudioDecoder>,
        track_id: u32,
        /// Source (container) sample rate — the units the length budget counts.
        sample_rate: u32,
        /// Running SOURCE-rate frame count, tracked separately from the 16 kHz
        /// accumulator because the length budget is expressed in source frames.
        source_frames: usize,
        /// Source-rate frame budget. `usize::MAX` when the caller imposed no
        /// limit — the streaming path is O(one window), so length is unbounded
        /// by default; the flat drain and the whole-buffer callers pass a finite
        /// budget (see [`max_samples_for_secs`]).
        max_samples: usize,
        /// The seconds limit `max_samples` was derived from, echoed verbatim in
        /// [`AudioTooLong`](crate::error::GigasttError::AudioTooLong) on a trip.
        limit_secs: f64,
        /// Boxed: it carries the heavyweight rubato FIR state, so keeping it
        /// behind a pointer keeps the `Streaming` variant small.
        resampler: Box<ResampleTo16k>,
        /// Per-packet interleaved scratch, hoisted out of the decode loop.
        interleaved: Vec<f32>,
    },
    /// OGG/Opus: symphonia demuxes it but ships no decoder, so packets go
    /// through the `opus-rs` fallback ([`OpusStream`]) and are mixed to mono
    /// and resampled as they arrive — same shape as `Streaming`, different
    /// decoder.
    Opus {
        format: Box<dyn FormatReader>,
        track_id: u32,
        /// Boxed: it carries libopus decoder state.
        stream: Box<OpusStream>,
        /// Running decoded frame count at the Opus decode rate (48 kHz), which
        /// is what the budget below counts and what a trip reports.
        decoded_48k: usize,
        max_samples: usize,
        limit_secs: f64,
        resampler: Box<ResampleTo16k>,
        /// Decoded mono samples not yet staged. The whole-buffer path fed the
        /// resampler in exact [`RESAMPLE_STAGING_FRAMES`] chunks; holding the
        /// remainder here reproduces that chunk sequence packet-by-packet, so
        /// the flush boundaries — and with them the resampled output — do not
        /// move. Bounded by one chunk plus one packet.
        pending: Vec<f32>,
    },
    /// The whole 16 kHz stream is already in [`FileWindows::buf`]. Used by the
    /// G.722-in-WAV telephony path (no symphonia decoder).
    Eager,
}

/// A [`PcmWindows`] source that pulls overlapping decode windows straight from an
/// audio container, so peak audio memory is O(one window) rather than O(file).
///
/// It holds only a rolling 16 kHz buffer — one window plus a packet of
/// look-ahead — and drops each window's consumed prefix before decoding the
/// next, so a three-hour file costs the same resident audio memory as a
/// thirty-second one. The window sequence is byte-identical to
/// [`SliceWindows`] over the same fully-decoded buffer:
///
/// - a stream that fits the single-pass ceiling yields exactly one window over
///   the whole buffer, matching `Engine::decode_words`' non-windowed branch;
/// - a longer stream yields the standard overlapping geometry.
///
/// [`FileWindows::drain_to_vec`] runs the same decode flat (no windowing) and is
/// byte-identical to the whole-buffer decoder the public `decode_audio_*`
/// wrappers used to call.
#[cfg(feature = "file-decode")]
pub(crate) struct FileWindows {
    src: Source,
    /// True once the container is exhausted (or eagerly materialized).
    eof: bool,
    /// True once the resampler's staged remainder has been flushed at EOF.
    finished: bool,
    /// Rolling 16 kHz buffer holding `[buf_start_abs, decoded_16k_total)`.
    buf: Vec<f32>,
    /// Absolute sample index (@16 kHz) of `buf[0]`.
    buf_start_abs: usize,
    /// Total 16 kHz samples decoded so far (== `buf_start_abs + buf.len()`).
    decoded_16k_total: usize,
    spec: WindowSpec,
    /// Which channel the packet loop keeps.
    channel: ChannelSelect,
    cursor: WindowCursor,
}

#[cfg(feature = "file-decode")]
impl FileWindows {
    /// Open a file for windowed decode. Mirrors `decode_audio_file`'s probe/hint
    /// setup, including the G.722-in-WAV telephony sniff.
    pub(crate) fn open(path: &str, spec: WindowSpec, max_audio_secs: Option<f64>) -> Result<Self> {
        if sniffs_as_g722_wav(path)? {
            let bytes = std::fs::read(path)
                .with_context(|| format!("Failed to read audio file: {path}"))?;
            if let Some(result) = try_decode_g722_wav(&bytes, max_audio_secs) {
                return Ok(Self::eager(result?, spec));
            }
        }
        let file = std::fs::File::open(path)
            .with_context(|| format!("Failed to open audio file: {path}"))?;
        let mss = MediaSourceStream::new(Box::new(file), Default::default());
        let mut hint = Hint::new();
        if let Some(ext) = std::path::Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
        {
            hint.with_extension(ext);
        }
        Self::from_mss(mss, hint, spec, max_audio_secs, ChannelSelect::Mono)
    }

    /// Open a shared [`Bytes`] buffer for windowed decode. `BytesMediaSource` is
    /// seekable, so the isomp4 demuxer's trailing-`moov` seek works; no spool.
    pub(crate) fn from_bytes(
        data: Bytes,
        spec: WindowSpec,
        max_audio_secs: Option<f64>,
    ) -> Result<Self> {
        if let Some(result) = try_decode_g722_wav(&data, max_audio_secs) {
            return Ok(Self::eager(result?, spec));
        }
        let source = BytesMediaSource::new(data);
        let mss = MediaSourceStream::new(Box::new(source), Default::default());
        Self::from_mss(mss, Hint::new(), spec, max_audio_secs, ChannelSelect::Mono)
    }

    /// Probe the container and either set up the streaming decoder or, for Opus /
    /// G.722, eagerly materialize the whole 16 kHz buffer. Scalar params are
    /// copied out (and the non-Opus decoder built) inside one borrow scope so the
    /// `FormatReader` is free to move or be driven afterwards — the same shape as
    /// the whole-buffer `decode_audio_inner`.
    fn from_mss(
        mss: MediaSourceStream<'static>,
        hint: Hint,
        spec: WindowSpec,
        max_audio_secs: Option<f64>,
        channel: ChannelSelect,
    ) -> Result<Self> {
        let format = symphonia::default::get_probe()
            .probe(
                &hint,
                mss,
                FormatOptions::default(),
                MetadataOptions::default(),
            )
            .context("Unsupported audio format")?;

        let (track_id, sample_rate, channels, decoder_opt) = {
            let track = format
                .default_track(TrackType::Audio)
                .context("No audio track found")?;
            let track_id = track.id;
            let audio_params = track
                .codec_params
                .as_ref()
                .and_then(|p| p.audio())
                .context("No audio codec parameters")?;
            let sample_rate = audio_params.sample_rate.context("Unknown sample rate")?;
            if sample_rate == 0 || sample_rate > MAX_SAMPLE_RATE {
                anyhow::bail!("Unsupported sample rate: {sample_rate}Hz");
            }
            let channels = audio_params
                .channels
                .as_ref()
                .map(|c| c.count())
                .unwrap_or(1);
            // Opus is demuxed by symphonia but has no symphonia decoder; the
            // `opus-rs` fallback needs the `FormatReader`, so build no decoder
            // here and take the eager branch below.
            let decoder_opt = if audio_params.codec == CODEC_ID_OPUS {
                None
            } else {
                Some(
                    symphonia::default::get_codecs()
                        .make_audio_decoder(audio_params, &AudioDecoderOptions::default())
                        .context("Unsupported audio codec")?,
                )
            };
            (track_id, sample_rate, channels, decoder_opt)
        };

        let (max_samples, limit_secs) = resolve_budget(max_audio_secs, sample_rate);

        match decoder_opt {
            None => {
                // The budget counts at the Opus decode rate (48 kHz), which is
                // what the decoder actually emits and what a trip reports —
                // the container's declared input rate is not necessarily the
                // same number. No whole-buffer clamp: this path streams now.
                let (max_samples, limit_secs) = resolve_budget(max_audio_secs, OPUS_DECODE_RATE);
                tracing::info!("Audio (opus): {sample_rate}Hz, {channels}ch (streaming windows)");
                Ok(Self {
                    src: Source::Opus {
                        format,
                        track_id,
                        stream: Box::new(OpusStream::new(channels, channel)?),
                        decoded_48k: 0,
                        max_samples,
                        limit_secs,
                        resampler: Box::new(ResampleTo16k::new(SampleRate(sample_rate), None)),
                        pending: Vec::with_capacity(RESAMPLE_STAGING_FRAMES),
                    },
                    eof: false,
                    finished: false,
                    buf: Vec::new(),
                    buf_start_abs: 0,
                    decoded_16k_total: 0,
                    spec,
                    channel,
                    cursor: WindowCursor::new(spec),
                })
            }
            Some(decoder) => {
                tracing::info!("Audio: {sample_rate}Hz, {channels}ch (streaming windows)");
                Ok(Self {
                    src: Source::Streaming {
                        format,
                        decoder,
                        track_id,
                        sample_rate,
                        source_frames: 0,
                        max_samples,
                        limit_secs,
                        // No length hint: the windowed path never materializes the
                        // whole 16 kHz stream, so it must not reserve for it.
                        resampler: Box::new(ResampleTo16k::new(SampleRate(sample_rate), None)),
                        interleaved: Vec::new(),
                    },
                    eof: false,
                    finished: false,
                    buf: Vec::new(),
                    buf_start_abs: 0,
                    decoded_16k_total: 0,
                    spec,
                    channel,
                    cursor: WindowCursor::new(spec),
                })
            }
        }
    }

    /// Open a shared [`Bytes`] buffer for windowed decode of **one** channel.
    ///
    /// The per-channel twin of [`FileWindows::from_bytes`], for the
    /// `channels=split` path: channel `k` is extracted in the packet loop and
    /// resampled on its own, so peak memory is one window rather than every
    /// channel of the whole file.
    ///
    /// The samples are the ones the whole-buffer per-channel decode produced —
    /// same packet cadence, same resampler, same staging — so the windows the
    /// decoder sees are unchanged.
    pub(crate) fn from_bytes_channel(
        data: Bytes,
        spec: WindowSpec,
        max_audio_secs: Option<f64>,
        channel: usize,
    ) -> Result<Self> {
        if let Some(result) = try_decode_g722_wav(&data, max_audio_secs) {
            // G.722-in-WAV is mono by construction; there is no channel to pick.
            return Ok(Self::eager(result?, spec));
        }
        let source = BytesMediaSource::new(data);
        let mss = MediaSourceStream::new(Box::new(source), Default::default());
        Self::from_mss(
            mss,
            Hint::new(),
            spec,
            max_audio_secs,
            ChannelSelect::One(channel),
        )
    }

    /// Build an eager source over an already-decoded 16 kHz buffer.
    fn eager(buf: Vec<f32>, spec: WindowSpec) -> Self {
        let total = buf.len();
        Self {
            src: Source::Eager,
            eof: true,
            finished: true,
            buf,
            buf_start_abs: 0,
            decoded_16k_total: total,
            spec,
            channel: ChannelSelect::Mono,
            cursor: WindowCursor::new(spec),
        }
    }

    /// Total 16 kHz samples decoded. Exact once the stream is drained (every
    /// window consumed), matching the whole-buffer decoder's sample count.
    pub(crate) fn total_16k_samples(&self) -> usize {
        self.decoded_16k_total
    }

    /// Decode the whole stream flat (no windowing) to one 16 kHz mono buffer.
    ///
    /// Byte-identical to the whole-buffer `decode_audio_inner`: the same packet
    /// loop, the same per-packet resampler flush cadence, the same final drain.
    pub(crate) fn drain_to_vec(mut self) -> Result<Vec<f32>> {
        self.fill_to(usize::MAX)?;
        Ok(std::mem::take(&mut self.buf))
    }

    /// Flat-decode a file to 16 kHz mono. The window geometry is irrelevant to
    /// `drain_to_vec`, so a sentinel spec is used. `max_audio_secs` is the
    /// whole-buffer length budget (this drain materializes the entire stream).
    pub(crate) fn decode_file(path: &str, max_audio_secs: Option<f64>) -> Result<Vec<f32>> {
        Self::open(path, WindowSpec::flat(), max_audio_secs)?.drain_to_vec()
    }

    /// Flat-decode a shared byte buffer to 16 kHz mono.
    pub(crate) fn decode_bytes(data: Bytes, max_audio_secs: Option<f64>) -> Result<Vec<f32>> {
        Self::from_bytes(data, WindowSpec::flat(), max_audio_secs)?.drain_to_vec()
    }

    /// Decode until `decoded_16k_total >= target` (or EOF), appending 16 kHz
    /// samples to `buf`. Enforces the source-rate length budget incrementally with
    /// the exact same error string as the whole-buffer path.
    fn fill_to(&mut self, target: usize) -> Result<()> {
        match self.src {
            Source::Streaming { .. } => self.fill_streaming(target),
            Source::Opus { .. } => self.fill_opus(target),
            // The whole buffer is already resident.
            Source::Eager => Ok(()),
        }
    }

    /// [`FileWindows::fill_to`] for a container symphonia can decode itself.
    fn fill_streaming(&mut self, target: usize) -> Result<()> {
        let channel = self.channel;
        let Source::Streaming {
            format,
            decoder,
            track_id,
            sample_rate,
            source_frames,
            max_samples,
            limit_secs,
            resampler,
            interleaved,
        } = &mut self.src
        else {
            return Ok(()); // Eager: the whole buffer is already resident.
        };

        while !self.eof && self.decoded_16k_total < target {
            let have_pcm = *source_frames > 0;
            let packet = match next_demux_packet(&mut **format, have_pcm)? {
                Some(p) => p,
                None => {
                    self.eof = true;
                    break;
                }
            };
            if packet.track_id != *track_id {
                continue;
            }

            let decoded = decoder.decode(&packet).context("Decode error")?;
            let num_frames = decoded.frames();
            let ch = decoded.spec().channels().count();

            if ch > 1 {
                interleaved.clear();
                decoded.copy_to_vec_interleaved(interleaved);
                let stage = resampler.stage();
                match channel {
                    ChannelSelect::Mono => {
                        for frame in 0..num_frames {
                            let mut sum = 0.0_f32;
                            for c in 0..ch {
                                sum += interleaved[frame * ch + c];
                            }
                            stage.push(sum / ch as f32);
                        }
                    }
                    // A packet short of the requested channel contributes
                    // nothing rather than shifting the timeline.
                    ChannelSelect::One(k) if k < ch => {
                        for frame in 0..num_frames {
                            stage.push(interleaved[frame * ch + k]);
                        }
                    }
                    ChannelSelect::One(_) => {}
                }
            } else if matches!(channel, ChannelSelect::Mono | ChannelSelect::One(0)) {
                // Single-channel packet: it *is* channel 0. Asking for a higher
                // index contributes nothing, rather than silently handing back
                // channel 0's audio under another channel's name.
                let stage = resampler.stage();
                let offset = stage.len();
                stage.resize(offset + num_frames, 0.0);
                decoded.copy_to_slice_interleaved(&mut stage[offset..]);
            }
            *source_frames += num_frames;

            if *source_frames > *max_samples {
                return Err(audio_too_long_err(
                    *source_frames,
                    *sample_rate,
                    *limit_secs,
                ));
            }

            resampler.flush_full()?;
            let before = self.buf.len();
            resampler.drain_ready_into(&mut self.buf);
            self.decoded_16k_total += self.buf.len() - before;
        }

        if self.eof && !self.finished {
            let before = self.buf.len();
            resampler.finish_into(&mut self.buf)?;
            self.decoded_16k_total += self.buf.len() - before;
            self.finished = true;
        }
        Ok(())
    }

    /// [`FileWindows::fill_to`] for OGG/Opus, decoded through the `opus-rs`
    /// fallback. Same loop as [`FileWindows::fill_streaming`]: one packet at a
    /// time, budget checked incrementally, resampler drained into `buf`.
    fn fill_opus(&mut self, target: usize) -> Result<()> {
        let Source::Opus {
            format,
            track_id,
            stream,
            decoded_48k,
            max_samples,
            limit_secs,
            resampler,
            pending,
        } = &mut self.src
        else {
            return Ok(());
        };

        while !self.eof && self.decoded_16k_total < target {
            let have_pcm = *decoded_48k > 0;
            let packet = match next_demux_packet(&mut **format, have_pcm)? {
                Some(p) => p,
                None => {
                    self.eof = true;
                    break;
                }
            };
            if packet.track_id != *track_id {
                continue;
            }

            *decoded_48k += stream.decode_packet(&packet.data, pending)?;
            if *decoded_48k > *max_samples {
                return Err(audio_too_long_err(
                    *decoded_48k,
                    OPUS_DECODE_RATE,
                    *limit_secs,
                ));
            }

            // Stage in exact `RESAMPLE_STAGING_FRAMES` chunks, keeping the
            // remainder in `pending`: that is the chunk sequence the
            // whole-buffer path produced, and the resampler drains whatever is
            // staged, so any other cadence would move the flush boundaries and
            // with them the output samples.
            while pending.len() >= RESAMPLE_STAGING_FRAMES {
                resampler
                    .stage()
                    .extend_from_slice(&pending[..RESAMPLE_STAGING_FRAMES]);
                pending.drain(..RESAMPLE_STAGING_FRAMES);
                resampler.flush_full()?;
            }
            let before = self.buf.len();
            resampler.drain_ready_into(&mut self.buf);
            self.decoded_16k_total += self.buf.len() - before;
        }

        if self.eof && !self.finished {
            resampler.stage().extend_from_slice(pending);
            pending.clear();
            let before = self.buf.len();
            resampler.finish_into(&mut self.buf)?;
            self.decoded_16k_total += self.buf.len() - before;
            self.finished = true;
        }
        Ok(())
    }
}

#[cfg(feature = "file-decode")]
impl PcmWindows for FileWindows {
    fn spec(&self) -> WindowSpec {
        self.spec
    }

    fn next_window(&mut self) -> Result<Option<PcmWindow<'_>>, GigasttError> {
        if self.cursor.is_done() {
            return Ok(None);
        }

        // Reclaim the previous window's consumed prefix: windows only move
        // forward, so everything before `next_start` is dead. This is what keeps
        // the resident buffer at one window plus look-ahead.
        let drop = self
            .cursor
            .next_start()
            .saturating_sub(self.buf_start_abs)
            .min(self.buf.len());
        if drop > 0 {
            self.buf.drain(0..drop);
            self.buf_start_abs += drop;
        }

        self.fill_to(self.cursor.fill_target())
            .map_err(decode_error)?;

        let Some((start, end)) = self.cursor.take(self.decoded_16k_total, self.eof) else {
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

#[cfg(test)]
mod tests;

/// [`FileWindows`] streaming-decode tests: prove that pulling windows from the
/// container yields byte-identical geometry to [`SliceWindows`] over the same
/// fully-decoded buffer — and, below the single-pass ceiling, exactly one window
/// over the whole buffer (matching `Engine::decode_words`' non-windowed branch).
/// No model is required.
#[cfg(all(test, feature = "file-decode"))]
mod file_windows_tests;
