//! Container decode (symphonia paths), channel mix, and dual-mono detection.

#[cfg(feature = "file-decode")]
use anyhow::Context;
use anyhow::Result;
use bytes::Bytes;
#[cfg(feature = "file-decode")]
use symphonia::core::codecs::audio::AudioDecoderOptions;
#[cfg(feature = "file-decode")]
use symphonia::core::codecs::audio::well_known::CODEC_ID_OPUS;
#[cfg(feature = "file-decode")]
use symphonia::core::formats::probe::Hint;
#[cfg(feature = "file-decode")]
use symphonia::core::formats::{FormatOptions, TrackType};
#[cfg(feature = "file-decode")]
use symphonia::core::io::{MediaSource, MediaSourceStream};
#[cfg(feature = "file-decode")]
use symphonia::core::meta::MetadataOptions;

use super::DUAL_MONO_CORRELATION_THRESHOLD;
#[cfg(feature = "file-decode")]
use super::MAX_SAMPLE_RATE;
#[cfg(feature = "file-decode")]
use super::opus::{decode_opus_channels, next_demux_packet};
#[cfg(feature = "file-decode")]
use super::resample::{RESAMPLE_STAGING_FRAMES, ResampleTo16k, SampleRate};
#[cfg(feature = "file-decode")]
use super::stream::FileWindows;
#[cfg(feature = "file-decode")]
use super::{audio_too_long_err, resolve_budget, whole_buffer_limit_secs};

/// A [`MediaSource`] that borrows its data from a reference-counted [`Bytes`]
/// buffer instead of cloning into a `Vec<u8>`.
///
/// Axum delivers REST upload bodies as `axum::body::Bytes`, which re-exports
/// `bytes::Bytes`. Before this type the decode path called `body.to_vec()` and
/// then wrapped the clone in `std::io::Cursor`, doubling the transient
/// memory footprint for every upload (a 50 MiB body briefly held 100 MiB in
/// RAM, plus another symphonia-side clone). `Bytes::clone` is a refcount bump,
/// so the shared variant decodes the original axum buffer in place.
///
/// The type is deliberately small and crate-private: it only needs to satisfy
/// `Read + Seek + Send + Sync` so symphonia's `MediaSourceStream` can drive it.
#[allow(dead_code)] // unused when `file-decode` is off (raw-PCM-only lean build)
pub(crate) struct BytesMediaSource {
    data: Bytes,
    pos: u64,
}

#[allow(dead_code)] // `new` is only called by the file-decode path
impl BytesMediaSource {
    pub(crate) fn new(data: Bytes) -> Self {
        Self { data, pos: 0 }
    }
}

impl std::io::Read for BytesMediaSource {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let len = self.data.len() as u64;
        if self.pos >= len {
            return Ok(0);
        }
        let start = self.pos as usize;
        let available = self.data.len() - start;
        let n = available.min(buf.len());
        buf[..n].copy_from_slice(&self.data[start..start + n]);
        self.pos += n as u64;
        Ok(n)
    }
}

impl std::io::Seek for BytesMediaSource {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        let len = self.data.len() as u64;
        // `std::io::Seek` semantics: seeking past the end is allowed; the next
        // read returns 0. Seeking to a negative offset is an error.
        let new_pos: i128 = match pos {
            std::io::SeekFrom::Start(n) => n as i128,
            std::io::SeekFrom::End(off) => len as i128 + off as i128,
            std::io::SeekFrom::Current(off) => self.pos as i128 + off as i128,
        };
        if new_pos < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "seek before start of buffer",
            ));
        }
        self.pos = new_pos as u64;
        Ok(self.pos)
    }
}

#[cfg(feature = "file-decode")]
impl MediaSource for BytesMediaSource {
    fn is_seekable(&self) -> bool {
        true
    }

    fn byte_len(&self) -> Option<u64> {
        Some(self.data.len() as u64)
    }
}

// docs-drift: codecs
// Canonical decode surface, one token per supported input. Kept in sync with
// the FORMATS table in scripts/check-docs-drift.py and the format lists in
// docs/api.md ("Audio formats and telephony codecs") and docs/cli.md
// ("Supports:" line) — update all three together when adding a codec.
// wav
// wav-g711
// wav-g722
// mp3
// m4a
// ogg-vorbis
// ogg-opus
// flac
// raw-pcmu
// raw-pcma
// raw-g722
// docs-drift: end

/// Decode any supported audio file to mono f32 samples at 16kHz.
///
/// Supports WAV, MP3, M4A/AAC, OGG/Vorbis, OGG/Opus (`.opus`), and FLAC.
/// Multi-channel audio is mixed to mono. This flat decode materializes the whole
/// buffer, so it is bounded by the ~30-minute whole-buffer safety ceiling; the
/// streaming file path (`Engine::transcribe_request`) pulls windows instead and
/// has no length limit.
///
/// # Errors
///
/// Returns an error if the file cannot be opened or decoded, or exceeds the
/// whole-buffer safety ceiling.
///
/// ```text
/// { !path.is_empty() }
/// fn decode_audio_file(path: &str) -> Result<Vec<f32>>
/// { ret.as_ref().map(|v| !v.is_empty() || path.is_empty()).unwrap_or(true) }
/// ```
#[cfg(feature = "file-decode")]
pub fn decode_audio_file(path: &str) -> Result<Vec<f32>> {
    decode_audio_file_bounded(path, None)
}

/// Flat decode with an explicit operator length budget. A flat drain
/// materializes the whole buffer, so it is always bounded by at least the
/// whole-buffer safety ceiling ([`whole_buffer_limit_secs`] clamps `max_audio_secs`
/// down to it); the engine's whole-buffer branch passes the request's
/// `--max-audio-secs`, the public wrapper passes `None` (ceiling only). Callers
/// that want peak memory independent of duration go through
/// `Engine::transcribe_request`, which pulls windows instead of draining.
#[cfg(feature = "file-decode")]
pub(crate) fn decode_audio_file_bounded(
    path: &str,
    max_audio_secs: Option<f64>,
) -> Result<Vec<f32>> {
    FileWindows::decode_file(path, Some(whole_buffer_limit_secs(max_audio_secs)))
}

/// Decode audio from raw bytes in memory (no temp file needed).
///
/// Backwards-compatible shim: clones `data` into a [`Bytes`] and delegates
/// to [`decode_audio_bytes_shared`]. New call sites should pass a
/// `bytes::Bytes` (or `axum::body::Bytes`) directly to avoid the copy.
///
/// # Errors
///
/// Returns an error if the bytes cannot be decoded or the audio exceeds the
/// whole-buffer safety ceiling.
///
/// ```text
/// { true }
/// fn decode_audio_bytes(data: &[u8]) -> Result<Vec<f32>>
/// { ret.as_ref().map(|v| !v.is_empty()).unwrap_or(true) }
/// ```
#[cfg(feature = "file-decode")]
pub fn decode_audio_bytes(data: &[u8]) -> Result<Vec<f32>> {
    decode_audio_bytes_shared(Bytes::copy_from_slice(data))
}

/// Decode audio from a shared [`Bytes`] buffer in place — no `to_vec()` clone.
///
/// Same logic as [`decode_audio_file`] but reads from a reference-counted
/// in-memory buffer. Supports WAV, MP3, M4A/AAC, OGG/Vorbis, OGG/Opus
/// (`.opus`), and FLAC. Multi-channel audio is mixed to mono. The whole-buffer
/// safety ceiling is enforced **incrementally** on each decoded packet: a
/// malicious or malformed upload is aborted before its decoded samples blow up
/// RAM.
///
/// # Errors
///
/// Returns an error if the bytes cannot be decoded or the audio exceeds the
/// whole-buffer safety ceiling.
///
/// ```text
/// { true }
/// fn decode_audio_bytes_shared(data: Bytes) -> Result<Vec<f32>>
/// { ret.as_ref().map(|v| !v.is_empty()).unwrap_or(true) }
/// ```
#[cfg(feature = "file-decode")]
pub fn decode_audio_bytes_shared(data: Bytes) -> Result<Vec<f32>> {
    decode_audio_bytes_shared_bounded(data, None)
}

/// Flat byte decode with an explicit operator length budget. `None` behaves
/// exactly like [`decode_audio_bytes_shared`] (whole-buffer ceiling); a
/// `Some(secs)` from `--max-audio-secs` lowers it. Public so the SSE streaming
/// handler, which materializes the whole buffer before chunking, can thread the
/// operator limit into its own decode.
#[cfg(feature = "file-decode")]
pub fn decode_audio_bytes_shared_bounded(
    data: Bytes,
    max_audio_secs: Option<f64>,
) -> Result<Vec<f32>> {
    FileWindows::decode_bytes(data, Some(whole_buffer_limit_secs(max_audio_secs)))
}

/// Read an audio container's declared duration (seconds) from its header
/// WITHOUT decoding a single audio packet.
///
/// Runs only symphonia's format probe and reads the default audio track's
/// declared frame count and sample rate — an O(header) read, not the O(T)
/// decode `decode_audio_bytes_shared` performs. Returns:
/// - `Ok(Some(secs))` when the container declares a positive frame count and a
///   plausible sample rate (WAV, FLAC, M4A, and OGG usually do);
/// - `Ok(None)` when the stream is probeable but declares no usable duration
///   (a raw MP3 stream typically does not, and neither does a crafted header);
/// - `Err(_)` when the bytes cannot be probed as a supported container at all.
///
/// Use this to size a long job's progress bar without paying for a full decode
/// whose samples are immediately discarded; fall back to a real decode on
/// anything other than `Ok(Some(_))`.
///
/// ```text
/// { true }
/// fn probe_duration_bytes(data: Bytes) -> Result<Option<f64>>
/// { true }
/// ```
#[cfg(feature = "file-decode")]
pub fn probe_duration_bytes(data: Bytes) -> Result<Option<f64>> {
    let source = BytesMediaSource::new(data);
    let mss = MediaSourceStream::new(Box::new(source), Default::default());
    probe_duration_inner(mss, Hint::new())
}

/// Path twin of [`probe_duration_bytes`]: read a file's declared duration from
/// its header without decoding. Seeds the probe hint from the extension exactly
/// as [`decode_audio_file`] does.
#[cfg(feature = "file-decode")]
pub fn probe_duration_file(path: &str) -> Result<Option<f64>> {
    let file =
        std::fs::File::open(path).with_context(|| format!("Failed to open audio file: {path}"))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
    {
        hint.with_extension(ext);
    }
    probe_duration_inner(mss, hint)
}

/// Shared probe: identify the container and read the default audio track's
/// declared frame count / sample rate, deriving duration = frames / rate. No
/// packet is ever decoded, so a container that does not declare its length
/// yields `Ok(None)` rather than a scanned duration.
#[cfg(feature = "file-decode")]
fn probe_duration_inner(mss: MediaSourceStream<'_>, hint: Hint) -> Result<Option<f64>> {
    let format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .context("Unsupported audio format")?;

    let Some(track) = format.default_track(TrackType::Audio) else {
        return Ok(None);
    };
    let Some(audio_params) = track.codec_params.as_ref().and_then(|p| p.audio()) else {
        return Ok(None);
    };
    let Some(sample_rate) = audio_params.sample_rate else {
        return Ok(None);
    };
    // A crafted / implausible header rate can't be trusted to scale a duration:
    // report "unknown" and let the caller fall back to a real decode, which
    // clamps the rate and enforces the length budget incrementally.
    if sample_rate == 0 || sample_rate > MAX_SAMPLE_RATE {
        return Ok(None);
    }
    match track.num_frames {
        Some(n) if n > 0 => Ok(Some(n as f64 / sample_rate as f64)),
        _ => Ok(None),
    }
}

/// Decode an audio file to one f32 sample vector per channel at 16 kHz.
///
/// Same probe/decode/resample pipeline as [`decode_audio_file`], but keeps
/// channels separate. The mono mix path remains unchanged.
#[cfg(feature = "file-decode")]
pub fn load_audio_channels(path: &str) -> Result<Vec<Vec<f32>>> {
    load_audio_channels_bounded(path, None)
}

/// `channels=split` decode with an explicit operator length budget. This path
/// holds every channel's decoded buffer in RAM at once, so it is bounded by at
/// least the whole-buffer safety ceiling; the server threads the request's
/// `--max-audio-secs` here, the public wrapper passes `None` (ceiling only).
#[cfg(feature = "file-decode")]
pub(crate) fn load_audio_channels_bounded(
    path: &str,
    max_audio_secs: Option<f64>,
) -> Result<Vec<Vec<f32>>> {
    let file =
        std::fs::File::open(path).with_context(|| format!("Failed to open audio file: {path}"))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
    {
        hint.with_extension(ext);
    }

    let source_label = format!(
        "format={}",
        std::path::Path::new(path)
            .extension()
            .unwrap_or_default()
            .to_string_lossy()
    );

    decode_audio_inner_channels(mss, hint, &source_label, max_audio_secs)
}

/// Decode raw audio bytes to one f32 sample vector per channel at 16 kHz.
#[cfg(feature = "file-decode")]
pub fn decode_audio_bytes_shared_channels(data: Bytes) -> Result<Vec<Vec<f32>>> {
    decode_audio_bytes_shared_channels_bounded(data, None)
}

/// `channels=split` byte decode with an explicit operator length budget — the
/// bytes twin of the crate-internal path decoder. `None` behaves exactly like
/// [`decode_audio_bytes_shared_channels`] (the whole-buffer safety ceiling); a
/// `Some(secs)` from the server's `--max-audio-secs` lowers it. Public so the
/// server can thread the operator limit into its own `channels=split` decode.
#[cfg(feature = "file-decode")]
pub fn decode_audio_bytes_shared_channels_bounded(
    data: Bytes,
    max_audio_secs: Option<f64>,
) -> Result<Vec<Vec<f32>>> {
    let source = BytesMediaSource::new(data);
    let mss = MediaSourceStream::new(Box::new(source), Default::default());
    decode_audio_inner_channels(mss, Hint::new(), "bytes", max_audio_secs)
}

/// What a one-pass scan of a container's channels found.
#[cfg(feature = "file-decode")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChannelScan {
    /// Channels the container declares.
    pub channels: usize,
    /// True only when there are exactly two and they are near-identical — a PBX
    /// that recorded the same mix to both. Transcribing those as two speakers
    /// would duplicate every word.
    pub dual_mono: bool,
}

impl ChannelScan {
    /// Whether `channels=split` should fall back to the mono mix, and why.
    /// `None` means genuine stereo: split it.
    pub fn mono_fallback_reason(&self) -> Option<&'static str> {
        match self.channels {
            0 => Some("no channels"),
            1 => Some("mono audio"),
            2 if self.dual_mono => Some("dual-mono audio"),
            2 => None,
            _ => Some("more than two channels"),
        }
    }
}

/// Decide how `channels=split` should treat `data` **without materializing it**.
///
/// The split path used to answer this by decoding every channel of the whole
/// file and correlating the two in full, which is what pinned it to a duration
/// ceiling. Almost all of the answer is in the header: anything that is not
/// exactly two channels falls back to the mono mix, and that needs no decode at
/// all. Only the two-channel case has to look at the audio, and
/// [`DualMonoDetector`] does that in one pass over six accumulators.
///
/// The correlation is taken on the **16 kHz resampled** channels, on the same
/// per-packet staging cadence the whole-buffer decode used, so it is the same
/// statistic on the same numbers — the verdict does not move.
///
/// OGG/Opus is the exception: it has no packet-wise decoder here, so it keeps
/// the whole-buffer decode (and its ceiling) for this decision.
#[cfg(feature = "file-decode")]
pub fn scan_channels(data: Bytes, max_audio_secs: Option<f64>) -> Result<ChannelScan> {
    let source = BytesMediaSource::new(data.clone());
    let mss = MediaSourceStream::new(Box::new(source), Default::default());
    let mut format = symphonia::default::get_probe()
        .probe(
            &Hint::new(),
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .context("Unsupported audio format")?;

    let (track_id, sample_rate, channels, is_opus) = {
        let track = format
            .default_track(TrackType::Audio)
            .context("No audio track found")?;
        let params = track
            .codec_params
            .as_ref()
            .and_then(|p| p.audio())
            .context("No audio codec parameters")?;
        let sample_rate = params.sample_rate.context("Unknown sample rate")?;
        if sample_rate == 0 || sample_rate > MAX_SAMPLE_RATE {
            anyhow::bail!("Unsupported sample rate: {sample_rate}Hz");
        }
        (
            track.id,
            sample_rate,
            params.channels.as_ref().map(|c| c.count()).unwrap_or(1),
            params.codec == CODEC_ID_OPUS,
        )
    };

    // Header-only verdict: no decode, whatever the file's length.
    if channels != 2 {
        return Ok(ChannelScan {
            channels,
            dual_mono: false,
        });
    }

    if is_opus {
        let decoded = decode_audio_bytes_shared_channels_bounded(data, max_audio_secs)?;
        return Ok(ChannelScan {
            channels: decoded.len(),
            dual_mono: is_dual_mono(&decoded),
        });
    }

    let (max_samples, limit_secs) = resolve_budget(max_audio_secs, sample_rate);
    let mut decoder = {
        let track = format
            .default_track(TrackType::Audio)
            .context("No audio track found")?;
        let params = track
            .codec_params
            .as_ref()
            .and_then(|p| p.audio())
            .context("No audio codec parameters")?;
        symphonia::default::get_codecs()
            .make_audio_decoder(params, &AudioDecoderOptions::default())
            .context("Unsupported audio codec")?
    };

    // One resampler per channel, fed on the same cadence the whole-buffer
    // decode used, so the 16 kHz samples reaching the detector are the ones it
    // would have correlated.
    let mut acc: Vec<ResampleTo16k> = (0..2)
        .map(|_| ResampleTo16k::new(SampleRate(sample_rate), None))
        .collect();
    let mut detector = DualMonoDetector::new();
    let mut ready: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
    let mut interleaved: Vec<f32> = Vec::new();
    let mut source_frames: usize = 0;

    loop {
        let Some(packet) = next_demux_packet(&mut *format, source_frames > 0)? else {
            break;
        };
        if packet.track_id != track_id {
            continue;
        }
        let decoded = decoder.decode(&packet).context("Decode error")?;
        let num_frames = decoded.frames();
        let ch = decoded.spec().channels().count();
        if ch < 2 {
            // A mid-stream drop to mono makes the pair meaningless; treat the
            // stream as not dual-mono and let the split path decide per channel.
            break;
        }
        interleaved.clear();
        decoded.copy_to_vec_interleaved(&mut interleaved);
        for (c, chan) in acc.iter_mut().enumerate() {
            let stage = chan.stage();
            for frame in 0..num_frames {
                stage.push(interleaved[frame * ch + c]);
            }
        }
        source_frames += num_frames;
        if source_frames > max_samples {
            return Err(audio_too_long_err(source_frames, sample_rate, limit_secs));
        }
        for (c, chan) in acc.iter_mut().enumerate() {
            chan.flush_full()?;
            chan.drain_ready_into(&mut ready[c]);
        }
        let (left, right) = ready.split_at_mut(1);
        detector.push(&left[0], &right[0]);
        ready[0].clear();
        ready[1].clear();
    }

    for (c, chan) in acc.into_iter().enumerate() {
        let mut tail = Vec::new();
        let mut chan = chan;
        chan.finish_into(&mut tail)?;
        ready[c] = tail;
    }
    let (left, right) = ready.split_at_mut(1);
    detector.push(&left[0], &right[0]);

    Ok(ChannelScan {
        channels: 2,
        dual_mono: detector.is_dual_mono(),
    })
}

/// Shared non-mixing decode: probe → format → decode → per-channel resample.
#[cfg(feature = "file-decode")]
fn decode_audio_inner_channels<'s>(
    mss: MediaSourceStream<'s>,
    hint: Hint,
    source_label: &str,
    max_audio_secs: Option<f64>,
) -> Result<Vec<Vec<f32>>> {
    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .context("Unsupported audio format")?;

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
    let n_frames_hint = track.num_frames;
    let (max_samples, limit_secs) =
        resolve_budget(Some(whole_buffer_limit_secs(max_audio_secs)), sample_rate);

    tracing::info!("Audio ({source_label}): {sample_rate}Hz, {channels}ch (split)");

    // Each channel is an independent stream, so each gets its own staging
    // buffer and cached resampler; none of them ever holds more than one
    // chunk of source-rate audio.
    let hint = match n_frames_hint {
        Some(n) if n > 0 && n <= max_samples as u64 => Some(n as usize),
        _ => None,
    };
    // Source-rate frame count of the first channel, tracked separately because
    // the accumulators now hold 16 kHz samples while the length budget is
    // expressed in source-rate frames.
    let mut source_frames: usize = 0;

    // Symphonia demuxes OGG/Opus but ships no Opus decoder, so Opus packets
    // go through the `opus-rs` fallback and rejoin the shared resample tail.
    let acc: Vec<ResampleTo16k> = if audio_params.codec == CODEC_ID_OPUS {
        let decoded =
            decode_opus_channels(&mut *format, track_id, channels, max_samples, limit_secs)?;
        source_frames = decoded.first().map(|v| v.len()).unwrap_or(0);
        let mut acc = Vec::with_capacity(decoded.len());
        for samples in decoded {
            let mut chan = ResampleTo16k::new(SampleRate(sample_rate), Some(samples.len()));
            for piece in samples.chunks(RESAMPLE_STAGING_FRAMES) {
                chan.stage().extend_from_slice(piece);
                chan.flush_full()?;
            }
            acc.push(chan);
        }
        acc
    } else {
        let mut decoder = symphonia::default::get_codecs()
            .make_audio_decoder(audio_params, &AudioDecoderOptions::default())
            .context("Unsupported audio codec")?;

        let mut acc: Vec<ResampleTo16k> = (0..channels)
            .map(|_| ResampleTo16k::new(SampleRate(sample_rate), hint))
            .collect();

        loop {
            let have_pcm = source_frames > 0;
            let Some(packet) = next_demux_packet(&mut *format, have_pcm)? else {
                break;
            };

            if packet.track_id != track_id {
                continue;
            }

            let decoded = decoder.decode(&packet).context("Decode error")?;
            let spec = decoded.spec().clone();
            let num_frames = decoded.frames();
            let ch = spec.channels().count();

            if ch > acc.len() {
                // A channel that appears mid-stream starts from this packet,
                // so it gets no length hint.
                acc.resize_with(ch, || ResampleTo16k::new(SampleRate(sample_rate), None));
            }

            if ch > 1 {
                let mut interleaved: Vec<f32> = Vec::with_capacity(num_frames * ch);
                decoded.copy_to_vec_interleaved(&mut interleaved);
                for frame in 0..num_frames {
                    for c in 0..ch {
                        acc[c].stage().push(interleaved[frame * ch + c]);
                    }
                }
            } else if !acc.is_empty() {
                let stage = acc[0].stage();
                let offset = stage.len();
                stage.resize(offset + num_frames, 0.0);
                decoded.copy_to_slice_interleaved(&mut stage[offset..]);
            }
            // Both branches above grow the first channel by `num_frames`; the
            // `ch <= 1` branch is skipped entirely when there is no channel.
            if !acc.is_empty() {
                source_frames += num_frames;
            }

            if source_frames > max_samples {
                return Err(audio_too_long_err(source_frames, sample_rate, limit_secs));
            }

            for chan in &mut acc {
                chan.flush_full()?;
            }
        }

        acc
    };

    let duration_s = source_frames as f64 / sample_rate as f64;
    tracing::info!(
        "Decoded {} channel(s), first channel {} samples at {}Hz ({:.1}s)",
        acc.len(),
        source_frames,
        sample_rate,
        duration_s
    );

    let channel_count = acc.len();
    let per_channel = acc
        .into_iter()
        .map(ResampleTo16k::finish)
        .collect::<Result<Vec<_>>>()?;
    if sample_rate != 16000 {
        tracing::info!("Resampled {channel_count} channel(s) to 16kHz");
    }

    Ok(per_channel)
}

/// Average multiple channels into a single mono vector.
pub fn mix_channels_to_mono(channels: &[Vec<f32>]) -> Vec<f32> {
    if channels.is_empty() {
        return Vec::new();
    }
    let n = channels.iter().map(|c| c.len()).min().unwrap_or(0);
    (0..n)
        .map(|i| channels.iter().map(|c| c[i]).sum::<f32>() / channels.len() as f32)
        .collect()
}

/// Return `true` if a two-channel stream is dual-mono (both channels nearly
/// identical). Empty or single-channel input returns `false`.
pub fn is_dual_mono(channels: &[Vec<f32>]) -> bool {
    if channels.len() != 2 {
        return false;
    }
    let (left, right) = (&channels[0], &channels[1]);
    if left.is_empty() || right.is_empty() {
        return false;
    }
    let len = left.len().min(right.len());
    normalized_correlation(&left[..len], &right[..len]) > DUAL_MONO_CORRELATION_THRESHOLD
}

/// Streaming form of the dual-mono correlation, for the `channels=split` path
/// over a container that is never fully resident.
///
/// [`is_dual_mono`] needs both channels end to end, which is the only reason
/// channel splitting had to hold the whole decode. The same statistic is
/// computable in one pass with Welford's co-moment recurrence, so this keeps
/// six `f64` accumulators instead of two whole channels.
///
/// The recurrence is used rather than raw power sums (`Σab − ΣaΣb/n`) on
/// purpose: the naive form loses the covariance to cancellation once `n` is in
/// the tens of millions, exactly the regime long files put it in, and the
/// decision it feeds is a knife-edge threshold. Agreement with the batch
/// function is asserted in `tests.rs`, on the value and on the decision.
#[derive(Debug, Default, Clone)]
pub struct DualMonoDetector {
    n: f64,
    mean_a: f64,
    mean_b: f64,
    /// Co-moment Σ(a−ā)(b−b̄).
    c_ab: f64,
    m2_a: f64,
    m2_b: f64,
}

impl DualMonoDetector {
    /// New detector with no samples observed.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed the next block of two channels. Only the overlapping prefix counts,
    /// matching [`is_dual_mono`]'s `min(len)` truncation.
    pub fn push(&mut self, a: &[f32], b: &[f32]) {
        for (&x, &y) in a.iter().zip(b) {
            let (x, y) = (x as f64, y as f64);
            self.n += 1.0;
            let dx = x - self.mean_a;
            let dy = y - self.mean_b;
            self.mean_a += dx / self.n;
            self.mean_b += dy / self.n;
            self.c_ab += dx * (y - self.mean_b);
            self.m2_a += dx * (x - self.mean_a);
            self.m2_b += dy * (y - self.mean_b);
        }
    }

    /// Normalized correlation of everything observed so far, on the same scale
    /// [`is_dual_mono`] thresholds. `0.0` when either channel is silent.
    pub fn correlation(&self) -> f64 {
        if self.n == 0.0 {
            return 0.0;
        }
        let denom = self.m2_a.sqrt() * self.m2_b.sqrt();
        if denom < 1e-12 {
            return 0.0;
        }
        self.c_ab / denom
    }

    /// Whether the two channels are near-identical — the same verdict
    /// [`is_dual_mono`] returns for a fully materialized pair. `false` when no
    /// samples were observed.
    pub fn is_dual_mono(&self) -> bool {
        self.n > 0.0 && self.correlation() > DUAL_MONO_CORRELATION_THRESHOLD
    }
}

/// Test-only alias so the streaming detector can be pinned against the exact
/// batch statistic it replaces.
#[cfg(test)]
pub(super) fn normalized_correlation_for_test(a: &[f32], b: &[f32]) -> f64 {
    normalized_correlation(a, b)
}

fn normalized_correlation(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len();
    if n == 0 || n != b.len() {
        return 0.0;
    }
    let mean_a = a.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    let mean_b = b.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for (&x, &y) in a.iter().zip(b) {
        let dx = x as f64 - mean_a;
        let dy = y as f64 - mean_b;
        cov += dx * dy;
        var_a += dx * dx;
        var_b += dy * dy;
    }
    let denom = var_a.sqrt() * var_b.sqrt();
    if denom < 1e-12 {
        return 0.0;
    }
    cov / denom
}
