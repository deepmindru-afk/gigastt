//! Ogg/Opus packet decode and soft-EOF demux helpers.

#[cfg(feature = "file-decode")]
use anyhow::Result;
#[cfg(feature = "file-decode")]
use symphonia::core::formats::FormatReader;

#[cfg(feature = "file-decode")]
use super::audio_too_long_err;

/// Opus always decodes at 48 kHz regardless of the container's declared input
/// rate (RFC 7845 §5.1), so this — not the header rate — is the unit the length
/// budget counts and a duration trip reports.
#[cfg(feature = "file-decode")]
pub(super) const OPUS_DECODE_RATE: u32 = 48_000;

/// Maximum decoded samples per channel for one Opus packet: 120 ms at 48 kHz
/// (RFC 6716 §3.2.5). A packet claiming more is malformed.
#[cfg(feature = "file-decode")]
const OPUS_MAX_PACKET_SAMPLES: usize = 5760;

/// Total decoded samples per channel for an Opus packet at 48 kHz, parsed
/// from the TOC byte (RFC 6716 §3.1): the 5-bit configuration selects the
/// per-frame duration and the 2 low bits the frame count (code 3 reads it
/// from the second byte). The `opus-rs` decoder API takes the exact packet
/// duration rather than an output-buffer capacity, so it is computed here
/// instead of trusting demuxer timestamps.
#[cfg(feature = "file-decode")]
pub(crate) fn opus_packet_frame_size(packet: &[u8]) -> Option<usize> {
    // Per-frame duration in 48 kHz samples for each of the 32 TOC
    // configurations (RFC 6716 Table 2): SILK 10/20/40/60 ms, hybrid 10/20
    // ms, CELT 2.5/5/10/20 ms.
    #[rustfmt::skip]
    const FRAME_DURATION_48K: [usize; 32] = [
        480, 960, 1920, 2880, // SILK narrowband
        480, 960, 1920, 2880, // SILK mediumband
        480, 960, 1920, 2880, // SILK wideband
        480, 960,             // hybrid super-wideband
        480, 960,             // hybrid fullband
        120, 240, 480, 960,   // CELT narrowband
        120, 240, 480, 960,   // CELT wideband
        120, 240, 480, 960,   // CELT super-wideband
        120, 240, 480, 960,   // CELT fullband
    ];
    let toc = *packet.first()?;
    let frames = match toc & 0b11 {
        0 => 1,
        1 | 2 => 2,
        _ => usize::from(packet.get(1)? & 0x3F),
    };
    if frames == 0 {
        return None;
    }
    let size = FRAME_DURATION_48K[(toc >> 3) as usize] * frames;
    (size <= OPUS_MAX_PACKET_SAMPLES).then_some(size)
}

/// True when a demuxer `next_packet` failure is a recoverable end-of-stream.
///
/// Symphonia surfaces a missing Ogg EOS page as `IoError(UnexpectedEof)` rather
/// than `Ok(None)`. Real-world producers (notably Android Telegram voice notes)
/// often omit EOS; if any PCM has already been decoded, treat that EOF as a
/// clean stream end so the upload still transcribes (see issue #217).
#[cfg(feature = "file-decode")]
pub(crate) fn is_recoverable_packet_eof(err: &symphonia::core::errors::Error) -> bool {
    matches!(
        err,
        symphonia::core::errors::Error::IoError(ioe)
            if ioe.kind() == std::io::ErrorKind::UnexpectedEof
    )
}

/// Pull the next demux packet, treating UnexpectedEof after successful PCM as EOS.
///
/// Returns `Ok(Some(packet))` to decode, `Ok(None)` to end the loop, or `Err`
/// for non-recoverable demux failures (and EOF with no audio yet).
#[cfg(feature = "file-decode")]
pub(super) fn next_demux_packet(
    format: &mut dyn FormatReader,
    have_pcm: bool,
) -> Result<Option<symphonia::core::packet::Packet>> {
    match format.next_packet() {
        Ok(Some(p)) => Ok(Some(p)),
        Ok(None) => Ok(None),
        Err(e) if is_recoverable_packet_eof(&e) && have_pcm => {
            tracing::debug!(
                "Demux UnexpectedEof after PCM already decoded; treating as end of stream"
            );
            Ok(None)
        }
        Err(e) => Err(anyhow::anyhow!("Error reading packet: {e}")),
    }
}

/// Reject channel layouts the `opus-rs` fallback cannot decode. Mono/stereo
/// covers Telegram voice notes, browser MediaRecorder captures, and `.opus`
/// files; multistream (>2ch) OGG/Opus is out.
#[cfg(feature = "file-decode")]
fn check_opus_channels(channels: usize) -> Result<()> {
    if !(1..=2).contains(&channels) {
        anyhow::bail!("Opus with {channels} channels is not supported (mono/stereo only)");
    }
    Ok(())
}

/// Per RFC 7845 the decode rate is always 48 kHz, whatever the container's
/// declared input rate.
#[cfg(feature = "file-decode")]
fn new_opus_decoder(channels: usize) -> Result<opus_rs::OpusDecoder> {
    opus_rs::OpusDecoder::new(OPUS_DECODE_RATE as i32, channels)
        .map_err(|e| anyhow::anyhow!("Opus decoder init failed: {e}"))
}

/// Decode one packet into `pcm` (interleaved, `channels` deep), returning the
/// per-channel frame count. The single place that knows the packet contract —
/// the `opus-rs` API takes the exact packet duration rather than a buffer
/// capacity — shared by the whole-buffer and streaming decoders so they cannot
/// drift apart.
#[cfg(feature = "file-decode")]
fn decode_packet_interleaved(
    decoder: &mut opus_rs::OpusDecoder,
    channels: usize,
    data: &[u8],
    pcm: &mut Vec<f32>,
) -> Result<usize> {
    let frame_size =
        opus_packet_frame_size(data).ok_or_else(|| anyhow::anyhow!("Malformed Opus packet"))?;
    pcm.resize(frame_size * channels, 0.0);
    let decoded = decoder
        .decode(data, frame_size, pcm)
        .map_err(|e| anyhow::anyhow!("Opus decode error: {e}"))?
        .min(frame_size);
    Ok(decoded)
}

/// Packet-at-a-time Opus decode, mixed down to mono as it goes.
///
/// The streaming counterpart of [`decode_opus_channels`], which has to hold
/// every channel of the whole file before anything can be mixed or resampled —
/// the reason the Opus path stayed on the whole-buffer duration ceiling while
/// every other container streamed. This one holds a single packet.
///
/// The mix is the same mean
/// [`mix_channels_to_mono`](super::decode::mix_channels_to_mono) computes, in
/// the same summation order, so the samples are identical rather than merely
/// equivalent.
#[cfg(feature = "file-decode")]
pub(super) struct OpusStream {
    decoder: opus_rs::OpusDecoder,
    channels: usize,
    /// Interleaved per-packet scratch, hoisted out of the decode loop.
    pcm: Vec<f32>,
}

#[cfg(feature = "file-decode")]
impl OpusStream {
    pub(super) fn new(channels: usize) -> Result<Self> {
        check_opus_channels(channels)?;
        Ok(Self {
            decoder: new_opus_decoder(channels)?,
            channels,
            pcm: Vec::new(),
        })
    }

    /// Decode one packet, appending its mono 48 kHz samples to `out`. Returns
    /// the number of frames appended.
    pub(super) fn decode_packet(&mut self, data: &[u8], out: &mut Vec<f32>) -> Result<usize> {
        let frames =
            decode_packet_interleaved(&mut self.decoder, self.channels, data, &mut self.pcm)?;
        push_mono_mix(&self.pcm, self.channels, frames, out);
        Ok(frames)
    }
}

/// Append the mono mix of the first `frames` interleaved frames of `pcm`.
///
/// Deliberately the same arithmetic as
/// [`mix_channels_to_mono`](super::decode::mix_channels_to_mono) over the same
/// samples de-interleaved — same summation order, same divisor — so mixing
/// per packet cannot drift from mixing the whole file at once. Pinned against
/// it in `tests.rs`.
#[cfg(feature = "file-decode")]
pub(super) fn push_mono_mix(pcm: &[f32], channels: usize, frames: usize, out: &mut Vec<f32>) {
    let ch = channels as f32;
    for frame in 0..frames {
        let base = frame * channels;
        out.push(pcm[base..base + channels].iter().sum::<f32>() / ch);
    }
}

/// Decode the packets of an Opus track (OGG container) to per-channel f32
/// samples at 48 kHz.
///
/// Symphonia's OGG demuxer recognizes Opus (`CODEC_ID_OPUS`) but ships no
/// Opus decoder, so packets are decoded here with the pure-Rust `opus-rs`
/// libopus port (decoder only). Per RFC 7845 the decode rate is always
/// 48 kHz — the rate symphonia's mapper reports — and callers resample to
/// 16 kHz like for every other format. Only mono and stereo are supported,
/// which covers Telegram voice notes, browser MediaRecorder captures, and
/// `.opus` files; multistream (>2ch) OGG/Opus is rejected. `max_samples` is
/// the per-channel (48 kHz) sample budget, enforced incrementally as in the
/// symphonia decode loops; `limit_secs` is the seconds figure reported on a
/// trip.
#[cfg(feature = "file-decode")]
pub(super) fn decode_opus_channels(
    format: &mut dyn FormatReader,
    track_id: u32,
    channels: usize,
    max_samples: usize,
    limit_secs: f64,
) -> Result<Vec<Vec<f32>>> {
    check_opus_channels(channels)?;
    let mut decoder = new_opus_decoder(channels)?;
    let mut per_channel: Vec<Vec<f32>> = (0..channels).map(|_| Vec::new()).collect();
    let mut pcm: Vec<f32> = Vec::new();
    loop {
        let have_pcm = per_channel.first().is_some_and(|c| !c.is_empty());
        let Some(packet) = next_demux_packet(format, have_pcm)? else {
            break;
        };
        if packet.track_id != track_id {
            continue;
        }
        let decoded = decode_packet_interleaved(&mut decoder, channels, &packet.data, &mut pcm)?;
        if channels == 1 {
            per_channel[0].extend_from_slice(&pcm[..decoded]);
        } else {
            for frame in 0..decoded {
                for (c, buf) in per_channel.iter_mut().enumerate() {
                    buf.push(pcm[frame * channels + c]);
                }
            }
        }
        // Incremental length budget, same as the symphonia decode loops.
        let decoded_len = per_channel.first().map(|v| v.len()).unwrap_or(0);
        if decoded_len > max_samples {
            return Err(audio_too_long_err(
                decoded_len,
                OPUS_DECODE_RATE,
                limit_secs,
            ));
        }
    }
    Ok(per_channel)
}
