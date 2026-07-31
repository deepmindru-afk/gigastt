//! Ogg/Opus packet decode and soft-EOF demux helpers.

#[cfg(feature = "file-decode")]
use anyhow::{Context as _, Result};
#[cfg(feature = "file-decode")]
use symphonia::core::formats::FormatReader;

#[cfg(feature = "file-decode")]
use super::audio_too_long_err;
#[cfg(feature = "file-decode")]
use super::stream::ChannelSelect;

/// Opus always decodes at 48 kHz regardless of the container's declared input
/// rate (RFC 7845 §5.1), so this — not the header rate — is the unit the length
/// budget counts and a duration trip reports.
#[cfg(feature = "file-decode")]
pub(super) const OPUS_DECODE_RATE: u32 = 48_000;

/// Maximum decoded samples per channel for one Opus packet: 120 ms at 48 kHz
/// (RFC 6716 §3.2.5). A packet claiming more is malformed.
#[cfg(feature = "file-decode")]
const OPUS_MAX_PACKET_SAMPLES: usize = 5760;

/// Maximum frames in one Opus packet: 120 ms of 2.5 ms frames (RFC 6716 §3.2.5).
#[cfg(feature = "file-decode")]
const OPUS_MAX_FRAMES: usize = 48;

/// Decoded samples per *frame* at 48 kHz for a TOC byte (RFC 6716 §3.1 and
/// Table 2): the 5-bit configuration selects the per-frame duration.
#[cfg(feature = "file-decode")]
pub(crate) fn opus_samples_per_frame(toc: u8) -> usize {
    // SILK 10/20/40/60 ms, hybrid 10/20 ms, CELT 2.5/5/10/20 ms.
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
    FRAME_DURATION_48K[(toc >> 3) as usize]
}

/// The frames of one Opus packet, sliced out of the packet buffer.
#[cfg(feature = "file-decode")]
pub(crate) struct OpusFraming<'a> {
    /// The packet's TOC byte, verbatim.
    toc: u8,
    /// Decoded samples per frame at 48 kHz — every frame in a packet shares
    /// the single TOC, so one duration covers them all.
    samples_per_frame: usize,
    frames: [&'a [u8]; OPUS_MAX_FRAMES],
    count: usize,
}

#[cfg(feature = "file-decode")]
impl<'a> OpusFraming<'a> {
    /// The packet's frames, in order.
    fn frames(&self) -> &[&'a [u8]] {
        &self.frames[..self.count]
    }

    /// Total decoded samples per channel for the whole packet at 48 kHz.
    fn packet_samples(&self) -> usize {
        self.samples_per_frame * self.count
    }
}

/// A frame length as written on the wire (RFC 6716 §3.2.1): one byte below
/// 252, otherwise two bytes encoding `second * 4 + first`. Returns the length
/// and how many bytes it occupied.
#[cfg(feature = "file-decode")]
fn opus_frame_len(data: &[u8]) -> Result<(usize, usize)> {
    let first = usize::from(
        *data
            .first()
            .context("Opus packet: truncated frame length")?,
    );
    if first < 252 {
        return Ok((first, 1));
    }
    let second = usize::from(*data.get(1).context("Opus packet: truncated frame length")?);
    Ok((second * 4 + first, 2))
}

/// The padding length of a code 3 packet (RFC 6716 §3.2.5): a run of bytes
/// where 255 contributes 254 and continues, and any other value terminates.
/// Returns the padding size and how many bytes the count itself occupied.
#[cfg(feature = "file-decode")]
fn opus_padding_len(data: &[u8]) -> Result<(usize, usize)> {
    let mut padding = 0usize;
    let mut used = 0usize;
    loop {
        let byte = usize::from(
            *data
                .get(used)
                .context("Opus packet: truncated padding count")?,
        );
        used += 1;
        if byte == 255 {
            padding += 254;
        } else {
            padding += byte;
            return Ok((padding, used));
        }
    }
}

/// Slice an Opus packet into its frames per RFC 6716 §3.2.
///
/// Done here rather than left to `opus-rs`, whose own multi-frame parser is
/// wrong in two ways we hit in the wild: it never reads the code 3 VBR flag
/// (bit 7 of the frame-count byte), so a CBR code 3 packet — what Chromium's
/// `MediaRecorder` emits at its 60 ms default, three 20 ms frames per packet —
/// is parsed as though the frame lengths were on the wire and rejected; and it
/// decodes frame lengths with a 15-bit continuation scheme instead of §3.2.1,
/// which mis-slices every explicit length of 128 bytes or more. Feeding the
/// decoder one frame at a time as a code 0 packet routes around both.
#[cfg(feature = "file-decode")]
pub(crate) fn split_opus_packet(packet: &[u8]) -> Result<OpusFraming<'_>> {
    let toc = *packet.first().context("Opus packet is empty")?;
    let payload = &packet[1..];
    let mut frames = [&[][..]; OPUS_MAX_FRAMES];

    let count = match toc & 0b11 {
        // One frame, the rest of the packet.
        0 => {
            frames[0] = payload;
            1
        }
        // Two equal frames.
        1 => {
            if !payload.len().is_multiple_of(2) {
                anyhow::bail!("Opus packet: code 1 payload is not evenly divisible");
            }
            let half = payload.len() / 2;
            frames[0] = &payload[..half];
            frames[1] = &payload[half..];
            2
        }
        // Two frames, the first with an explicit length.
        2 => {
            let (first_len, used) = opus_frame_len(payload)?;
            let rest = payload
                .get(used..)
                .context("Opus packet: code 2 length overruns packet")?;
            if first_len > rest.len() {
                anyhow::bail!("Opus packet: code 2 first frame overruns packet");
            }
            frames[0] = &rest[..first_len];
            frames[1] = &rest[first_len..];
            2
        }
        // Arbitrary frame count, optional padding, CBR or VBR lengths.
        _ => {
            let count_byte = *payload
                .first()
                .context("Opus packet: code 3 is missing its frame count")?;
            let count = usize::from(count_byte & 0x3F);
            if !(1..=OPUS_MAX_FRAMES).contains(&count) {
                anyhow::bail!("Opus packet: code 3 frame count {count} out of range");
            }
            let vbr = count_byte & 0x80 != 0;
            let padded = count_byte & 0x40 != 0;

            let mut at = 1usize;
            let mut end = payload.len();
            if padded {
                let (padding, used) = opus_padding_len(
                    payload
                        .get(at..)
                        .context("Opus packet: truncated padding count")?,
                )?;
                at += used;
                end = end
                    .checked_sub(padding)
                    .context("Opus packet: padding overruns packet")?;
                if at > end {
                    anyhow::bail!("Opus packet: padding overlaps the frame data");
                }
            }

            let mut sizes = [0usize; OPUS_MAX_FRAMES];
            if vbr {
                // Every frame but the last carries its length; the last one
                // takes whatever data is left.
                let mut claimed = 0usize;
                for size in sizes[..count - 1].iter_mut() {
                    let (len, used) = opus_frame_len(
                        payload
                            .get(at..end)
                            .context("Opus packet: code 3 lengths overrun packet")?,
                    )?;
                    at += used;
                    *size = len;
                    claimed += len;
                }
                let remaining = end
                    .checked_sub(at)
                    .context("Opus packet: code 3 lengths overrun packet")?;
                if claimed > remaining {
                    anyhow::bail!("Opus packet: code 3 frame lengths overrun packet");
                }
                sizes[count - 1] = remaining - claimed;
            } else {
                // CBR: no lengths on the wire, the data divides evenly.
                let remaining = end
                    .checked_sub(at)
                    .context("Opus packet: code 3 header overruns packet")?;
                if !remaining.is_multiple_of(count) {
                    anyhow::bail!(
                        "Opus packet: code 3 CBR payload of {remaining} does not divide into {count} frames"
                    );
                }
                sizes[..count].fill(remaining / count);
            }

            for (frame, size) in frames[..count].iter_mut().zip(&sizes[..count]) {
                let next = at
                    .checked_add(*size)
                    .filter(|next| *next <= end)
                    .context("Opus packet: code 3 frame overruns packet")?;
                *frame = &payload[at..next];
                at = next;
            }
            if at != end {
                anyhow::bail!("Opus packet: code 3 frames do not fill the packet");
            }
            count
        }
    };

    let framing = OpusFraming {
        toc,
        samples_per_frame: opus_samples_per_frame(toc),
        frames,
        count,
    };
    if framing.packet_samples() > OPUS_MAX_PACKET_SAMPLES {
        anyhow::bail!(
            "Opus packet: {} samples exceeds 120 ms",
            framing.packet_samples()
        );
    }
    Ok(framing)
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
/// the `opus-rs` API takes the exact frame duration rather than a buffer
/// capacity — shared by the whole-buffer and streaming decoders so they cannot
/// drift apart.
///
/// Each frame is handed over as its own code 0 packet (`scratch` holds the
/// rewritten TOC plus the frame). The decoder carries its state across the
/// calls exactly as it would across the frames of one packet, so this is the
/// same decode — only with [`split_opus_packet`] doing the framing instead of
/// the `opus-rs` parser it cannot be trusted with.
#[cfg(feature = "file-decode")]
fn decode_packet_interleaved(
    decoder: &mut opus_rs::OpusDecoder,
    channels: usize,
    data: &[u8],
    pcm: &mut Vec<f32>,
    scratch: &mut Vec<u8>,
) -> Result<usize> {
    let framing = split_opus_packet(data)?;
    let frame_size = framing.samples_per_frame;
    pcm.resize(framing.packet_samples() * channels, 0.0);
    let mut decoded = 0usize;
    for frame in framing.frames() {
        scratch.clear();
        scratch.reserve(frame.len() + 1);
        // Clearing the two code bits turns the shared TOC into a code 0
        // (single-frame) one; the rest of the configuration is untouched.
        scratch.push(framing.toc & 0xFC);
        scratch.extend_from_slice(frame);
        let written = decoder
            .decode(scratch, frame_size, &mut pcm[decoded * channels..])
            .map_err(|e| anyhow::anyhow!("Opus decode error: {e}"))?
            .min(frame_size);
        decoded += written;
    }
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
    /// Mix down, or keep one channel — `channels=split` needs the latter, and
    /// without it a per-channel read of an `.opus` would silently get the mix.
    channel: ChannelSelect,
    /// Interleaved per-packet scratch, hoisted out of the decode loop.
    pcm: Vec<f32>,
    /// Per-frame code 0 packet scratch, hoisted out of the decode loop.
    frame: Vec<u8>,
}

#[cfg(feature = "file-decode")]
impl OpusStream {
    pub(super) fn new(channels: usize, channel: ChannelSelect) -> Result<Self> {
        check_opus_channels(channels)?;
        Ok(Self {
            decoder: new_opus_decoder(channels)?,
            channels,
            channel,
            pcm: Vec::new(),
            frame: Vec::new(),
        })
    }

    /// Decode one packet, appending its mono 48 kHz samples to `out`. Returns
    /// the number of frames appended.
    pub(super) fn decode_packet(&mut self, data: &[u8], out: &mut Vec<f32>) -> Result<usize> {
        let frames = decode_packet_interleaved(
            &mut self.decoder,
            self.channels,
            data,
            &mut self.pcm,
            &mut self.frame,
        )?;
        match self.channel {
            ChannelSelect::Mono => push_mono_mix(&self.pcm, self.channels, frames, out),
            ChannelSelect::One(k) if k < self.channels => {
                for frame in 0..frames {
                    out.push(self.pcm[frame * self.channels + k]);
                }
            }
            ChannelSelect::One(_) => {}
        }
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
    let mut frame: Vec<u8> = Vec::new();
    loop {
        let have_pcm = per_channel.first().is_some_and(|c| !c.is_empty());
        let Some(packet) = next_demux_packet(format, have_pcm)? else {
            break;
        };
        if packet.track_id != track_id {
            continue;
        }
        let decoded =
            decode_packet_interleaved(&mut decoder, channels, &packet.data, &mut pcm, &mut frame)?;
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

#[cfg(all(test, feature = "file-decode"))]
mod tests {
    use super::*;

    /// TOC config 31 (CELT fullband, 20 ms frames) with the requested frame
    /// code — the configuration Chromium's `MediaRecorder` emits.
    fn celt_toc(code: u8) -> u8 {
        (31 << 3) | code
    }

    fn split(packet: &[u8]) -> Result<Vec<Vec<u8>>> {
        let framing = split_opus_packet(packet)?;
        Ok(framing.frames().iter().map(|f| f.to_vec()).collect())
    }

    #[test]
    fn test_samples_per_frame_covers_the_toc_table() {
        // SILK 10/20/40/60 ms.
        assert_eq!(opus_samples_per_frame(0b0000_0000), 480);
        assert_eq!(opus_samples_per_frame(0b0001_1000), 2880);
        // Hybrid 10 ms (config 12).
        assert_eq!(opus_samples_per_frame(0b0110_0000), 480);
        // CELT 2.5 ms (config 16) and CELT fullband 20 ms (config 31).
        assert_eq!(opus_samples_per_frame(0b1000_0000), 120);
        assert_eq!(opus_samples_per_frame(celt_toc(0)), 960);
    }

    #[test]
    fn test_split_code0_is_one_frame() {
        let packet = [celt_toc(0), 1, 2, 3];
        let framing = split_opus_packet(&packet).expect("code 0");
        assert_eq!(framing.frames(), [&[1u8, 2, 3][..]]);
        assert_eq!(framing.packet_samples(), 960);
    }

    #[test]
    fn test_split_code1_halves_the_payload() {
        assert_eq!(
            split(&[celt_toc(1), 1, 2, 3, 4]).expect("code 1"),
            vec![vec![1, 2], vec![3, 4]]
        );
        // 120 ms cap: two 20 ms frames is fine, and the duration doubles.
        let packet = [celt_toc(1), 1, 2];
        let framing = split_opus_packet(&packet).expect("code 1");
        assert_eq!(framing.packet_samples(), 1920);
    }

    #[test]
    fn test_split_code1_odd_payload_is_rejected() {
        assert!(split_opus_packet(&[celt_toc(1), 1, 2, 3]).is_err());
    }

    #[test]
    fn test_split_code2_reads_a_one_byte_length() {
        assert_eq!(
            split(&[celt_toc(2), 2, 1, 2, 3, 4]).expect("code 2"),
            vec![vec![1, 2], vec![3, 4]]
        );
    }

    #[test]
    fn test_split_code2_reads_a_two_byte_length() {
        // RFC 6716 §3.2.1: a first byte of 252 or more means a second byte
        // follows and the length is `second * 4 + first`. `opus-rs` instead
        // treats bit 7 as a continuation flag, so it mis-slices here and at
        // every explicit length of 128 bytes or more.
        let first_len = 253 + 4; // 253 + 1 * 4, encoded as [253, 1]
        let mut packet = vec![celt_toc(2), 253, 1];
        packet.extend(std::iter::repeat_n(0xAA, first_len));
        packet.extend([1, 2, 3]);
        let frames = split(&packet).expect("code 2 with a long first frame");
        assert_eq!(frames[0].len(), first_len);
        assert_eq!(frames[1], vec![1, 2, 3]);
    }

    #[test]
    fn test_split_code2_length_overrunning_the_packet_is_rejected() {
        assert!(split_opus_packet(&[celt_toc(2), 9, 1, 2]).is_err());
    }

    #[test]
    fn test_split_code3_cbr_divides_the_payload_evenly() {
        // The shape Chromium emits: three 20 ms frames, CBR, no padding.
        let mut packet = vec![celt_toc(3), 3];
        packet.extend([1, 2, 3, 4, 5, 6]);
        assert_eq!(
            split(&packet).expect("code 3 CBR"),
            vec![vec![1, 2], vec![3, 4], vec![5, 6]]
        );
    }

    #[test]
    fn test_split_code3_cbr_uneven_payload_is_rejected() {
        let mut packet = vec![celt_toc(3), 3];
        packet.extend([1, 2, 3, 4, 5]);
        assert!(split_opus_packet(&packet).is_err());
    }

    #[test]
    fn test_split_code3_vbr_reads_lengths_for_all_but_the_last_frame() {
        // VBR flag set: the first two frames carry explicit lengths, the last
        // takes the remainder.
        let mut packet = vec![celt_toc(3), 0x80 | 3, 1, 2];
        packet.extend([1, 2, 2, 3, 4, 5]);
        assert_eq!(
            split(&packet).expect("code 3 VBR"),
            vec![vec![1], vec![2, 2], vec![3, 4, 5]]
        );
    }

    #[test]
    fn test_split_code3_cbr_strips_padding() {
        // Padding flag set, one padding byte declaring two bytes of padding.
        let mut packet = vec![celt_toc(3), 0x40 | 2, 2];
        packet.extend([1, 2, 3, 4]);
        packet.extend([0, 0]);
        assert_eq!(
            split(&packet).expect("code 3 CBR with padding"),
            vec![vec![1, 2], vec![3, 4]]
        );
    }

    #[test]
    fn test_split_code3_padding_count_of_255_continues() {
        // A padding byte of 255 contributes 254 and reads another byte.
        let mut packet = vec![celt_toc(3), 0x40 | 1, 255, 1];
        packet.extend([7, 7, 7]);
        packet.extend(std::iter::repeat_n(0, 255));
        assert_eq!(
            split(&packet).expect("code 3 with a long padding run"),
            vec![vec![7, 7, 7]]
        );
    }

    #[test]
    fn test_split_code3_vbr_strips_padding() {
        let mut packet = vec![celt_toc(3), 0xC0 | 2, 2, 1];
        packet.extend([9, 8, 7]);
        packet.extend([0, 0]);
        assert_eq!(
            split(&packet).expect("code 3 VBR with padding"),
            vec![vec![9], vec![8, 7]]
        );
    }

    #[test]
    fn test_split_code3_zero_frame_count_is_rejected() {
        assert!(split_opus_packet(&[celt_toc(3), 0]).is_err());
    }

    #[test]
    fn test_split_code3_over_120ms_is_rejected() {
        // Seven 20 ms frames is 140 ms, past the RFC 6716 packet maximum.
        let mut packet = vec![celt_toc(3), 7];
        packet.extend(std::iter::repeat_n(1, 7));
        assert!(split_opus_packet(&packet).is_err());
    }

    #[test]
    fn test_split_code3_truncated_headers_are_rejected() {
        // Frame count byte missing, padding count missing, and VBR lengths
        // that run past the end.
        assert!(split_opus_packet(&[celt_toc(3)]).is_err());
        assert!(split_opus_packet(&[celt_toc(3), 0x40 | 1]).is_err());
        assert!(split_opus_packet(&[celt_toc(3), 0x80 | 3, 200, 200, 1]).is_err());
    }

    #[test]
    fn test_split_empty_packet_is_rejected() {
        assert!(split_opus_packet(&[]).is_err());
    }
}
