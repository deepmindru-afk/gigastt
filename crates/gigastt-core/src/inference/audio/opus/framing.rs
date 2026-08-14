//! RFC 6716 packet framing for the in-tree Opus decoder.

use anyhow::{Context as _, Result};

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
    pub(super) toc: u8,
    /// Decoded samples per frame at 48 kHz — every frame in a packet shares
    /// the single TOC, so one duration covers them all.
    pub(super) samples_per_frame: usize,
    frames: [&'a [u8]; OPUS_MAX_FRAMES],
    count: usize,
}

#[cfg(feature = "file-decode")]
impl<'a> OpusFraming<'a> {
    /// The packet's frames, in order.
    pub(super) fn frames(&self) -> &[&'a [u8]] {
        &self.frames[..self.count]
    }

    /// Total decoded samples per channel for the whole packet at 48 kHz.
    pub(super) fn packet_samples(&self) -> usize {
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
