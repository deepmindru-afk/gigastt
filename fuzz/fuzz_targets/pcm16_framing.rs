//! Fuzz the odd-length PCM16 carry-over framing logic.
//!
//! WebSocket clients may split a PCM16 stream on arbitrary byte boundaries;
//! `parse_pcm16_with_carry` keeps a trailing odd byte in `pending` so an
//! odd-length frame doesn't introduce a one-sample phase shift. This target
//! slices the input into many chunks (driven by the input bytes themselves)
//! and threads them through a single `pending` to stress the cross-frame
//! state machine. Property: never panics, and the total sample count equals
//! total_bytes / 2 (the carry invariant) regardless of how bytes are split.
#![no_main]

use gigastt_core::inference::audio::parse_pcm16_with_carry;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let mut pending: Option<u8> = None;
    let mut total_samples = 0usize;
    let mut total_bytes = 0usize;

    // Walk the buffer, carving out chunks of pseudo-arbitrary length so the
    // odd/even split boundaries vary across the run.
    let mut rest = data;
    while !rest.is_empty() {
        // Use the first byte of the remaining slice to pick a chunk length in
        // 0..=255, clamped to what's left.
        let want = rest[0] as usize;
        let take = want.min(rest.len());
        let (chunk, tail) = rest.split_at(take);
        rest = tail;

        total_bytes += chunk.len();
        let samples = parse_pcm16_with_carry(chunk, &mut pending);
        total_samples += samples.len();

        // Guard against an infinite loop when `take == 0` (want == 0 and the
        // slice is non-empty): force progress by consuming one byte.
        if take == 0 {
            total_bytes += 1;
            let one = &rest[..1];
            let s = parse_pcm16_with_carry(one, &mut pending);
            total_samples += s.len();
            rest = &rest[1..];
        }
    }

    // Carry invariant: emitted samples must equal floor(total_bytes / 2).
    assert_eq!(total_samples, total_bytes / 2);
    assert_eq!(pending.is_some(), total_bytes % 2 == 1);
});
