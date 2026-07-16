//! CTC greedy decoding for the GigaAM Multilingual CTC head.
//!
//! Unlike the RNN-T path (encoder + prediction network + joiner), the CTC head is
//! a single encoder that emits per-frame class log-probabilities. Greedy CTC
//! decode = per-frame argmax over the vocab, collapse consecutive repeats (a blank
//! between two identical labels breaks the run and keeps both), then drop blanks.
//!
//! `log_probs` is the encoder output `[1, T', V]` read row-major, so frame `t`'s
//! logits are `log_probs[t*V .. (t+1)*V]` — **frame-major**: `T'` is the outer
//! axis, the vocab the inner. This differs from the RNN-T encoder's channels-first
//! `[1, D, T]` layout (see [`super::decode::extract_encoder_frame`]).

use super::decode::{TokenInfo, argmax_with_confidence};

/// Greedy CTC decode over a flat `log_probs` buffer of shape `[t_total, vocab]`
/// (row-major). Returns the collapsed, blank-stripped tokens with a per-token
/// frame index (relative to this window) and softmax confidence — the same
/// [`TokenInfo`] the RNN-T path emits, so downstream word formatting is shared.
///
/// - `t_len`: number of valid frames (from the encoder's `encoded_lengths[0]`);
///   frames past it are right-padding and ignored, even if the tensor's outer dim
///   is larger.
/// - `vocab`: class count (71 for GigaAM Multilingual).
/// - `blank_id`: CTC blank (70 = `vocab - 1`).
pub(crate) fn ctc_greedy_decode(
    log_probs: &[f32],
    t_len: usize,
    vocab: usize,
    blank_id: usize,
) -> Vec<TokenInfo> {
    if vocab == 0 {
        return Vec::new();
    }
    let usable = t_len.min(log_probs.len() / vocab);
    let mut out = Vec::new();
    let mut prev: Option<usize> = None;
    for t in 0..usable {
        let row = &log_probs[t * vocab..(t + 1) * vocab];
        let (id, confidence) = argmax_with_confidence(row, blank_id);
        // Collapse: skip a frame whose argmax equals the previous frame's argmax
        // (blank or not). Tracking the raw argmax — not the last *emitted* token —
        // is what makes a blank between two identical labels keep both.
        if Some(id) == prev {
            continue;
        }
        prev = Some(id);
        if id == blank_id {
            continue;
        }
        out.push(TokenInfo {
            token_id: id,
            frame_index: t,
            confidence,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `[t, vocab]` row-major log-prob buffer where frame `t` argmaxes to
    /// `ids[t]`.
    fn logits(ids: &[usize], vocab: usize) -> Vec<f32> {
        let mut lp = vec![-10.0f32; ids.len() * vocab];
        for (t, &id) in ids.iter().enumerate() {
            lp[t * vocab + id] = 5.0;
        }
        lp
    }

    #[test]
    fn collapses_repeats_and_drops_blank() {
        // vocab=3, blank=2. frames: a a <blk> b  ->  [a, b]
        let lp = logits(&[0, 0, 2, 1], 3);
        let toks = ctc_greedy_decode(&lp, 4, 3, 2);
        assert_eq!(
            toks.iter().map(|t| t.token_id).collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(toks[0].frame_index, 0);
        assert_eq!(toks[1].frame_index, 3);
    }

    #[test]
    fn blank_separates_identical_labels() {
        // a a <blk> a  ->  [a, a]  (blank breaks the run of identical labels)
        let lp = logits(&[0, 0, 2, 0], 3);
        let toks = ctc_greedy_decode(&lp, 4, 3, 2);
        assert_eq!(
            toks.iter().map(|t| t.token_id).collect::<Vec<_>>(),
            vec![0, 0]
        );
    }

    #[test]
    fn honours_t_len_truncation() {
        // 3 frames in the buffer, but only the first 2 are valid.
        let lp = logits(&[0, 1, 0], 3);
        let toks = ctc_greedy_decode(&lp, 2, 3, 2);
        assert_eq!(toks.len(), 2);
        assert_eq!(toks[1].frame_index, 1);
    }

    #[test]
    fn all_blank_is_empty() {
        let lp = logits(&[2, 2, 2], 3);
        assert!(ctc_greedy_decode(&lp, 3, 3, 2).is_empty());
    }
}
