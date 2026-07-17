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
use super::tokenizer::{Tokenizer, WORD_BOUNDARY};
use super::{SECONDS_PER_FRAME, WordInfo};

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

/// Group CTC-decoded tokens into words with timestamps and confidence.
///
/// istupakov's CTC vocab encodes the inter-word space as the `▁` word-boundary
/// marker at `vocab[0]` (mirroring the RNN-T vocab), NOT a literal `' '`. Words
/// split on that marker; every other token is a single character concatenated
/// into the current word. The blank token never appears here (it is dropped in
/// [`ctc_greedy_decode`]).
pub(crate) fn ctc_tokens_to_words(
    tokenizer: &Tokenizer,
    tokens: &[TokenInfo],
    frame_offset: usize,
) -> Vec<WordInfo> {
    let mut words = Vec::new();
    let mut current_word = String::new();
    let mut word_start_frame: Option<usize> = None;
    let mut word_end_frame: usize = 0;
    let mut word_confidences: Vec<f32> = Vec::new();

    let flush = |word: &mut String,
                 start: &mut Option<usize>,
                 end: usize,
                 confs: &mut Vec<f32>,
                 out: &mut Vec<WordInfo>| {
        if word.is_empty() {
            return;
        }
        let avg_conf: f32 = if confs.is_empty() {
            1.0
        } else {
            confs.iter().sum::<f32>() / confs.len() as f32
        };
        out.push(WordInfo {
            word: std::mem::take(word),
            start: (start.unwrap_or(0) + frame_offset) as f64 * SECONDS_PER_FRAME,
            end: (end + frame_offset) as f64 * SECONDS_PER_FRAME,
            confidence: avg_conf,
            speaker: None,
        });
        *start = None;
        confs.clear();
    };

    for token in tokens {
        let ch = tokenizer.token_text(token.token_id);
        if ch.starts_with(WORD_BOUNDARY) {
            flush(
                &mut current_word,
                &mut word_start_frame,
                word_end_frame,
                &mut word_confidences,
                &mut words,
            );
            continue;
        }
        if !ch.is_empty() {
            current_word.push_str(ch);
            if word_start_frame.is_none() {
                word_start_frame = Some(token.frame_index);
            }
            word_end_frame = token.frame_index;
            word_confidences.push(token.confidence);
        }
    }

    flush(
        &mut current_word,
        &mut word_start_frame,
        word_end_frame,
        &mut word_confidences,
        &mut words,
    );

    words
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

    /// Build a CTC-style char tokenizer whose vocab[0] is the `▁` word-boundary
    /// marker (matching istupakov's `multilingual_vocab.txt`), followed by the
    /// letters used below and a trailing `<blk>`.
    fn ctc_tokenizer(letters: &[&str]) -> Tokenizer {
        let mut toks = vec!["\u{2581}".to_string()];
        toks.extend(letters.iter().map(|s| s.to_string()));
        toks.push("<blk>".to_string());
        Tokenizer::from_tokens(toks)
    }

    fn tok(id: usize, frame: usize) -> TokenInfo {
        TokenInfo {
            token_id: id,
            frame_index: frame,
            confidence: 1.0,
        }
    }

    #[test]
    fn groups_words_on_boundary_marker() {
        // vocab: 0=▁, 1..=6 = п р и в е т, 7..=9 = м и р, 10=<blk>
        let t = ctc_tokenizer(&["п", "р", "и", "в", "е", "т", "м", "и", "р"]);
        // "привет мир": п р и в е т ▁ м и р
        let toks = [
            tok(1, 0),
            tok(2, 1),
            tok(3, 2),
            tok(4, 3),
            tok(5, 4),
            tok(6, 5),
            tok(0, 6), // ▁ separator
            tok(7, 7),
            tok(8, 8),
            tok(9, 9),
        ];
        let words = ctc_tokens_to_words(&t, &toks, 0);
        assert_eq!(
            words.iter().map(|w| w.word.as_str()).collect::<Vec<_>>(),
            vec!["привет", "мир"]
        );
        // Timestamps come from the first/last frame of each word.
        assert!((words[0].start - 0.0).abs() < 1e-9);
        assert!(words[1].start > words[0].end);
    }

    #[test]
    fn leading_and_trailing_boundaries_emit_no_empty_words() {
        let t = ctc_tokenizer(&["а", "б"]);
        // ▁ а б ▁  → one word "аб", no empty words from the edge separators.
        let toks = [tok(0, 0), tok(1, 1), tok(2, 2), tok(0, 3)];
        let words = ctc_tokens_to_words(&t, &toks, 0);
        assert_eq!(
            words.iter().map(|w| w.word.as_str()).collect::<Vec<_>>(),
            vec!["аб"]
        );
    }
}
