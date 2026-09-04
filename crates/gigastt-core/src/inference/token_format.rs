//! Token → word formatting and long-form chunk stitch.
//!
//! Split out of [`super::engine`] so formatting and stitch policy are
//! unit-testable without a loaded model — they depend only on the
//! [`Tokenizer`] / [`WordInfo`], not on any ONNX session.

use super::SECONDS_PER_FRAME;
use super::decode;
use super::state::WordInfo;
use super::tokenizer::{self, Tokenizer};

/// Merge a later chunk's words into the running `merged` list, de-duplicating
/// the overlap region around `seam_s` (absolute seconds).
///
/// Both lists carry absolute timestamps (each chunk's words were already offset
/// by the chunk start). The heuristic keeps `merged` words whose start is at or
/// before the seam and `next` words whose start is strictly after the seam, so
/// the ~2s overlap is attributed to exactly one chunk: the earlier chunk owns
/// the front half of the overlap, the later chunk owns the back half. A word
/// straddling the seam is decoded with full context in at least one chunk, so
/// no unique word is dropped and no overlap word is emitted twice in the common
/// case. The merged list is monotonic in `start` by construction (the earlier
/// chunk's kept words all start ≤ seam < the later chunk's kept words).
///
/// Pure and free-standing so the stitch policy is unit-testable without a
/// loaded model.
pub(crate) fn stitch_chunk_words(
    mut merged: Vec<WordInfo>,
    next: Vec<WordInfo>,
    seam_s: f64,
) -> Vec<WordInfo> {
    if merged.is_empty() {
        return next;
    }
    // Drop the earlier chunk's tail that reaches past the seam — those words are
    // re-decoded by `next` with more right context, so prefer the later chunk
    // for the back half of the overlap. `merged` is monotonic in `start`, so the
    // words to drop are exactly a suffix: binary-search the seam and truncate.
    // (`retain` would rescan every word merged so far on every chunk — O(chunks
    // × words) — for a policy that only ever trims the tail.)
    merged.truncate(merged.partition_point(|w| w.start <= seam_s));
    merged.extend(next.into_iter().filter(|w| w.start > seam_s));
    merged
}

/// Absolute-seconds midpoint of the overlap between this window and the
/// previous one. `start_sample` is the window's 16 kHz origin; `overlap_samples`
/// is the shared tail/head (2 s = 32_000 at the file-chunk geometry).
pub(crate) fn overlap_mid_seconds(start_sample: usize, overlap_samples: usize) -> f64 {
    (start_sample as f64 + overlap_samples as f64 / 2.0) / 16_000.0
}

/// Groups RNN-T decoded tokens into words at BPE word boundaries (`▁`).
///
/// Split out of `Engine` so the formatting logic is unit-testable without a
/// loaded model — it depends only on the [`Tokenizer`], not on any ONNX
/// session.
pub(crate) struct TokenFormatter;

impl TokenFormatter {
    /// Group `tokens` into words. `frame_offset` shifts per-token frame indices
    /// into absolute stream time; word confidence is the mean over the word's
    /// constituent BPE tokens.
    pub(crate) fn tokens_to_words(
        tokenizer: &Tokenizer,
        tokens: &[decode::TokenInfo],
        frame_offset: usize,
    ) -> Vec<WordInfo> {
        if tokens.is_empty() {
            return Vec::new();
        }

        // Group tokens by words (BPE ▁ marks word boundaries)
        let mut words = Vec::new();
        let mut current_word = String::new();
        let mut word_start_frame: Option<usize> = None;
        let mut word_end_frame: usize = 0;
        let mut word_confidences: Vec<f32> = Vec::new();

        for token in tokens {
            let token_text = tokenizer.token_text(token.token_id);
            let is_word_boundary = token_text.starts_with(tokenizer::WORD_BOUNDARY);

            if is_word_boundary && !current_word.is_empty() {
                // Emit previous word
                let avg_conf: f32 = if word_confidences.is_empty() {
                    1.0
                } else {
                    word_confidences.iter().sum::<f32>() / word_confidences.len() as f32
                };
                words.push(WordInfo {
                    word: std::mem::take(&mut current_word),
                    start: (word_start_frame.unwrap_or(0) + frame_offset) as f64
                        * SECONDS_PER_FRAME,
                    end: (word_end_frame + frame_offset) as f64 * SECONDS_PER_FRAME,
                    confidence: avg_conf,
                    speaker: None,
                });
                current_word.clear();
                word_confidences.clear();
                word_start_frame = None;
            }

            let clean = if let Some(stripped) = token_text.strip_prefix(tokenizer::WORD_BOUNDARY) {
                stripped
            } else {
                token_text
            };
            if !clean.is_empty() {
                current_word.push_str(clean);
                if word_start_frame.is_none() {
                    word_start_frame = Some(token.frame_index);
                }
                word_end_frame = token.frame_index;
                word_confidences.push(token.confidence);
            }
        }

        // Emit last word
        if !current_word.is_empty() {
            let avg_conf: f32 = if word_confidences.is_empty() {
                1.0
            } else {
                word_confidences.iter().sum::<f32>() / word_confidences.len() as f32
            };
            words.push(WordInfo {
                word: current_word,
                start: (word_start_frame.unwrap_or(0) + frame_offset) as f64 * SECONDS_PER_FRAME,
                end: (word_end_frame + frame_offset) as f64 * SECONDS_PER_FRAME,
                confidence: avg_conf,
                speaker: None,
            });
        }

        words
    }
}

#[cfg(test)]
mod tests;
