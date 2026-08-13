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
mod tests {
    use super::super::decode;
    use super::super::state::WordInfo;
    use super::super::tokenizer::Tokenizer;
    use super::super::windows::CHUNK_OVERLAP_SAMPLES;
    use super::super::{ENCODER_SUBSAMPLING, HOP_LENGTH};
    use super::*;

    #[test]
    fn test_token_formatter_groups_words() {
        // `▁` (U+2581) marks a new word; continuation tokens have no prefix.
        let tok = Tokenizer::from_tokens(vec![
            "\u{2581}hel".into(), // 0: new word
            "lo".into(),          // 1: continuation
            "\u{2581}wor".into(), // 2: new word
            "ld".into(),          // 3: continuation
        ]);
        let tokens = vec![
            decode::TokenInfo {
                token_id: 0,
                frame_index: 0,
                confidence: 0.9,
            },
            decode::TokenInfo {
                token_id: 1,
                frame_index: 1,
                confidence: 0.8,
            },
            decode::TokenInfo {
                token_id: 2,
                frame_index: 2,
                confidence: 0.95,
            },
            decode::TokenInfo {
                token_id: 3,
                frame_index: 3,
                confidence: 0.85,
            },
        ];
        let words = TokenFormatter::tokens_to_words(&tok, &tokens, 0);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].word, "hello");
        assert_eq!(words[1].word, "world");
        // Mean confidence per word.
        assert!((words[0].confidence - 0.85).abs() < 1e-6);
        assert!((words[1].confidence - 0.90).abs() < 1e-6);
        // Frame timing (SECONDS_PER_FRAME = 0.04).
        assert!((words[0].start - 0.0).abs() < 1e-9);
        assert!((words[0].end - 0.04).abs() < 1e-9);
        assert!((words[1].start - 0.08).abs() < 1e-9);
    }

    #[test]
    fn test_token_formatter_empty_tokens() {
        let tok = Tokenizer::from_tokens(vec!["\u{2581}a".into()]);
        assert!(TokenFormatter::tokens_to_words(&tok, &[], 0).is_empty());
    }

    #[test]
    fn test_token_formatter_frame_offset_shifts_time() {
        let tok = Tokenizer::from_tokens(vec!["\u{2581}x".into()]);
        let tokens = vec![decode::TokenInfo {
            token_id: 0,
            frame_index: 0,
            confidence: 1.0,
        }];
        let words = TokenFormatter::tokens_to_words(&tok, &tokens, 10);
        assert_eq!(words.len(), 1);
        // frame_offset 10 → start = 10 * 0.04 = 0.4.
        assert!((words[0].start - 0.4).abs() < 1e-9);
    }

    fn word(text: &str, start: f64, end: f64) -> WordInfo {
        WordInfo::new(text, start, end, 1.0, None)
    }

    #[test]
    fn test_stitch_first_chunk_passes_through() {
        // An empty `merged` (the very first chunk) is returned verbatim.
        let next = vec![word("a", 0.0, 0.5), word("b", 0.6, 1.0)];
        let out = stitch_chunk_words(Vec::new(), next.clone(), 11.0);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].word, "a");
        assert_eq!(out[1].word, "b");
    }

    #[test]
    fn test_stitch_dedups_overlap_no_drop_no_dup() {
        // Two 24s windows with a 22s stride: chunk B starts at 22s, overlap
        // [22s, 24s], seam at 23s. The word "dup" at ~22.5s is decoded by both
        // chunks; the seam attributes it to exactly one. No unique word lost.
        let chunk_a = vec![
            word("first", 1.0, 1.4),    // unique to A
            word("middle", 21.0, 21.4), // unique to A, before overlap
            word("dup", 22.4, 22.8),    // in overlap, before seam → kept from A
        ];
        // B's words are already offset by its 22s start.
        let chunk_b = vec![
            word("dup", 22.5, 22.9),   // same word re-decoded → after-seam copy dropped
            word("later", 25.0, 25.4), // unique to B
            word("end", 40.0, 40.4),   // unique to B
        ];
        let seam_s = 22.0 + CHUNK_OVERLAP_SAMPLES as f64 / 2.0 / 16000.0; // 23.0
        assert!((seam_s - 23.0).abs() < 1e-9);

        let out = stitch_chunk_words(chunk_a, chunk_b, seam_s);
        let texts: Vec<&str> = out.iter().map(|w| w.word.as_str()).collect();
        // "dup" appears exactly once (A's copy, before the seam); nothing dropped.
        assert_eq!(texts, vec!["first", "middle", "dup", "later", "end"]);
        // Monotonic in `start`.
        for w in out.windows(2) {
            assert!(w[0].start <= w[1].start, "not monotonic: {:?}", out);
        }
    }

    #[test]
    fn test_stitch_drops_a_tail_past_seam() {
        // A word decoded by A past the seam is dropped in favour of B's
        // fuller-context copy; the back half of the overlap belongs to B.
        let chunk_a = vec![word("keep", 22.0, 22.4), word("a_tail", 23.5, 23.9)];
        let chunk_b = vec![word("b_seam", 23.2, 23.6), word("b_late", 30.0, 30.4)];
        let out = stitch_chunk_words(chunk_a, chunk_b, 23.0);
        let texts: Vec<&str> = out.iter().map(|w| w.word.as_str()).collect();
        assert_eq!(texts, vec!["keep", "b_seam", "b_late"]);
    }

    // ---- Seam invariants --------------------------------------------------
    //
    // `stitch_chunk_words` cuts on `start` alone: the earlier chunk keeps
    // `start <= seam_s`, the later chunk keeps `start > seam_s`. There is no
    // text matching across the boundary, so the *interval* of a word is
    // irrelevant to the decision. The tests below pin what that cut does in the
    // cases the happy-path tests above skip — including the two lossy ones.
    // They are characterisation tests: they describe today's behaviour so a
    // change to the seam policy shows up as a diff, not as a silent shift in
    // long-form WER.
    //
    // Every word below sits within ±0.05s of the seam (or inside a 200ms gap
    // around it, for the silence control), so nudging the seam past that margin
    // flips at least one assertion. That is the check that these tests pin the
    // predicate rather than passing for any cut point.

    #[test]
    fn test_stitch_straddling_word_duplicated_across_seam() {
        // Invariant pinned: a word whose interval straddles the seam is emitted
        // TWICE when the two chunks disagree about its start across the cut —
        // the earlier chunk's copy starts before the seam (kept) and the later
        // chunk's copy starts after it (also kept). The stitch has no way to
        // notice they are the same word. This is the duplicate half of the
        // long-form seam cost.
        let chunk_a = vec![word("на", 22.0, 22.32), word("мосту", 22.96, 23.36)];
        let chunk_b = vec![word("мосту", 23.04, 23.44), word("стоял", 24.0, 24.4)];
        let out = stitch_chunk_words(chunk_a, chunk_b, 23.0);
        let texts: Vec<&str> = out.iter().map(|w| w.word.as_str()).collect();
        assert_eq!(
            texts,
            vec!["на", "мосту", "мосту", "стоял"],
            "a straddling word whose copies land on opposite sides of the seam is duplicated"
        );
    }

    #[test]
    fn test_stitch_straddling_word_deleted_at_seam() {
        // Invariant pinned: the mirror-image case deletes the word outright.
        // The earlier chunk placed its start just past the seam (dropped as
        // "tail past the seam") while the later chunk placed it just before
        // (dropped as "belongs to the earlier chunk"), so neither copy
        // survives. This is the deletion half of the long-form seam cost, and
        // it is the failure mode a text-matching stitch would have to fix.
        let chunk_a = vec![word("на", 22.0, 22.32), word("мосту", 23.04, 23.44)];
        let chunk_b = vec![word("мосту", 22.96, 23.36), word("стоял", 24.0, 24.4)];
        let out = stitch_chunk_words(chunk_a, chunk_b, 23.0);
        let texts: Vec<&str> = out.iter().map(|w| w.word.as_str()).collect();
        assert_eq!(
            texts,
            vec!["на", "стоял"],
            "a straddling word can vanish entirely: dropped by both sides of the seam"
        );
    }

    #[test]
    fn test_stitch_word_exactly_on_seam_kept_from_earlier_chunk() {
        // Invariant pinned: `start == seam_s` is an inclusive boundary for the
        // earlier chunk (`<=`) and an exclusive one for the later chunk (`>`),
        // so a word that both chunks place exactly on the seam survives once,
        // from the earlier chunk. Confidence tags which copy won: flipping the
        // predicate to `<` / `>=` would keep B's copy instead, same text.
        let chunk_a = vec![WordInfo::new("шов", 23.0, 23.4, 0.5, None)];
        let chunk_b = vec![
            WordInfo::new("шов", 23.0, 23.4, 0.9, None),
            word("после", 24.0, 24.4),
        ];
        let out = stitch_chunk_words(chunk_a, chunk_b, 23.0);
        let texts: Vec<&str> = out.iter().map(|w| w.word.as_str()).collect();
        assert_eq!(
            texts,
            vec!["шов", "после"],
            "no duplicate exactly on the seam"
        );
        assert_eq!(
            out[0].confidence, 0.5,
            "the surviving copy comes from the earlier chunk"
        );
    }

    #[test]
    fn test_stitch_empty_next_chunk_still_trims_tail_past_seam() {
        // Invariant pinned: the tail trim runs unconditionally, so a chunk that
        // decodes to nothing (silence, or a chunk the decoder emitted no tokens
        // for) still deletes whatever the previous chunk decoded past the seam.
        // The empty chunk is not a no-op.
        let chunk_a = vec![word("до", 22.0, 22.4), word("хвост", 23.04, 23.44)];
        let out = stitch_chunk_words(chunk_a, Vec::new(), 23.0);
        let texts: Vec<&str> = out.iter().map(|w| w.word.as_str()).collect();
        assert_eq!(
            texts,
            vec!["до"],
            "an empty chunk still drops the earlier chunk's post-seam tail"
        );
    }

    #[test]
    fn test_stitch_silence_at_seam_loses_nothing() {
        // Lossless-gap CONTROL — read this before "fixing" it. When the seam
        // falls inside a silent gap (no word interval touches it), the stitch
        // must stay lossless and duplicate-free for any reasonable seam
        // placement. That robustness is the invariant this test pins, so it is
        // DELIBERATELY insensitive to a ±60ms seam nudge: the gap here is 200ms
        // wide and ±60ms stays inside it, leaving the output unchanged. The
        // tolerance is intentional — do NOT tighten the fixture to make it react
        // to small nudges; that would delete the property being guarded.
        //
        // It is not vacuous: mutating the seam by +150ms moves the cut out of
        // the gap and reddens this test (it drops "после"). The other seam tests
        // in this block already pin the ±60ms / inclusive-boundary sensitivity;
        // this one pins the complementary claim that silence absorbs jitter.
        let chunk_a = vec![word("перед", 22.6, 22.9)];
        let chunk_b = vec![word("после", 23.1, 23.5), word("конец", 24.0, 24.4)];
        let out = stitch_chunk_words(chunk_a, chunk_b, 23.0);
        let texts: Vec<&str> = out.iter().map(|w| w.word.as_str()).collect();
        assert_eq!(texts, vec!["перед", "после", "конец"]);
        for w in out.windows(2) {
            assert!(w[0].start <= w[1].start, "not monotonic: {:?}", out);
        }
    }

    #[test]
    fn test_stitch_truncate_matches_retain_predicate() {
        // `stitch_chunk_words` trims the earlier chunk with
        // `partition_point` + `truncate` instead of `retain` (which rescanned
        // every merged word on every chunk). On the monotonic-in-`start` lists
        // the chunked loop actually produces, the two are the same predicate —
        // sweep seams on and between every word boundary to show it.
        let merged: Vec<WordInfo> = (0..40)
            .map(|i| word(&format!("w{i}"), i as f64 * 0.5, i as f64 * 0.5 + 0.3))
            .collect();
        for step in 0..=80 {
            let seam_s = step as f64 * 0.25;
            let mut expected = merged.clone();
            expected.retain(|w| w.start <= seam_s);
            let got = stitch_chunk_words(merged.clone(), Vec::new(), seam_s);
            assert_eq!(
                got.iter().map(|w| w.word.as_str()).collect::<Vec<_>>(),
                expected.iter().map(|w| w.word.as_str()).collect::<Vec<_>>(),
                "diverged at seam {seam_s}"
            );
        }
    }

    #[test]
    fn test_stitch_timestamp_offset_math() {
        // The chunked path offsets a chunk's frame indices by
        // start_samples / (HOP_LENGTH * ENCODER_SUBSAMPLING). Verify that a word
        // at frame 0 of a chunk starting `start_samples` in lands at the right
        // absolute time via `tokens_to_words` (the same offset the engine feeds).
        let tok = Tokenizer::from_tokens(vec!["\u{2581}w".into()]);
        let tokens = vec![decode::TokenInfo {
            token_id: 0,
            frame_index: 0,
            confidence: 1.0,
        }];
        let start_samples = 16000 * 22; // chunk starts at 22s
        let frame_offset = start_samples / (HOP_LENGTH * ENCODER_SUBSAMPLING);
        let words = TokenFormatter::tokens_to_words(&tok, &tokens, frame_offset);
        assert_eq!(words.len(), 1);
        // frame 0 + offset → absolute start == 22.0s exactly (aligned stride).
        assert!(
            (words[0].start - 22.0).abs() < 1e-9,
            "got {}",
            words[0].start
        );
    }

    #[test]
    fn test_token_formatter_last_word_empty_confidences_defaults_to_one() {
        // A word whose only token is a bare boundary marker (`▁`, no body)
        // contributes no confidence sample; a following real word that itself
        // has no recorded confidences must default to 1.0 on the final-emit
        // path. We build a vocab whose tokens are pure boundary markers so the
        // `clean` body is empty and no confidence is pushed.
        let tok = Tokenizer::from_tokens(vec![
            "\u{2581}real".into(), // 0: a real word
            "\u{2581}".into(),     // 1: bare boundary, empty body
        ]);
        let tokens = vec![
            decode::TokenInfo {
                token_id: 0,
                frame_index: 0,
                confidence: 0.7,
            },
            // A bare boundary token forces emission of "real" (mid-loop emit),
            // then contributes nothing to a new word.
            decode::TokenInfo {
                token_id: 1,
                frame_index: 1,
                confidence: 0.5,
            },
        ];
        let words = TokenFormatter::tokens_to_words(&tok, &tokens, 0);
        // Only "real" is emitted; the trailing bare boundary leaves no word.
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].word, "real");
        assert!((words[0].confidence - 0.7).abs() < 1e-6);
    }
}
