use super::*;

fn biaser(seqs: Vec<Vec<usize>>, boost: f32) -> Biaser {
    Biaser::from_sequences(seqs, boost).expect("non-empty sequences")
}

#[test]
fn test_from_sequences_empty_returns_none() {
    assert!(Biaser::from_sequences(vec![], 5.0).is_none());
    assert!(Biaser::from_sequences(vec![vec![]], 5.0).is_none());
}

#[test]
fn test_boost_applies_to_first_token_of_each_hotword() {
    // Two hotwords: [1,2] and [3]. At the root both 1 and 3 are boostable.
    // The greedy path pays the configured boost per step, whatever the
    // phrase's length — see `boost_logits`.
    let b = biaser(vec![vec![1, 2], vec![3]], 5.0);
    let state = b.new_state();
    let mut logits = vec![0.0; 5];
    b.boost_logits(&state, &mut logits);
    assert_eq!(logits[1], 5.0);
    assert_eq!(logits[3], 5.0);
    assert_eq!(logits[2], 0.0, "mid-hotword token not boosted at root");
    assert_eq!(logits[0], 0.0);
}

#[test]
fn test_advance_then_boost_continuation() {
    // After emitting token 1, the hotword [1,2] should boost token 2.
    let b = biaser(vec![vec![1, 2]], 5.0);
    let mut state = b.new_state();
    b.advance(&mut state, 1);
    let mut logits = vec![0.0; 5];
    b.boost_logits(&state, &mut logits);
    assert_eq!(logits[2], 5.0, "continuation token boosted after prefix");
    // Token 1 is also boostable again because the root stays active.
    assert_eq!(logits[1], 5.0, "root keeps a fresh hotword start available");
}

#[test]
fn test_advance_off_prefix_resets_to_root_only() {
    // Emit a non-matching token: only the root-level starts remain boosted.
    let b = biaser(vec![vec![1, 2]], 5.0);
    let mut state = b.new_state();
    b.advance(&mut state, 1); // on prefix [1]
    b.advance(&mut state, 9); // off prefix → reset to root
    let mut logits = vec![0.0; 5];
    b.boost_logits(&state, &mut logits);
    assert_eq!(logits[2], 0.0, "continuation no longer boosted after reset");
    assert_eq!(logits[1], 5.0, "root start still boosted");
}

#[test]
fn test_shared_prefix_keeps_both_branches_active() {
    // Hotwords [1,2] and [1,3] share the first token.
    let b = biaser(vec![vec![1, 2], vec![1, 3]], 4.0);
    let mut state = b.new_state();
    b.advance(&mut state, 1);
    let mut logits = vec![0.0; 5];
    b.boost_logits(&state, &mut logits);
    assert_eq!(logits[2], 4.0);
    assert_eq!(logits[3], 4.0);
}

#[test]
fn test_boost_ignores_out_of_range_token_id() {
    // A hotword token id beyond the logits length must not panic.
    let b = biaser(vec![vec![99]], 5.0);
    let state = b.new_state();
    let mut logits = vec![0.0; 5];
    b.boost_logits(&state, &mut logits); // no panic
    assert!(logits.iter().all(|&l| l == 0.0));
}

use crate::inference::tokenizer::Tokenizer;

/// A char-vocab tokenizer covering the Cyrillic letters used below plus a
/// `▁` word-boundary marker, so `encode_phrase` produces deterministic ids.
fn char_tokenizer() -> Tokenizer {
    let tokens = vec![
        "а".to_string(),
        "б".to_string(),
        "в".to_string(),
        "г".to_string(),
        "д".to_string(),
        "\u{2581}".to_string(), // word-boundary marker
        "<unk>".to_string(),
        "<blk>".to_string(),
    ];
    Tokenizer::from_tokens(tokens)
}

#[test]
fn the_phrase_entry_marker_is_never_boosted() {
    // Every phrase `encode_phrase` produces starts with the same
    // word-boundary marker, so paying for it means paying at every word
    // boundary in the audio whatever the glossary says. That is exactly
    // what happened: three different glossaries — one of them a word absent
    // from the recording — produced byte-identical transcripts, because the
    // only token being boosted was the space.
    let tok = char_tokenizer();
    let marker = tok.encode_phrase("а").expect("representable")[0];
    for phrases in [
        vec![("а".to_string(), 1.0)],
        vec![("аб".to_string(), 1.0)],
        vec![("вг".to_string(), 1.0), ("д".to_string(), 1.0)],
    ] {
        let b = Biaser::from_phrases(&tok, &phrases, 6.0).expect("compiles");
        let state = b.new_state();
        let mut logits = vec![0.0; 8];
        b.boost_logits(&state, &mut logits);
        assert_eq!(
            logits[marker], 0.0,
            "the entry marker took a boost for {phrases:?}"
        );
        assert!(
            logits.iter().all(|&l| l == 0.0),
            "nothing is boostable before a word boundary is emitted"
        );
    }
}

#[test]
fn a_phrase_earns_one_boost_however_long_it_is() {
    // Paid per token, a nine-letter phrase would be worth nine boosts and
    // outrank whatever was actually said — that is how `любовницы`
    // displaced `люк кейдж` on real audio. Length must not buy rank.
    for len in [1usize, 3, 9] {
        let seq: Vec<usize> = (1..=len).collect();
        let b = Biaser::from_sequences(vec![seq.clone()], 6.0).expect("biaser");
        let mut path = BiasPath::default();
        let mut earned = 0.0;
        for &tok in &seq {
            let (delta, next) = b.score_token(path, tok);
            earned += delta;
            path = next;
        }
        assert!(
            (earned - 6.0).abs() < 1e-4,
            "a {len}-token phrase earned {earned}, expected the boost once"
        );
        assert_eq!(path.pending(), 0.0, "a finished phrase owes nothing back");
    }
}

#[test]
fn an_abandoned_phrase_earns_nothing() {
    // Credit accrues while a phrase is under way so the beam keeps it
    // alive; walking away has to give all of it back, or a partial match
    // would be rewarded for going nowhere.
    let b = Biaser::from_sequences(vec![vec![1, 2, 3, 4]], 6.0).expect("biaser");
    let mut path = BiasPath::default();
    let mut earned = 0.0;
    for tok in [1usize, 2] {
        let (delta, next) = b.score_token(path, tok);
        earned += delta;
        path = next;
    }
    assert!(path.pending() > 0.0, "mid-phrase credit is outstanding");
    let (delta, path) = b.score_token(path, 99);
    earned += delta;
    assert!(earned.abs() < 1e-4, "abandoned phrase netted {earned}");
    assert_eq!(path.pending(), 0.0);
}

#[test]
fn test_from_phrases_zero_boost_returns_none() {
    let tok = char_tokenizer();
    let phrases = vec![("аб".to_string(), 1.0)];
    assert!(Biaser::from_phrases(&tok, &phrases, 0.0).is_none());
}

#[test]
fn test_from_phrases_negative_boost_returns_none() {
    let tok = char_tokenizer();
    let phrases = vec![("аб".to_string(), 1.0)];
    assert!(Biaser::from_phrases(&tok, &phrases, -3.0).is_none());
}

#[test]
fn test_from_phrases_empty_slice_returns_none() {
    let tok = char_tokenizer();
    assert!(Biaser::from_phrases(&tok, &[], 5.0).is_none());
}

#[test]
fn test_from_phrases_all_zero_weight_returns_none() {
    // Positive boost but every phrase filtered out by weight <= 0 → None.
    let tok = char_tokenizer();
    let phrases = vec![("аб".to_string(), 0.0), ("вг".to_string(), -1.0)];
    assert!(Biaser::from_phrases(&tok, &phrases, 5.0).is_none());
}

#[test]
fn test_from_phrases_unrepresentable_only_returns_none() {
    // A phrase with no codepoints in the vocab is dropped; nothing survives.
    let tok = char_tokenizer();
    let phrases = vec![("xyz".to_string(), 1.0)];
    assert!(Biaser::from_phrases(&tok, &phrases, 5.0).is_none());
}

#[test]
fn test_from_phrases_single_token_phrase_boosts_first_token() {
    // "а" encodes to a leading ▁ (id 5) then char id 0. The marker only
    // opens the phrase; the character is what gets paid for.
    let tok = char_tokenizer();
    let phrases = vec![("а".to_string(), 1.0)];
    let b = Biaser::from_phrases(&tok, &phrases, 7.0).expect("phrase compiles");
    assert_eq!(b.phrase_count(), 1);

    let ids = tok.encode_phrase("а").expect("representable");
    let mut state = b.new_state();
    let mut logits = vec![0.0; 8];
    b.boost_logits(&state, &mut logits);
    assert_eq!(logits[ids[0]], 0.0, "the boundary marker is never boosted");

    b.advance(&mut state, ids[0]);
    let mut logits = vec![0.0; 8];
    b.boost_logits(&state, &mut logits);
    assert_eq!(
        logits[ids[1]], 7.0,
        "a one-character phrase is worth it all"
    );
}

#[test]
fn test_from_phrases_multi_token_phrase_boosts_continuation() {
    // "аб" → [▁(5), а(0), б(1)]. After advancing through the encoded
    // prefix, the next continuation token must be boosted.
    let tok = char_tokenizer();
    let phrases = vec![("аб".to_string(), 1.0)];
    let b = Biaser::from_phrases(&tok, &phrases, 4.0).expect("phrase compiles");
    assert_eq!(b.phrase_count(), 1);

    let ids = tok.encode_phrase("аб").expect("representable");
    assert_eq!(ids, vec![5, 0, 1]);

    let mut state = b.new_state();
    b.advance(&mut state, ids[0]); // ▁
    b.advance(&mut state, ids[1]); // а
    let mut logits = vec![0.0; 8];
    b.boost_logits(&state, &mut logits);
    assert_eq!(
        logits[ids[2]], 4.0,
        "third token boosted after two-token prefix"
    );
}

#[test]
fn test_from_phrases_drops_unrepresentable_keeps_representable() {
    // One good phrase, one with an out-of-vocab codepoint. The good one
    // survives; the count reflects only the compiled phrase.
    let tok = char_tokenizer();
    let phrases = vec![("аб".to_string(), 1.0), ("аz".to_string(), 1.0)];
    let b = Biaser::from_phrases(&tok, &phrases, 5.0).expect("one phrase compiles");
    assert_eq!(b.phrase_count(), 1);
}

/// A lowercase Cyrillic char vocab without `ё`, the shape of the `rnnt`
/// head's 34-token vocabulary.
fn cyrillic_tokenizer() -> Tokenizer {
    let tokens = [
        "г", "и", "а", "э", "м", "п", "т", "р", "е", "\u{2581}", "<unk>", "<blk>",
    ];
    Tokenizer::from_tokens(tokens.iter().map(|t| (*t).to_string()).collect())
}

#[test]
fn test_from_phrases_capitalized_phrase_is_kept() {
    // Users write brands capitalized; every shipped vocab is lowercase, so
    // the written form was dropped outright and glossaries silently lost
    // most of their entries.
    let tok = cyrillic_tokenizer();
    let phrases = vec![("Гигаэм".to_string(), 1.0)];
    let b = Biaser::from_phrases(&tok, &phrases, 5.0).expect("capitalized phrase compiles");
    assert_eq!(b.phrase_count(), 1);
    // It compiles to exactly the lowercase spelling's tokens.
    let ids = tok.encode_phrase("гигаэм").expect("representable");
    let mut state = b.new_state();
    b.advance(&mut state, ids[0]); // past the boundary marker
    let mut logits = vec![0.0; 12];
    b.boost_logits(&state, &mut logits);
    assert!(
        logits[ids[1]] > 0.0,
        "the lowercased spelling is what biases"
    );
}

#[test]
fn test_from_phrases_yo_folds_to_e_when_the_vocab_lacks_it() {
    // A head with no `ё` token emits `е` in its place, so that is the
    // spelling a hotword has to match.
    let tok = cyrillic_tokenizer();
    assert!(tok.encode_phrase("пётр").is_none(), "vocab has no ё");
    let phrases = vec![("Пётр".to_string(), 1.0)];
    let b = Biaser::from_phrases(&tok, &phrases, 5.0).expect("ё folds to е");
    assert_eq!(b.phrase_count(), 1);
}

#[test]
fn test_from_phrases_cased_vocab_keeps_the_written_spelling() {
    // The `e2e_rnnt` BPE vocab carries case. Lowercasing unconditionally
    // would re-tokenize — and worsen — phrases that already fit.
    let tokens = ["Аб", "а", "б", "\u{2581}", "<unk>", "<blk>"];
    let tok = Tokenizer::from_tokens(tokens.iter().map(|t| (*t).to_string()).collect());
    let written = tok.encode_phrase("Аб").expect("cased vocab represents it");
    let b = Biaser::from_phrases(&tok, &[("Аб".to_string(), 1.0)], 5.0).expect("compiles");
    let mut state = b.new_state();
    b.advance(&mut state, written[0]); // past the boundary marker
    let mut logits = vec![0.0; 6];
    b.boost_logits(&state, &mut logits);
    assert_eq!(
        logits[written[1]], 5.0,
        "the written form must survive untouched"
    );
}

#[test]
fn test_from_phrases_latin_stays_unrepresentable_on_a_cyrillic_vocab() {
    // The honest ceiling: no amount of case folding puts Latin letters in a
    // Cyrillic-only vocabulary, so `ChatGPT` is still dropped — and the
    // warning names it so the user can write it phonetically instead.
    let tok = cyrillic_tokenizer();
    let phrases = vec![("ChatGPT".to_string(), 1.0), ("Гигаэм".to_string(), 1.0)];
    let b = Biaser::from_phrases(&tok, &phrases, 5.0).expect("one phrase compiles");
    assert_eq!(b.phrase_count(), 1);
}

#[test]
fn test_from_phrases_weight_filters_per_phrase() {
    // Two representable phrases; one has weight 0 and is dropped before
    // tokenization, leaving a single compiled phrase.
    let tok = char_tokenizer();
    let phrases = vec![("аб".to_string(), 1.0), ("вг".to_string(), 0.0)];
    let b = Biaser::from_phrases(&tok, &phrases, 5.0).expect("one phrase compiles");
    assert_eq!(b.phrase_count(), 1);
}
