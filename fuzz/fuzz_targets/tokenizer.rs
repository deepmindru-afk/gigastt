//! Fuzz the BPE tokenizer decode path with arbitrary token-id sequences.
//!
//! Uses a synthetic in-memory vocab ([`Tokenizer::synthetic`], behind the
//! `__internals` feature) so no model is needed. Each pair of input bytes is
//! folded into a `usize` token id — deliberately allowed to exceed the vocab
//! size so the out-of-range / special-token skip paths are exercised. Also
//! probes `token_text` per id. Property: `decode` and `token_text` never
//! panic on arbitrary ids.
#![no_main]

use gigastt_core::inference::tokenizer::Tokenizer;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let tok = Tokenizer::synthetic();

    // Interpret the input as a stream of u16-derived token ids. Mixing in the
    // raw byte value occasionally pushes ids well past the synthetic vocab
    // size, covering the `id >= tokens.len()` branch.
    let ids: Vec<usize> = data
        .chunks(2)
        .map(|c| match c {
            [a, b] => ((*a as usize) << 8) | (*b as usize),
            [a] => *a as usize,
            _ => 0,
        })
        .collect();

    // token_text must tolerate any id, in-range or not.
    for &id in &ids {
        let _ = tok.token_text(id);
    }

    // decode joins token text, skips blank/unk/out-of-range, never panics.
    let _ = tok.decode(&ids);
});
