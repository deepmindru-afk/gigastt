//! Criterion micro-benchmark for the BPE tokenizer decode path.
//!
//! Uses a synthetic in-memory vocab (no model file). Requires the private
//! `__internals` feature for [`Tokenizer::synthetic`]. Run with
//! `cargo bench -p gigastt-core --features __internals`.

use criterion::{Criterion, criterion_group, criterion_main};
use gigastt_core::inference::tokenizer::Tokenizer;
use std::hint::black_box;

/// Deterministic pseudo-random token-id sequence within the vocab range.
fn synth_ids(vocab_size: usize, len: usize) -> Vec<usize> {
    let mut rng: u32 = 0xDEAD_BEEF;
    (0..len)
        .map(|_| {
            rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (rng as usize) % vocab_size
        })
        .collect()
}

fn bench_tokenizer(c: &mut Criterion) {
    let tok = Tokenizer::synthetic();
    let vocab = tok.vocab_size();
    let mut group = c.benchmark_group("tokenizer_decode");
    for &len in &[64_usize, 1024] {
        let ids = synth_ids(vocab, len);
        group.bench_function(format!("decode_{len}_ids"), |b| {
            b.iter(|| {
                let text = tok.decode(black_box(&ids));
                black_box(text);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_tokenizer);
criterion_main!(benches);
