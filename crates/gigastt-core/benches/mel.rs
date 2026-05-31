//! Criterion micro-benchmark for log-mel spectrogram feature extraction.
//!
//! Uses synthetic 16 kHz audio (sine + noise) — no model required. Run with
//! `cargo bench -p gigastt-core --features __internals`.

use criterion::{Criterion, criterion_group, criterion_main};
use gigastt_core::inference::FeatureExtractor;
use std::hint::black_box;

/// Deterministic synthetic 16 kHz mono buffer: a 440 Hz sine mixed with a
/// cheap LCG pseudo-noise so the FFT sees broadband energy across mel bins.
fn synth_audio(seconds: f32) -> Vec<f32> {
    let sample_rate = 16_000.0_f32;
    let n = (sample_rate * seconds) as usize;
    let mut rng: u32 = 0x1234_5678;
    (0..n)
        .map(|i| {
            // LCG (Numerical Recipes constants) → [-0.1, 0.1] noise floor.
            rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (rng >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
            let t = i as f32 / sample_rate;
            0.8 * (2.0 * std::f32::consts::PI * 440.0 * t).sin() + 0.2 * noise
        })
        .collect()
}

fn bench_mel(c: &mut Criterion) {
    let extractor = FeatureExtractor::new();
    let mut group = c.benchmark_group("mel_spectrogram");
    for &secs in &[1.0_f32, 5.0] {
        let samples = synth_audio(secs);
        group.bench_function(format!("{secs}s_16khz"), |b| {
            b.iter(|| {
                let (features, frames) = extractor.compute(black_box(&samples));
                black_box((features, frames));
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_mel);
criterion_main!(benches);
