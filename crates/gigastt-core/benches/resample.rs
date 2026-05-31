//! Criterion micro-benchmark for the rubato polyphase resampler.
//!
//! Synthetic input only (sine + noise) — no model required. Run with
//! `cargo bench -p gigastt-core --features __internals`.

use criterion::{Criterion, criterion_group, criterion_main};
use gigastt_core::inference::audio::{SampleRate, resample};
use std::hint::black_box;

/// Deterministic synthetic mono buffer at `sample_rate` Hz: sine + LCG noise.
fn synth_audio(sample_rate: u32, seconds: f32) -> Vec<f32> {
    let sr = sample_rate as f32;
    let n = (sr * seconds) as usize;
    let mut rng: u32 = 0x9E37_79B9;
    (0..n)
        .map(|i| {
            rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (rng >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
            let t = i as f32 / sr;
            0.8 * (2.0 * std::f32::consts::PI * 440.0 * t).sin() + 0.2 * noise
        })
        .collect()
}

fn bench_resample(c: &mut Criterion) {
    let mut group = c.benchmark_group("resample");
    // 48 kHz → 16 kHz is the default downsample path for WebSocket/file audio.
    let input_48k = synth_audio(48_000, 1.0);
    group.bench_function("48k_to_16k_1s", |b| {
        b.iter(|| {
            let out = resample(
                black_box(&input_48k),
                SampleRate(48_000),
                SampleRate(16_000),
            )
            .expect("resample 48k->16k");
            black_box(out);
        });
    });
    // 8 kHz → 16 kHz upsample path.
    let input_8k = synth_audio(8_000, 1.0);
    group.bench_function("8k_to_16k_1s", |b| {
        b.iter(|| {
            let out = resample(black_box(&input_8k), SampleRate(8_000), SampleRate(16_000))
                .expect("resample 8k->16k");
            black_box(out);
        });
    });
    group.finish();
}

criterion_group!(benches, bench_resample);
criterion_main!(benches);
