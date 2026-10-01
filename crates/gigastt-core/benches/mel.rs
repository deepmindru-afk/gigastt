//! Criterion micro-benchmark for log-mel spectrogram feature extraction.
//!
//! Uses synthetic 16 kHz audio (sine + noise) — no model required. Run with
//! `cargo bench -p gigastt-core --no-default-features --features __internals --bench mel`.
//! Prints allocation counts before timing one-shot and warmed reusable buffers.

use criterion::{Criterion, criterion_group, criterion_main};
use gigastt_core::inference::FeatureExtractor;
use rustfft::{FftPlanner, num_complex::Complex};
use std::hint::black_box;

#[path = "support/allocations.rs"]
mod allocations;

#[global_allocator]
static ALLOCATOR: allocations::CountingAllocator = allocations::CountingAllocator;

fn check_allocations(extractor: &FeatureExtractor) {
    let fft = FftPlanner::<f32>::new().plan_fft_forward(320);
    let mut input = vec![Complex::new(0.0, 0.0); 320];
    let mut scratch = vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];
    let allocating = allocations::count(|| {
        for _ in 0..1_000 {
            fft.process(black_box(&mut input));
        }
    });
    let reused = allocations::count(|| {
        for _ in 0..1_000 {
            fft.process_with_scratch(black_box(&mut input), &mut scratch);
        }
    });
    assert_eq!(reused, 0, "reused FFT scratch allocated");
    eprintln!(
        "FFT=320, 1000 calls: allocating={allocating}, reused={reused}, scratch={}",
        scratch.len()
    );

    let samples = synth_audio(5.0);
    let (mut fft_buf, mut power, mut output) = (Vec::new(), Vec::new(), Vec::new());
    extractor.compute_mel(&samples, &mut fft_buf, &mut power, &mut output);
    let mel_allocations = allocations::count(|| {
        for _ in 0..10 {
            black_box(extractor.compute_mel(
                black_box(&samples),
                &mut fft_buf,
                &mut power,
                &mut output,
            ));
        }
    });
    assert_eq!(mel_allocations, 0, "warmed mel buffers allocated");
    eprintln!("5-second mel, 10 warmed calls: allocations={mel_allocations}");
}

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
    check_allocations(&extractor);
    let mut group = c.benchmark_group("mel_spectrogram");
    for &secs in &[1.0_f32, 5.0] {
        let samples = synth_audio(secs);
        group.bench_function(format!("{secs}s_16khz"), |b| {
            b.iter(|| {
                let (features, frames) = extractor.compute(black_box(&samples));
                black_box((features, frames));
            });
        });
        let (mut fft_buf, mut power, mut output) = (Vec::new(), Vec::new(), Vec::new());
        extractor.compute_mel(&samples, &mut fft_buf, &mut power, &mut output);
        group.bench_function(format!("{secs}s_16khz_reused"), |b| {
            b.iter(|| {
                black_box(extractor.compute_mel(
                    black_box(&samples),
                    &mut fft_buf,
                    &mut power,
                    &mut output,
                ));
                black_box(&output);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_mel);
criterion_main!(benches);
