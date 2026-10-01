//! Synthetic arithmetic comparison for deferred token confidence normalization.
//! Run with: rustc -O scripts/bench-confidence-normalization.rs -o /tmp/confidence-bench
//!           /tmp/confidence-bench
//!
//! Uses 1,000 rows, 80% blanks, 100 repetitions and 10 alternating samples.
//! Compares the same argmax/softmax arithmetic before and after emission checks;
//! this is not an inference benchmark and cannot establish end-to-end speedup.
//! Black-boxing the eager score prevents optimizing away the baseline work.

use std::{hint::black_box, time::Instant};
fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0
}
fn confidence(row: &[f32], token: usize) -> f32 {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = row.iter().map(|l| (l - max).exp()).sum();
    (row[token] - max).exp() / sum
}
fn run(rows: &[Vec<f32>], deferred: bool, ctc: bool) -> f32 {
    let mut sum = 0.;
    for _ in 0..100 {
        let mut prev = None;
        for row in rows {
            let row = black_box(row);
            let token = argmax(row);
            let score = if deferred {
                0.
            } else {
                black_box(confidence(row, token))
            };
            let repeat = prev == Some(token);
            prev = Some(token);
            if token == row.len() - 1 || (ctc && repeat) {
                continue;
            }
            sum += if deferred {
                confidence(row, token)
            } else {
                score
            };
        }
    }
    black_box(sum)
}
fn main() {
    for (vocab, ctc) in [(71, true), (34, false), (1025, false)] {
        let rows: Vec<_> = (0..1000)
            .map(|i| {
                let mut r: Vec<f32> = (0..vocab)
                    .map(|j| ((i * 17 + j * 13) % 101) as f32 / 101.)
                    .collect();
                r[if i % 5 == 0 {
                    i % (vocab - 1)
                } else {
                    vocab - 1
                }] = 3.;
                r
            })
            .collect();
        assert_eq!(
            run(&rows, false, ctc).to_bits(),
            run(&rows, true, ctc).to_bits()
        );
        let mut times = [Vec::new(), Vec::new()];
        for round in 0..10 {
            for k in 0..2 {
                let mode = (round + k) % 2;
                let start = Instant::now();
                black_box(run(&rows, mode == 1, ctc));
                times[mode].push(start.elapsed().as_secs_f64() * 1000.);
            }
        }
        for t in &mut times {
            t.sort_by(f64::total_cmp);
        }
        println!(
            "vocab={vocab} ctc={ctc} baseline_ms={:.3} deferred_ms={:.3}",
            (times[0][4] + times[0][5]) / 2.,
            (times[1][4] + times[1][5]) / 2.
        );
    }
}
