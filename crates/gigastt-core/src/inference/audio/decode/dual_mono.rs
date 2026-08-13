//! Channel mix and dual-mono detection.

use super::super::DUAL_MONO_CORRELATION_THRESHOLD;

/// Average multiple channels into a single mono vector.
pub fn mix_channels_to_mono(channels: &[Vec<f32>]) -> Vec<f32> {
    if channels.is_empty() {
        return Vec::new();
    }
    let n = channels.iter().map(|c| c.len()).min().unwrap_or(0);
    (0..n)
        .map(|i| channels.iter().map(|c| c[i]).sum::<f32>() / channels.len() as f32)
        .collect()
}

/// Return `true` if a two-channel stream is dual-mono (both channels nearly
/// identical). Empty or single-channel input returns `false`.
pub fn is_dual_mono(channels: &[Vec<f32>]) -> bool {
    if channels.len() != 2 {
        return false;
    }
    let (left, right) = (&channels[0], &channels[1]);
    if left.is_empty() || right.is_empty() {
        return false;
    }
    let len = left.len().min(right.len());
    normalized_correlation(&left[..len], &right[..len]) > DUAL_MONO_CORRELATION_THRESHOLD
}

/// Streaming form of the dual-mono correlation, for the `channels=split` path
/// over a container that is never fully resident.
///
/// [`is_dual_mono`] needs both channels end to end, which is the only reason
/// channel splitting had to hold the whole decode. The same statistic is
/// computable in one pass with Welford's co-moment recurrence, so this keeps
/// six `f64` accumulators instead of two whole channels.
///
/// The recurrence is used rather than raw power sums (`Σab − ΣaΣb/n`) on
/// purpose: the naive form loses the covariance to cancellation once `n` is in
/// the tens of millions, exactly the regime long files put it in, and the
/// decision it feeds is a knife-edge threshold. Agreement with the batch
/// function is asserted in `tests.rs`, on the value and on the decision.
#[derive(Debug, Default, Clone)]
pub struct DualMonoDetector {
    n: f64,
    mean_a: f64,
    mean_b: f64,
    /// Co-moment Σ(a−ā)(b−b̄).
    c_ab: f64,
    m2_a: f64,
    m2_b: f64,
}

impl DualMonoDetector {
    /// New detector with no samples observed.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed the next block of two channels. Only the overlapping prefix counts,
    /// matching [`is_dual_mono`]'s `min(len)` truncation.
    pub fn push(&mut self, a: &[f32], b: &[f32]) {
        for (&x, &y) in a.iter().zip(b) {
            let (x, y) = (x as f64, y as f64);
            self.n += 1.0;
            let dx = x - self.mean_a;
            let dy = y - self.mean_b;
            self.mean_a += dx / self.n;
            self.mean_b += dy / self.n;
            self.c_ab += dx * (y - self.mean_b);
            self.m2_a += dx * (x - self.mean_a);
            self.m2_b += dy * (y - self.mean_b);
        }
    }

    /// Normalized correlation of everything observed so far, on the same scale
    /// [`is_dual_mono`] thresholds. `0.0` when either channel is silent.
    pub fn correlation(&self) -> f64 {
        if self.n == 0.0 {
            return 0.0;
        }
        let denom = self.m2_a.sqrt() * self.m2_b.sqrt();
        if denom < 1e-12 {
            return 0.0;
        }
        self.c_ab / denom
    }

    /// Whether the two channels are near-identical — the same verdict
    /// [`is_dual_mono`] returns for a fully materialized pair. `false` when no
    /// samples were observed.
    pub fn is_dual_mono(&self) -> bool {
        self.n > 0.0 && self.correlation() > DUAL_MONO_CORRELATION_THRESHOLD
    }
}

/// Test-only alias so the streaming detector can be pinned against the exact
/// batch statistic it replaces.
#[cfg(test)]
pub(crate) fn normalized_correlation_for_test(a: &[f32], b: &[f32]) -> f64 {
    normalized_correlation(a, b)
}

fn normalized_correlation(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len();
    if n == 0 || n != b.len() {
        return 0.0;
    }
    let mean_a = a.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    let mean_b = b.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for (&x, &y) in a.iter().zip(b) {
        let dx = x as f64 - mean_a;
        let dy = y as f64 - mean_b;
        cov += dx * dy;
        var_a += dx * dx;
        var_b += dy * dy;
    }
    let denom = var_a.sqrt() * var_b.sqrt();
    if denom < 1e-12 {
        return 0.0;
    }
    cov / denom
}
