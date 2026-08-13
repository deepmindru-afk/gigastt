//! Log-mel spectrogram feature extraction for GigaAM v3.

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::f32::consts::PI;
use std::sync::Arc;

/// One HTK triangular mel band as a contiguous sparse slice of FFT bins.
///
/// Triangle filters are zero outside `[start, start + weights.len())`, so the
/// dense `[n_mels × n_freqs]` multiply (mostly zeros) is replaced by a tight
/// weighted sum over the non-zero support only — same math, fewer MACs.
#[derive(Clone, Debug)]
struct SparseMelBand {
    /// First FFT bin with a non-zero weight for this band.
    start: usize,
    /// Contiguous non-zero weights starting at `start`.
    weights: Vec<f32>,
}

pub struct MelSpectrogram {
    n_fft: usize,
    hop_length: usize,
    window: Vec<f32>,
    /// Sparse HTK mel filterbank (one band per mel bin).
    mel_bands: Vec<SparseMelBand>,
    fft: Arc<dyn Fft<f32>>,
}

impl Default for MelSpectrogram {
    fn default() -> Self {
        Self::new()
    }
}

impl MelSpectrogram {
    pub fn new() -> Self {
        let n_fft = super::N_FFT;
        let hop_length = super::HOP_LENGTH;
        let n_mels = super::N_MELS;
        let sample_rate = 16000.0_f32;
        let fmin = 0.0_f32;
        let fmax = sample_rate / 2.0;

        // Hann window
        let window: Vec<f32> = (0..n_fft)
            .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f32 / (n_fft - 1) as f32).cos()))
            .collect();

        // HTK mel filterbank (dense build → sparse bands for apply)
        let mel_filterbank = Self::create_mel_filterbank(n_fft, n_mels, sample_rate, fmin, fmax);
        let mel_bands = Self::sparsify_mel_filterbank(&mel_filterbank, n_mels, n_fft / 2 + 1);

        // Pre-plan FFT
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n_fft);

        Self {
            n_fft,
            hop_length,
            window,
            mel_bands,
            fft,
        }
    }

    fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
    }

    fn create_mel_filterbank(
        n_fft: usize,
        n_mels: usize,
        sample_rate: f32,
        fmin: f32,
        fmax: f32,
    ) -> Vec<f32> {
        let n_freqs = n_fft / 2 + 1; // 161

        let mel_min = Self::hz_to_mel(fmin);
        let mel_max = Self::hz_to_mel(fmax);

        // n_mels + 2 equally spaced points in mel space
        let mel_points: Vec<f32> = (0..=(n_mels + 1))
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
            .collect();

        let hz_points: Vec<f32> = mel_points.iter().map(|&m| Self::mel_to_hz(m)).collect();

        // Convert Hz to FFT bin indices (float for interpolation)
        let bin_points: Vec<f32> = hz_points
            .iter()
            .map(|&hz| hz * n_fft as f32 / sample_rate)
            .collect();

        let mut filterbank = vec![0.0_f32; n_mels * n_freqs];

        for m in 0..n_mels {
            let f_left = bin_points[m];
            let f_center = bin_points[m + 1];
            let f_right = bin_points[m + 2];

            let row_start = m * n_freqs;
            for k in 0..n_freqs {
                let freq = k as f32;
                let val = if freq >= f_left && freq <= f_center && f_center > f_left {
                    (freq - f_left) / (f_center - f_left)
                } else if freq > f_center && freq <= f_right && f_right > f_center {
                    (f_right - freq) / (f_right - f_center)
                } else {
                    0.0
                };
                filterbank[row_start + k] = val;
            }
        }

        filterbank
    }

    /// Convert a dense `[n_mels × n_freqs]` HTK filterbank into per-band sparse
    /// slices. Empty rows become a zero-weight single bin at index 0 so apply
    /// still has a well-defined band object (energy stays at the log floor).
    fn sparsify_mel_filterbank(
        filterbank: &[f32],
        n_mels: usize,
        n_freqs: usize,
    ) -> Vec<SparseMelBand> {
        debug_assert_eq!(filterbank.len(), n_mels * n_freqs);
        let mut bands = Vec::with_capacity(n_mels);
        for m in 0..n_mels {
            let row = &filterbank[m * n_freqs..(m + 1) * n_freqs];
            let first = row.iter().position(|&w| w != 0.0);
            let last = row.iter().rposition(|&w| w != 0.0);
            match (first, last) {
                (Some(start), Some(end)) => bands.push(SparseMelBand {
                    start,
                    weights: row[start..=end].to_vec(),
                }),
                _ => bands.push(SparseMelBand {
                    start: 0,
                    weights: vec![0.0],
                }),
            }
        }
        bands
    }

    /// Compute log-mel spectrogram from f32 audio samples.
    /// Returns features in shape [n_mels, num_frames] as a flat Vec.
    pub fn compute(&self, samples: &[f32]) -> (Vec<f32>, usize) {
        let n_freqs = self.n_fft / 2 + 1;
        let mut fft_input = vec![Complex::new(0.0_f32, 0.0); self.n_fft];
        let mut power = vec![0.0_f32; n_freqs];
        let mut output = Vec::new();
        let num_frames =
            self.compute_with_buffers(samples, &mut fft_input, &mut power, &mut output);
        (output, num_frames)
    }

    /// Compute log-mel spectrogram reusing pre-allocated `fft_input` and `power` buffers.
    ///
    /// `fft_input` must have length >= `self.n_fft`; `power` must have length >= `n_freqs`.
    /// Both buffers are resized automatically if too small.
    pub fn compute_with_buffers(
        &self,
        samples: &[f32],
        fft_input: &mut Vec<Complex<f32>>,
        power: &mut Vec<f32>,
        output: &mut Vec<f32>,
    ) -> usize {
        let n_freqs = self.n_fft / 2 + 1;

        // Number of frames (center=false)
        let n_mels = self.mel_bands.len();
        if samples.len() < self.n_fft {
            output.resize(n_mels, 0.0);
            return 1;
        }
        let num_frames = (samples.len() - self.n_fft) / self.hop_length + 1;

        output.resize(n_mels * num_frames, 0.0_f32);

        // Ensure reusable buffers are large enough
        if fft_input.len() < self.n_fft {
            fft_input.resize(self.n_fft, Complex::new(0.0_f32, 0.0));
        }
        if power.len() < n_freqs {
            power.resize(n_freqs, 0.0_f32);
        }

        for frame_idx in 0..num_frames {
            let start = frame_idx * self.hop_length;

            // Apply window and fill FFT input in-place
            for i in 0..self.n_fft {
                let sample = if start + i < samples.len() {
                    samples[start + i]
                } else {
                    0.0
                };
                fft_input[i] = Complex::new(sample * self.window[i], 0.0);
            }

            // FFT
            self.fft.process(&mut fft_input[..self.n_fft]);

            // Power spectrum (first n_fft/2 + 1 bins)
            for k in 0..n_freqs {
                power[k] = fft_input[k].norm_sqr();
            }

            // Sparse mel filterbank + log (identical to dense row dot-products)
            for (m, band) in self.mel_bands.iter().enumerate() {
                let mut mel_energy: f32 = 0.0;
                let end = band.start + band.weights.len();
                // Safety: sparsify only keeps bins in 0..n_freqs.
                debug_assert!(end <= n_freqs);
                for (i, &w) in band.weights.iter().enumerate() {
                    mel_energy += w * power[band.start + i];
                }
                // Log with floor
                output[m * num_frames + frame_idx] = (mel_energy.max(1e-10)).ln();
            }
        }

        num_frames
    }
}

// Excluded under Miri: every test drives a real `rustfft` forward transform
// over 1 s / 200 ms buffers (dozens of FFTs per test). The FFT inner loops are
// orders of magnitude too slow under the Miri interpreter to finish in the
// nightly job's budget. These are numeric-correctness tests, not pointer /
// aliasing tests — they add nothing to Miri's UB signal and run natively on
// every `cargo test`. Documented coverage gap: the mel FFT path is not
// Miri-checked.
#[cfg(all(test, not(miri)))]
mod tests;
