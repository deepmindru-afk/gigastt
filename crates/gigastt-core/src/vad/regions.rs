//! Pure speech-region math over a probability sequence.

use super::VadConfig;

/// Turn a per-frame speech-probability sequence into merged `[start, end)`
/// speech-sample spans. Pure (no model) so it is unit-testable on synthetic
/// probabilities.
///
/// `frame_samples` is the samples-per-probability stride ([`crate::vad::VAD_FRAME_SAMPLES`]
/// in production); `total_samples` clamps the final span to the real signal
/// length. Applies, in order: threshold, min-silence merge (gaps shorter than
/// `min_silence_ms` do not split a region), min-speech drop, and symmetric
/// `speech_pad_ms` padding (clamped to `[0, total_samples]`, then re-merged if
/// padding makes neighbours overlap).
pub fn regions_from_probs(
    probs: &[f32],
    frame_samples: usize,
    total_samples: usize,
    cfg: &VadConfig,
) -> Vec<(usize, usize)> {
    if probs.is_empty() || total_samples == 0 {
        return Vec::new();
    }

    let min_silence = VadConfig::ms_to_samples(cfg.min_silence_ms);
    let min_speech = VadConfig::ms_to_samples(cfg.min_speech_ms);
    let pad = VadConfig::ms_to_samples(cfg.speech_pad_ms);

    // 1. Raw speech runs from the thresholded probabilities.
    let mut regions: Vec<(usize, usize)> = Vec::new();
    let mut run_start: Option<usize> = None;
    for (i, &p) in probs.iter().enumerate() {
        let speech = p >= cfg.threshold;
        if speech && run_start.is_none() {
            run_start = Some(i * frame_samples);
        } else if !speech && let Some(s) = run_start.take() {
            regions.push((s, i * frame_samples));
        }
    }
    if let Some(s) = run_start.take() {
        regions.push((s, total_samples));
    }
    if regions.is_empty() {
        return regions;
    }

    // 2. Merge regions separated by a silence gap shorter than min_silence.
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(regions.len());
    for (s, e) in regions {
        match merged.last_mut() {
            Some(last) if s.saturating_sub(last.1) < min_silence => last.1 = e,
            _ => merged.push((s, e)),
        }
    }

    // 3. Drop regions shorter than min_speech (measured before padding).
    merged.retain(|(s, e)| e - s >= min_speech);
    if merged.is_empty() {
        return merged;
    }

    // 4. Pad each side, clamp to the signal, then re-merge any overlaps the
    //    padding introduced.
    let mut padded: Vec<(usize, usize)> = Vec::with_capacity(merged.len());
    for (s, e) in merged {
        let ps = s.saturating_sub(pad);
        let pe = (e + pad).min(total_samples);
        match padded.last_mut() {
            Some(last) if ps <= last.1 => last.1 = last.1.max(pe),
            _ => padded.push((ps, pe)),
        }
    }
    padded
}

/// Map a timestamp on the compressed (silence-removed) timeline back to the
/// original timeline, given the kept speech `regions` (original `[start, end)`
/// sample ranges, in order) and `sample_rate`. Pure — unit-tested directly.
///
/// File transcription with VAD decodes a buffer formed by concatenating the
/// speech regions, so decoded word timestamps are in compressed time; this
/// undoes that compression. A time at or past the end of all regions clamps to
/// the last region's end (guards rounding past the final frame).
pub fn remap_compressed_seconds(
    t_compressed_s: f64,
    regions: &[(usize, usize)],
    sample_rate: f64,
) -> f64 {
    if regions.is_empty() {
        return t_compressed_s;
    }
    let target = (t_compressed_s * sample_rate).max(0.0);
    let mut acc = 0.0f64; // compressed-sample offset at the current region's start
    for &(s, e) in regions {
        let len = (e - s) as f64;
        if target <= acc + len {
            let into = (target - acc).max(0.0);
            return (s as f64 + into) / sample_rate;
        }
        acc += len;
    }
    let &(_, end) = regions.last().expect("non-empty checked above");
    end as f64 / sample_rate
}
