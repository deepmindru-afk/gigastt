//! Research-only frame reuse prototype, deliberately outside production paths.
use super::*;

#[derive(Default)]
struct FrameCache {
    start: usize,
    frames: usize,
    features: Vec<f32>,
}
impl FrameCache {
    // All calls belong to one immutable audio timeline; start is an absolute
    // sample index in that timeline. A new stream must construct a new cache.
    // This prototype allocates/copies its channel-major output and is not a
    // production cache API or a claim about encoder context reuse.
    fn compute(
        &mut self,
        mel: &MelSpectrogram,
        start: usize,
        samples: &[f32],
    ) -> (Vec<f32>, usize, usize) {
        if samples.len() < mel.n_fft {
            let (features, frames) = mel.compute(samples);
            self.frames = 0;
            self.features.clear();
            return (features, frames, 0);
        }
        let frames = (samples.len() - mel.n_fft) / mel.hop_length + 1;
        let offset = start
            .checked_sub(self.start)
            .filter(|offset| offset % mel.hop_length == 0)
            .map(|offset| offset / mel.hop_length);
        let reused = offset.map_or(0, |offset| self.frames.saturating_sub(offset).min(frames));
        let (tail, tail_frames) = if reused < frames {
            mel.compute(&samples[reused * mel.hop_length..])
        } else {
            (Vec::new(), 0)
        };
        assert_eq!(tail_frames + reused, frames);
        let mut features = Vec::with_capacity(mel.mel_bands.len() * frames);
        for band in 0..mel.mel_bands.len() {
            if reused > 0 {
                let first = band * self.frames + offset.unwrap();
                features.extend_from_slice(&self.features[first..first + reused]);
            }
            features.extend_from_slice(&tail[band * tail_frames..(band + 1) * tail_frames]);
        }
        self.start = start;
        self.frames = frames;
        self.features.clone_from(&features);
        (features, frames, reused)
    }
}

#[test]
fn test_frontend_reuse_matches_fresh_frames_and_invalidates_reanchoring() {
    let samples: Vec<_> = (0..32000)
        .map(|i| ((i as f32) * 0.071).sin() + ((i as f32) * 0.019).cos())
        .collect();
    let mel = MelSpectrogram::new();
    let mut cache = FrameCache::default();
    let mut saved = 0;
    for (start, end, may_reuse) in [
        (0, 3200, false),
        (0, 9600, true),
        (1600, 14400, true),
        (1601, 16000, false),
        (1761, 20000, true),
        (5, 6400, false),
        (5, 100, false),
        (5, 6400, false),
        (165, 16000, true),
    ] {
        let (actual, frames, reused) = cache.compute(&mel, start, &samples[start..end]);
        let (expected, expected_frames) = mel.compute(&samples[start..end]);
        assert_eq!(frames, expected_frames);
        assert_eq!(actual, expected, "window {start}..{end}");
        if may_reuse {
            assert!(reused > 0);
        } else {
            assert_eq!(reused, 0);
        }
        saved += reused;
    }
    assert!(saved > 0);
}

#[cfg(feature = "file-decode")]
#[test]
#[ignore = "local frontend replay; requires GIGASTT_LIVE_REPLAY baseline artifact"]
fn benchmark_frontend_reuse_on_recorded_windows() {
    let Ok(path) = std::env::var("GIGASTT_LIVE_REPLAY") else {
        eprintln!("skip frontend replay: set GIGASTT_LIVE_REPLAY to a baseline JSON artifact");
        return;
    };
    let raw: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let fixtures =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../gigastt/tests/fixtures");
    let mel = MelSpectrogram::new();
    let mut fresh_ns = 0;
    let mut cached_ns = 0;
    let mut total_frames = 0;
    let mut reused_frames = 0;
    let mut reanchors = 0;
    for row in raw["rows"].as_array().unwrap() {
        let name = row["file"].as_str().unwrap();
        if !name.ends_with(".wav") {
            continue;
        }
        let samples =
            crate::inference::audio::decode_audio_file(fixtures.join(name).to_str().unwrap())
                .unwrap();
        let mut cache = FrameCache::default();
        for window in row["probe"]["windows"].as_array().unwrap() {
            let start = window["start"].as_u64().unwrap() as usize;
            let end = start + window["samples"].as_u64().unwrap() as usize;
            if cache.frames > 0 && start % mel.hop_length != cache.start % mel.hop_length {
                reanchors += 1;
            }
            let begin = std::time::Instant::now();
            let expected = mel.compute(&samples[start..end]);
            fresh_ns += begin.elapsed().as_nanos();
            let begin = std::time::Instant::now();
            let (actual, frames, reused) = cache.compute(&mel, start, &samples[start..end]);
            cached_ns += begin.elapsed().as_nanos();
            assert_eq!(frames, expected.1);
            assert_eq!(actual, expected.0, "{name} {start}..{end}");
            total_frames += frames;
            reused_frames += reused;
        }
    }
    assert!(total_frames > 0);
    eprintln!(
        "frontend_replay fresh_ns={fresh_ns} cached_ns={cached_ns} frames={total_frames} reused_frames={reused_frames} phase_invalidations={reanchors}"
    );
}
