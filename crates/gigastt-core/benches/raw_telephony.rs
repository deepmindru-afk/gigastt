//! Prep-only raw telephony CPU, allocation and process high-water measurements.
use gigastt_core::inference::audio::{
    TelephonyCodec, decode_telephony_raw, decode_telephony_raw_bounded_with_abort,
    encode_wav_pcm16, quantize_wav_pcm16_in_place,
};
use serde_json::json;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

struct CountingAllocator;
static ENABLED: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
fn add(size: usize) {
    ALLOCS.fetch_add(1, Relaxed);
    BYTES.fetch_add(size, Relaxed);
    PEAK.fetch_max(LIVE.fetch_add(size, Relaxed) + size, Relaxed);
}
// SAFETY: forward each allocation unchanged to System; only inspect its size.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() && ENABLED.load(Relaxed) {
            add(layout.size());
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if ENABLED.load(Relaxed) {
            LIVE.fetch_sub(layout.size(), Relaxed);
        }
        unsafe { System.dealloc(ptr, layout) };
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let next = unsafe { System.realloc(ptr, layout, size) };
        if !next.is_null() && ENABLED.load(Relaxed) {
            LIVE.fetch_sub(layout.size(), Relaxed);
            add(size);
        }
        next
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

enum Prepared {
    Wav(Vec<u8>),
    Pcm(Vec<f32>),
}
impl Prepared {
    fn bytes(&self) -> usize {
        match self {
            Self::Wav(v) => v.len(),
            Self::Pcm(v) => v.len() * 4,
        }
    }
}
fn prepare(data: &[u8], codec: TelephonyCodec, mode: &str) -> anyhow::Result<Prepared> {
    if mode == "legacy" {
        let samples = decode_telephony_raw(data, codec, 8000)?;
        Ok(Prepared::Wav(encode_wav_pcm16(&samples, 16000)))
    } else {
        let mut samples = decode_telephony_raw_bounded_with_abort(data, codec, 8000, None, None)?;
        if mode == "hybrid" && samples.len() > 30 * 16_000 {
            return Ok(Prepared::Wav(encode_wav_pcm16(&samples, 16000)));
        }
        for chunk in samples.chunks_mut(4096) {
            quantize_wav_pcm16_in_place(chunk);
        }
        Ok(Prepared::Pcm(samples))
    }
}
#[cfg(unix)]
fn resources() -> anyhow::Result<(f64, libc::c_long)> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: getrusage initializes the writable rusage on success; no value
    // is read on failure, and the pointer remains valid for the call.
    let result = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    anyhow::ensure!(result == 0, "getrusage failed");
    let usage = unsafe { usage.assume_init() };
    let cpu_ms = (usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) as f64 * 1000.0
        + (usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) as f64 / 1000.0;
    Ok((cpu_ms, usage.ru_maxrss))
}
#[cfg(not(unix))]
fn resources() -> anyhow::Result<(f64, i64)> {
    Ok((0.0, 0))
}
fn parity() -> anyhow::Result<()> {
    use gigastt_core::inference::{
        Engine, FeatureExtractor, TranscribeOverrides, TranscribeRequest, TranscribeSource,
    };
    use gigastt_core::model::ModelVariant;
    use gigastt_core::vad::{SileroVad, VadConfig};
    let models = std::env::var("GIGASTT_RAW_BENCH_MODEL_DIR")?;
    let vad = SileroVad::load(&std::path::Path::new(&models).join("vad/silero_vad.onnx"))?;
    let engine =
        Engine::load_with_pools_threads_variant(&models, Some(ModelVariant::MlCtc), 1, 1, 0, 4)?
            .with_vad(Some(vad), VadConfig::default());
    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../gigastt/tests/fixtures/telephony");
    let extractor = FeatureExtractor::new();
    let mut slot = engine.pool.checkout_blocking()?;
    for (codec, raw_file, native_file, repetitions) in [
        (TelephonyCodec::Pcmu, "speech.ulaw", "speech_mulaw.wav", 1),
        (TelephonyCodec::Pcma, "speech.alaw", "speech_alaw.wav", 1),
        (TelephonyCodec::G722, "speech.g722", "speech_g722.wav", 1),
        (TelephonyCodec::Pcmu, "speech.ulaw", "speech_mulaw.wav", 12),
    ] {
        let raw = std::fs::read(fixtures.join(raw_file))?.repeat(repetitions);
        let Prepared::Wav(wav) = prepare(&raw, codec, "legacy")? else {
            unreachable!()
        };
        let prepared = prepare(&raw, codec, "hybrid")?;
        let pcm = match &prepared {
            Prepared::Pcm(pcm) => pcm.clone(),
            Prepared::Wav(wav) => gigastt_core::inference::audio::decode_audio_bytes(wav)?,
        };
        let expected = gigastt_core::inference::audio::decode_audio_bytes(&wav)?;
        anyhow::ensure!(
            pcm.len() == expected.len()
                && pcm
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
            "PCM differs for {raw_file}"
        );
        let (a, af) = extractor.compute(&pcm);
        let (b, bf) = extractor.compute(&expected);
        anyhow::ensure!(
            af == bf && a.iter().zip(&b).all(|(a, b)| a.to_bits() == b.to_bits()),
            "features differ for {raw_file}"
        );
        for vad in [false, true] {
            let overrides = TranscribeOverrides {
                punctuation: Some(false),
                itn: Some(false),
                vad: Some(vad),
            };
            let legacy = engine.transcribe_request(
                TranscribeRequest::new(TranscribeSource::Bytes(wav.clone().into()))
                    .with_overrides(overrides),
                &mut slot,
            )?;
            let candidate = engine.transcribe_request(
                TranscribeRequest::new(match &prepared {
                    Prepared::Pcm(pcm) => TranscribeSource::Samples(pcm),
                    Prepared::Wav(wav) => TranscribeSource::Bytes(wav.clone().into()),
                })
                .with_overrides(overrides),
                &mut slot,
            )?;
            let equal = serde_json::to_vec(&legacy)? == serde_json::to_vec(&candidate)?;
            println!(
                "{}",
                json!({"kind":"parity","codec":format!("{codec:?}"),"repetitions":repetitions,"vad":vad,
                "route":if matches!(prepared, Prepared::Pcm(_)) {"pcm"} else {"wav"},"samples":pcm.len(),"pcm_bits_equal":true,"features_bits_equal":true,"result_equal":equal,
                "legacy":legacy,"candidate":candidate})
            );
            anyhow::ensure!(
                equal,
                "raw WAV and Samples recognition differ for {raw_file}, vad={vad}, repetitions={repetitions}"
            );
        }
        if repetitions == 1 {
            let result = engine.transcribe_request(
                TranscribeRequest::new(TranscribeSource::Bytes(
                    std::fs::read(fixtures.join(native_file))?.into(),
                ))
                .with_overrides(TranscribeOverrides {
                    punctuation: Some(false),
                    itn: Some(false),
                    vad: Some(false),
                }),
                &mut slot,
            )?;
            println!(
                "{}",
                json!({"kind":"native_container","file":native_file,"result":result})
            );
        }
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let Ok(mode) = std::env::var("GIGASTT_RAW_BENCH_MODE") else {
        eprintln!(
            "Set GIGASTT_RAW_BENCH_MODE=legacy, pcm, hybrid or parity; see docs/raw-telephony-pcm.md"
        );
        return Ok(());
    };
    if mode == "parity" {
        return parity();
    }
    anyhow::ensure!(
        matches!(mode.as_str(), "legacy" | "pcm" | "hybrid"),
        "unsupported mode"
    );
    let codec = std::env::var("GIGASTT_RAW_BENCH_CODEC").unwrap_or_else(|_| "pcmu".into());
    let (codec, fixture) = match codec.as_str() {
        "pcmu" => (TelephonyCodec::Pcmu, "speech.ulaw"),
        "pcma" => (TelephonyCodec::Pcma, "speech.alaw"),
        "g722" => (TelephonyCodec::G722, "speech.g722"),
        _ => anyhow::bail!("unsupported codec"),
    };
    let seconds: usize = std::env::var("GIGASTT_RAW_BENCH_SECONDS")
        .unwrap_or_else(|_| "4".into())
        .parse()?;
    anyhow::ensure!(
        (1..=1800).contains(&seconds),
        "seconds must be within 1..=1800"
    );
    let original = std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../gigastt/tests/fixtures/telephony")
            .join(fixture),
    )?;
    let data: Vec<u8> = original
        .iter()
        .copied()
        .cycle()
        .take(seconds * 8000)
        .collect();
    let mut wall_ms = Vec::new();
    let mut cpu_ms = Vec::new();
    for _ in 0..5 {
        let cpu = resources()?.0;
        let start = Instant::now();
        let prepared = prepare(&data, codec, &mode)?;
        wall_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        cpu_ms.push(resources()?.0 - cpu);
        std::hint::black_box(prepared);
    }
    ALLOCS.store(0, Relaxed);
    BYTES.store(0, Relaxed);
    LIVE.store(0, Relaxed);
    PEAK.store(0, Relaxed);
    ENABLED.store(true, Relaxed);
    let prepared = prepare(&data, codec, &mode)?;
    ENABLED.store(false, Relaxed);
    let (allocations, allocated, peak, retained) = (
        ALLOCS.load(Relaxed),
        BYTES.load(Relaxed),
        PEAK.load(Relaxed),
        LIVE.load(Relaxed),
    );
    println!(
        "{}",
        json!({"mode":mode,"codec":format!("{codec:?}"),"seconds":seconds,"fixture":fixture,
        "wall_ms":wall_ms,"cpu_ms":cpu_ms,"allocations":allocations,"allocated_bytes":allocated,"peak_live_bytes":peak,
        "retained_capacity_bytes":retained,"prepared_payload_bytes":prepared.bytes(),"process_max_rss_native_units":resources()?.1})
    );
    Ok(())
}
