//! Opt-in real-logit CTC profiling. See docs/ctc-beam-profile.md.
use anyhow::Context;
use gigastt_core::inference::{
    FeatureExtractor, audio,
    ctc_profile::{BeamProbe, file_sha256},
};
use gigastt_core::runtime_api::{Shape, Tensor, TensorData, TensorDataView, cpu_factory};
use serde_json::json;
use std::alloc::{GlobalAlloc, Layout, System};
use std::path::PathBuf;
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

fn main() -> anyhow::Result<()> {
    let Some(output) = std::env::var_os("GIGASTT_CTC_BENCH_OUTPUT") else {
        eprintln!("Set GIGASTT_CTC_BENCH_OUTPUT to capture and profile local ml_ctc logits");
        return Ok(());
    };
    let output = PathBuf::from(output);
    std::fs::create_dir_all(&output)?;
    let models = std::env::var_os("GIGASTT_CTC_BENCH_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".gigastt/models")
        });
    let model = models.join("multilingual_ctc.int8.onnx");
    let vocab = models.join("multilingual_vocab.txt");
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../gigastt/tests/fixtures");
    let mut inputs = Vec::new();
    let mut provenance = Vec::new();
    for i in 0..15 {
        let name = format!("golos_{i:02}.wav");
        let path = fixtures.join(&name);
        provenance.push(json!({"file":name,"sha256":file_sha256(&path)?}));
        inputs.push(audio::decode_audio_file(
            path.to_str().context("fixture path is not UTF-8")?,
        )?);
    }
    let short = inputs[0].clone();
    let long: Vec<f32> = inputs.into_iter().flatten().take(24 * 16000).collect();
    anyhow::ensure!(long.len() == 24 * 16000, "fixtures do not cover 24 seconds");
    let runtime = cpu_factory().create(4)?;
    let encoder = runtime.load_session(&model, true)?;
    let extractor = FeatureExtractor::new();
    let provenance = json!({"model_sha256":file_sha256(&model)?, "vocab_sha256":file_sha256(&vocab)?,
        "fixtures":provenance,"long_construction":"concatenate golos_00 through golos_14, truncate to 384000 samples",
        "sample_rate":16000,"encoder_threads":4,"provider":"CPU","ort":ort::info(),
        "beam_width":8,"top_k":6,"boost":5.0,"arch":std::env::consts::ARCH});
    std::fs::write(
        output.join("provenance.json"),
        serde_json::to_vec_pretty(&provenance)?,
    )?;
    let mut captures = Vec::new();
    for (name, samples) in [("short", short), ("long", long)] {
        let mut feature_ms = Vec::new();
        let mut encoder_ms = Vec::new();
        let mut captured = None;
        for iteration in 0..4 {
            let start = Instant::now();
            let (features, frames) = extractor.compute(&samples);
            let feature_elapsed = start.elapsed().as_secs_f64() * 1000.0;
            let tensors = [
                Tensor::new(Shape::new(vec![1, 64, frames]), TensorData::F32(features))?,
                Tensor::new(Shape::new(vec![1]), TensorData::I64(vec![frames as i64]))?,
            ];
            let start = Instant::now();
            let result = encoder.run(&tensors)?;
            let encoder_elapsed = start.elapsed().as_secs_f64() * 1000.0;
            let frames = match result[1].view().data() {
                TensorDataView::I32(v) => usize::try_from(v[0])?,
                TensorDataView::I64(v) => usize::try_from(v[0])?,
                _ => anyhow::bail!("unexpected output length type"),
            };
            let logits = result[0]
                .view()
                .data()
                .as_f32()
                .context("expected float logits")?
                .to_vec();
            captured = Some((frames, logits));
            if iteration > 0 {
                feature_ms.push(feature_elapsed);
                encoder_ms.push(encoder_elapsed);
            }
        }
        let (frames, logits) = captured.context("no captured logits")?;
        let binary: Vec<u8> = logits
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let path = output.join(format!("{name}.f32le"));
        std::fs::write(&path, binary)?;
        let capture = json!({"capture":name,"samples":samples.len(),"frames":frames,"classes":71,
            "feature_ms":feature_ms,"encoder_ms":encoder_ms,"logits_sha256":file_sha256(&path)?});
        println!("{capture}");
        std::fs::write(
            output.join(format!("{name}.json")),
            serde_json::to_vec_pretty(&capture)?,
        )?;
        captures.push((name, frames, logits));
    }
    // Uniform ties are deliberately adversarial; they are not speech measurements.
    captures.push(("ambiguous", 600, vec![0.0; 600 * 71]));
    let words = [
        "шестьдесят",
        "тысяч",
        "тенге",
        "смотрешке",
        "синергия",
        "яблоки",
        "зеленые",
        "торт",
        "графские",
        "развалины",
        "бизнес",
        "михаила",
        "мурадяна",
        "алиса",
        "закажи",
        "килограммовый",
    ];
    let mut reports = Vec::new();
    for (name, frames, logits) in captures {
        for (glossary, words) in [
            ("fixture_words", &words[..]),
            (
                "unrelated",
                &[
                    "алгоритм",
                    "база",
                    "вектор",
                    "граф",
                    "датчик",
                    "емкость",
                    "журнал",
                    "запрос",
                    "индекс",
                    "канал",
                    "логика",
                    "модуль",
                    "нейрон",
                    "объект",
                    "память",
                    "радар",
                    "signal",
                    "vector",
                    "memory",
                    "queue",
                ][..],
            ),
        ] {
            for count in [0, 1, 8, 32, 64] {
                let phrases: Vec<_> = (0..count)
                    .map(|i| {
                        let phrase = if i < words.len() {
                            words[i].to_string()
                        } else {
                            format!(
                                "{} {}",
                                words[i % words.len()],
                                words[(i / words.len() - 1) % words.len()]
                            )
                        };
                        (phrase, 1.0)
                    })
                    .collect();
                let probe = BeamProbe::new(&vocab, &phrases, 5.0)?;
                anyhow::ensure!(
                    probe.phrase_count() == count,
                    "glossary phrases were dropped"
                );
                let (expected, stats) = probe.profile(&logits, frames);
                ALLOCS.store(0, Relaxed);
                BYTES.store(0, Relaxed);
                LIVE.store(0, Relaxed);
                PEAK.store(0, Relaxed);
                ENABLED.store(true, Relaxed);
                let counted = probe.decode(&logits, frames, None);
                ENABLED.store(false, Relaxed);
                let allocations = ALLOCS.load(Relaxed);
                let allocated_bytes = BYTES.load(Relaxed);
                let peak_live_bytes = PEAK.load(Relaxed);
                anyhow::ensure!(counted == expected, "allocation pass changed output");
                let allocation = json!({"allocations":allocations,"allocated_bytes":allocated_bytes,
                "peak_live_bytes":peak_live_bytes});
                let mut timings = Vec::new();
                for _ in 0..5 {
                    let start = Instant::now();
                    let actual = probe.decode(&logits, frames, None);
                    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                    anyhow::ensure!(actual == expected, "observer changed output");
                    timings.push(elapsed);
                }
                let report = json!({"capture":name,"glossary":glossary,"phrases":count,"frames":frames,"timings_ms":timings,
                "allocations":allocation,"stats":stats,"text":probe.text(&expected),"alignment":expected.alignment()});
                println!("{report}");
                reports.push(report);
            }
        }
    }
    std::fs::write(
        output.join("results.json"),
        serde_json::to_vec_pretty(&reports)?,
    )?;
    Ok(())
}
