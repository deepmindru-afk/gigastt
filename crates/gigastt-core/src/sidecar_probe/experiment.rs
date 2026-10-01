use super::*;
use crate::inference::{Engine, TranscribeRequest, TranscribeSource, audio::decode_audio_file};
use serde_json::{Value, json};
use std::path::Path;
use std::sync::{Arc, Barrier};
use std::time::Duration;

fn cpu_ticks() -> u64 {
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap();
    let fields: Vec<_> = stat
        .rsplit_once(") ")
        .unwrap()
        .1
        .split_whitespace()
        .collect();
    fields[11].parse::<u64>().unwrap() + fields[12].parse::<u64>().unwrap()
}
fn rss_kib() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .unwrap()
        .lines()
        .find(|s| s.starts_with("VmRSS:"))
        .unwrap()
        .split_whitespace()
        .nth(1)
        .unwrap()
        .parse()
        .unwrap()
}
fn observations() -> Vec<Observation> {
    std::mem::take(&mut *RECORDS.lock())
}
fn batch(engine: &Engine, audio: &[f32], speaker: bool) -> Value {
    let start = Instant::now();
    let mut slot = engine.pool.checkout_blocking().unwrap();
    let checkout_ns = start.elapsed().as_nanos();
    let result = engine
        .transcribe_request(
            TranscribeRequest::new(TranscribeSource::Samples(audio)).with_diarization(speaker),
            &mut slot,
        )
        .unwrap();
    json!({"kind":"batch","checkout_ns":checkout_ns,"latency_ns":start.elapsed().as_nanos(),"result":result,"rss_kib":rss_kib()})
}
fn interactive(engine: &Engine, audio: &[f32], speaker: bool, paced: bool) -> Value {
    let start = Instant::now();
    let mut slot = engine.pool.checkout_blocking().unwrap();
    let checkout_ns = start.elapsed().as_nanos();
    let mut state = engine.create_state(speaker);
    let mut events = Vec::new();
    let mut first_partial_ns = None;
    let mut chunk_ns = Vec::new();
    let mut peak = rss_kib();
    let first_chunk = Instant::now();
    for (index, chunk) in audio.chunks(1600).enumerate() {
        if paced
            && let Some(delay) =
                Duration::from_millis(index as u64 * 100).checked_sub(first_chunk.elapsed())
        {
            std::thread::sleep(delay);
        }
        let chunk_start = Instant::now();
        let segments = engine.process_chunk(chunk, &mut state, &mut slot).unwrap();
        if first_partial_ns.is_none() && segments.iter().any(|s| !s.is_final && !s.text.is_empty())
        {
            first_partial_ns = Some(start.elapsed().as_nanos());
        }
        events.extend(segments);
        chunk_ns.push(chunk_start.elapsed().as_nanos());
        peak = peak.max(rss_kib());
    }
    if paced
        && let Some(delay) =
            Duration::from_secs_f64(audio.len() as f64 / 16000.0).checked_sub(first_chunk.elapsed())
    {
        std::thread::sleep(delay);
    }
    let finalize_start = Instant::now();
    events.push(engine.finish_stream(&mut state, &mut slot).unwrap());
    json!({"kind":"interactive","first_partial_ns":first_partial_ns,"stop_flush_ns":finalize_start.elapsed().as_nanos(),"checkout_ns":checkout_ns,"latency_ns":start.elapsed().as_nanos(),"chunk_ns":chunk_ns,"events":events,"rss_kib":peak})
}

fn recognition_output(mut value: Value) -> Value {
    if let Some(events) = value.as_array_mut() {
        for event in events {
            event.as_object_mut().unwrap().remove("timestamp");
        }
    }
    value
}

#[test]
#[ignore = "local sidecar contention research; requires GIGASTT_SIDECAR_PROBE"]
fn benchmark_sidecar_contention() {
    let Ok(output) = std::env::var("GIGASTT_SIDECAR_PROBE") else {
        eprintln!("skip sidecar contention research: set GIGASTT_SIDECAR_PROBE output path");
        return;
    };
    let case = std::env::var("GIGASTT_SIDECAR_CASE").unwrap_or_else(|_| "none".into());
    assert!(["none", "vad", "punctuation", "speaker", "all"].contains(&case.as_str()));
    let pool: usize = std::env::var("GIGASTT_SIDECAR_POOL")
        .unwrap_or_else(|_| "1".into())
        .parse()
        .unwrap();
    assert!([1, 2, 4].contains(&pool));
    let speaker = case == "speaker" || case == "all";
    let load_start = Instant::now();
    let mut engine = Engine::load_with_factory(
        Path::new(&crate::model::default_model_dir()),
        Some(crate::model::ModelVariant::Rnnt),
        pool,
        pool,
        0,
        crate::runtime::cpu_factory(),
        4,
        true,
        "cpu",
    )
    .unwrap()
    .with_itn(true);
    if case == "punctuation" || case == "all" {
        engine = engine.with_punctuator(Some(
            crate::punctuation::Punctuator::load(Path::new(
                &crate::model::default_punct_model_dir(),
            ))
            .unwrap(),
        ));
    }
    if case == "vad" || case == "all" {
        engine = engine.with_vad(
            Some(
                crate::vad::SileroVad::load(
                    &Path::new(&crate::model::default_vad_model_dir())
                        .join(crate::vad::VAD_MODEL_FILE),
                )
                .unwrap(),
            ),
            crate::vad::VadConfig::default(),
        );
    }
    assert!(!speaker || engine.has_speaker_encoder());
    let load_ns = load_start.elapsed().as_nanos();
    let after_load_rss = rss_kib();
    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../gigastt/tests/fixtures");
    let clips: Vec<_> = (0..3)
        .map(|i| {
            decode_audio_file(
                fixture_dir
                    .join(format!("golos_{i:02}.wav"))
                    .to_str()
                    .unwrap(),
            )
            .unwrap()
        })
        .collect();
    ENABLED.store(true, Ordering::Relaxed);
    let cold_batch = batch(&engine, &clips[0], speaker);
    let cold_batch_observations = observations();
    eprintln!("cold batch completed: {} ns", cold_batch["latency_ns"]);
    let cold_interactive = interactive(&engine, &clips[0], speaker, false);
    let cold_interactive_observations = observations();
    eprintln!(
        "first streaming call completed: {} ns",
        cold_interactive["latency_ns"]
    );
    engine.warmup().unwrap();
    let serial_batch: Vec<_> = clips
        .iter()
        .map(|clip| batch(&engine, clip, speaker)["result"].clone())
        .collect();
    let serial_interactive: Vec<_> = clips
        .iter()
        .map(|clip| interactive(&engine, clip, speaker, false)["events"].clone())
        .collect();
    observations();
    eprintln!("serial references completed; starting mixed workers");
    let engine = Arc::new(engine);
    let barrier = Arc::new(Barrier::new(4));
    let start = Instant::now();
    let cpu_start = cpu_ticks();
    let rows = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for worker in 0..4 {
            let engine = &engine;
            let clips = &clips;
            let barrier = &barrier;
            let serial_batch = &serial_batch;
            let serial_interactive = &serial_interactive;
            handles.push(scope.spawn(move || {
                barrier.wait();
                (0..3)
                    .map(|i| {
                        let mut row = if worker == 0 {
                            interactive(engine, &clips[i], speaker, true)
                        } else {
                            batch(engine, &clips[i], speaker)
                        };
                        let (actual, expected) = if worker == 0 {
                            (&row["events"], &serial_interactive[i])
                        } else {
                            (&row["result"], &serial_batch[i])
                        };
                        assert_eq!(
                            recognition_output(actual.clone()),
                            recognition_output(expected.clone()),
                            "same-configuration concurrent output changed"
                        );
                        row["worker"] = json!(worker);
                        row["clip"] = json!(i);
                        row
                    })
                    .collect::<Vec<_>>()
            }));
        }
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect::<Vec<_>>()
    });
    let cpu = cpu_ticks() - cpu_start;
    let elapsed_ns = start.elapsed().as_nanos();
    ENABLED.store(false, Ordering::Relaxed);
    let measured = observations();
    let raw = json!({"schema":1,"case":case,"pool":pool,"encoder_threads_per_slot":4,"load_ns":load_ns,"after_load_rss_kib":after_load_rss,"cold_batch":cold_batch,"cold_batch_observations":cold_batch_observations,"cold_interactive":cold_interactive,"cold_interactive_observations":cold_interactive_observations,"rows":rows,"observations":measured,"process_cpu_ticks":cpu,"elapsed_ns":elapsed_ns,"audio_seconds":clips.iter().map(|c| c.len()).sum::<usize>() as f64/16000.0*4.0});
    std::fs::write(output, serde_json::to_vec_pretty(&raw).unwrap()).unwrap();
}

#[test]
fn test_recognition_comparison_ignores_only_wall_clock_timestamp() {
    let event = json!([{"timestamp":1234.0,"text":"word","confidence":0.9,"words":[{"start":0.2,"end":0.4}],"is_final":true}]);
    let normalized = recognition_output(event);
    assert!(normalized[0].get("timestamp").is_none());
    assert_eq!(normalized[0]["confidence"], 0.9);
    assert_eq!(normalized[0]["words"][0]["start"], 0.2);
    assert_eq!(normalized[0]["is_final"], true);
}
