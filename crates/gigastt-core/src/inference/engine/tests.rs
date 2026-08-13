use super::*;
#[cfg(feature = "diarization")]
#[cfg(feature = "diarization")]
use crate::inference::diarization::SPEAKER_EMBEDDING_DIM;
use crate::inference::windows::CHUNK_THRESHOLD_SAMPLES;
use crate::inference::{EndpointMode, WordInfo};
#[cfg(feature = "diarization")]
use polyvoice::Embedder;

fn word(text: &str, start: f64, end: f64) -> WordInfo {
    WordInfo::new(text, start, end, 1.0, None)
}

#[test]
fn test_split_pool_routes_items_to_two_pools() {
    // Exercises the real split underlying `split_triplets` with a synthetic
    // `Pool<u32>` (no model). 4 items, batch 1 → interactive 3, batch 1.
    let (pool, batch) = Engine::split_pool(vec![1u32, 2, 3, 4], 1);
    assert_eq!(pool.total(), 3);
    assert_eq!(batch.as_ref().map(|b| b.total()), Some(1));

    // batch_pool_size 0 → split disabled, no batch pool.
    let (pool, batch) = Engine::split_pool(vec![1u32, 2, 3, 4], 0);
    assert_eq!(pool.total(), 4);
    assert!(batch.is_none());

    // Over-request clamps so at least one triplet stays interactive.
    let (pool, batch) = Engine::split_pool(vec![1u32, 2, 3], 9);
    assert_eq!(pool.total(), 1);
    assert_eq!(batch.as_ref().map(|b| b.total()), Some(2));

    // A single item can't be split.
    let (pool, batch) = Engine::split_pool(vec![1u32], 1);
    assert_eq!(pool.total(), 1);
    assert!(batch.is_none());
}

#[test]
fn test_engine_load_missing_dir() {
    let result = Engine::load_with_pool_size("/nonexistent/path/for/tests", 1);
    assert!(matches!(result, Err(GigasttError::ModelLoad { .. })));
}

#[test]
fn test_engine_load_empty_dir() {
    let dir = tempfile::tempdir().unwrap();
    let result = Engine::load_with_pool_size(dir.path().to_str().unwrap(), 1);
    assert!(matches!(result, Err(GigasttError::ModelLoad { .. })));
}

#[test]
fn test_speech_endpoint_policy_matrix() {
    // Auto: blank without VAD, or VAD fire.
    assert!(Engine::speech_endpoint(
        EndpointMode::Auto,
        true,
        false,
        false
    ));
    assert!(Engine::speech_endpoint(
        EndpointMode::Auto,
        false,
        true,
        true
    ));
    assert!(!Engine::speech_endpoint(
        EndpointMode::Auto,
        true,
        false,
        true
    )); // blank ignored w/ VAD
    // Assistant: only VAD.
    assert!(!Engine::speech_endpoint(
        EndpointMode::Assistant,
        true,
        false,
        false
    ));
    assert!(Engine::speech_endpoint(
        EndpointMode::Assistant,
        false,
        true,
        true
    ));
    // Manual: never auto.
    assert!(!Engine::speech_endpoint(
        EndpointMode::Manual,
        true,
        true,
        true
    ));
}

#[test]
#[ignore = "requires model"]
fn test_warmup_runs_silent_inference_on_every_triplet() {
    let engine = Engine::load_with_pool_size(&crate::model::default_model_dir(), 2)
        .expect("engine should load");
    engine
        .warmup()
        .expect("warmup must succeed on a working engine");
    assert_eq!(
        engine.pool.available(),
        engine.pool.total(),
        "every triplet must be returned to the pool after warmup"
    );
}

// ---- Model-backed coverage (process_chunk / transcribe / state) --------
//
// These need the GigaAM model on disk; CI / coverage runs them with
// `--include-ignored`. They exercise the real streaming + file-decode
// branches (empty input, sub-stride, sub-N_FFT, full decode + slide,
// silence/short transcription, finish_stream / flush_state).

#[test]
#[ignore = "requires model"]
fn test_create_state_initial_fields() {
    let engine = Engine::load_with_pool_size(&crate::model::default_model_dir(), 1)
        .expect("engine should load");
    let state = engine.create_state(false);
    assert!(state.audio_buffer.is_empty());
    assert!(state.assembler.is_empty());
    assert_eq!(state.window_start_samples, 0);
    assert_eq!(state.context_samples, 0);
    assert_eq!(state.pending_samples, 0);
    assert!(state.resampler.is_none());
    // No VAD attached on a default engine → no endpointer.
    assert!(state.vad_endpointer.is_none());
    // Decoder state seeded to blank.
    assert_eq!(state.decoder.consecutive_blanks, 0);
}

#[test]
#[ignore = "requires model"]
fn test_create_state_diarization_flag_ignored_without_feature() {
    // Without the `diarization` feature the flag is silently ignored and a
    // perfectly usable state still comes back.
    let engine = Engine::load_with_pool_size(&crate::model::default_model_dir(), 1)
        .expect("engine should load");
    let state = engine.create_state(true);
    assert!(state.audio_buffer.is_empty());
}

// Model-free guard for the 2.11.2 diarization fix: `load_speaker_encoder` must
// stay wired to polyvoice's `FbankOnnxExtractor` (rank-3 fbank input) via its
// 3-arg constructor, NOT the old rank-2 raw-waveform `OnnxEmbeddingExtractor`
// (4-arg, with a segment-samples window) that caused the `Got: 2 Expected: 3`
// failure. The extractor reads the ONNX model at construction, so a nonexistent
// path returns Err (never panics/Ok). This runs on every PR without the ~26 MB
// model, and won't compile if the loader's return type / constructor arity
// regresses to the waveform extractor.
#[cfg(feature = "diarization")]
#[test]
fn test_load_speaker_encoder_missing_model_errors() {
    let missing = Path::new("/nonexistent/gigastt-test/wespeaker_resnet34.onnx");
    let result = diarization::load_speaker_encoder(missing, 1);
    assert!(
        result.is_err(),
        "a missing WeSpeaker model must surface as Err, not panic or Ok"
    );
}

#[cfg(feature = "diarization")]
#[test]
#[ignore = "requires the WeSpeaker diarization model"]
fn test_speaker_encoder_accepts_waveform_audio() {
    let model_path = Path::new(&crate::model::default_model_dir()).join("wespeaker_resnet34.onnx");
    let encoder =
        diarization::load_speaker_encoder(&model_path, 1).expect("speaker encoder should load");
    let samples: Vec<f32> = (0..24_000)
        .map(|i| {
            let phase = std::f32::consts::TAU * 220.0 * i as f32 / 16_000.0;
            0.1 * phase.sin()
        })
        .collect();

    let embedding = encoder
        .embed(&samples)
        .expect("waveform must be converted to rank-3 fbank features");

    assert_eq!(embedding.len(), SPEAKER_EMBEDDING_DIM);
    assert!(embedding.iter().all(|value| value.is_finite()));
}

#[test]
#[ignore = "requires model"]
fn test_process_chunk_empty_input_returns_no_segments() {
    let engine = Engine::load_with_pool_size(&crate::model::default_model_dir(), 1)
        .expect("engine should load");
    let mut guard = engine.pool.checkout_blocking().expect("checkout");
    let mut state = engine.create_state(false);
    let segs = engine
        .process_chunk(&[], &mut state, &mut guard)
        .expect("empty chunk must not error");
    assert!(segs.is_empty(), "empty input yields no segments");
    assert_eq!(state.audio_buffer.len(), 0);
}

#[test]
#[ignore = "requires model"]
fn test_process_chunk_sub_stride_buffers_without_decoding() {
    // A chunk smaller than the decode stride is buffered and triggers no
    // decode (the stride gate returns early).
    let engine = Engine::load_with_pool_size(&crate::model::default_model_dir(), 1)
        .expect("engine should load");
    let mut guard = engine.pool.checkout_blocking().expect("checkout");
    let mut state = engine.create_state(false);
    let small = vec![0.0f32; 1600]; // 0.1s ≪ 0.8s stride
    let segs = engine
        .process_chunk(&small, &mut state, &mut guard)
        .expect("sub-stride chunk must not error");
    assert!(segs.is_empty(), "sub-stride chunk yields no segments yet");
    assert_eq!(state.audio_buffer.len(), 1600, "samples are buffered");
    assert_eq!(state.pending_samples, 1600, "pending counter advances");
}

#[test]
#[ignore = "requires model"]
fn test_process_chunk_silence_over_stride_decodes_no_words() {
    // Enough silence to cross the decode stride: the encoder runs but
    // produces no words (silence), so the partial path returns no segments
    // and the pending counter resets.
    let engine = Engine::load_with_pool_size(&crate::model::default_model_dir(), 1)
        .expect("engine should load");
    let mut guard = engine.pool.checkout_blocking().expect("checkout");
    let mut state = engine.create_state(false);
    let chunk = vec![0.0f32; 16000]; // 1s of silence, > 0.8s stride
    let segs = engine
        .process_chunk(&chunk, &mut state, &mut guard)
        .expect("decode of silence must not error");
    // Silence → no words → empty assembler → no partial segment emitted.
    assert!(segs.is_empty(), "silence decodes to no words");
    assert_eq!(
        state.pending_samples, 0,
        "decode resets the pending counter"
    );
}

#[test]
#[ignore = "requires model"]
fn test_flush_state_empty_returns_none() {
    let engine = Engine::load_with_pool_size(&crate::model::default_model_dir(), 1)
        .expect("engine should load");
    let mut state = engine.create_state(false);
    assert!(
        engine.flush_state(&mut state).is_none(),
        "an empty assembler flushes to None"
    );
}

#[test]
#[ignore = "requires model"]
fn test_flush_state_nonempty_returns_final_segment() {
    let engine = Engine::load_with_pool_size(&crate::model::default_model_dir(), 1)
        .expect("engine should load");
    let mut state = engine.create_state(false);
    state.assembler.set_words(vec![word("hello", 0.0, 0.4)]);
    let seg = engine
        .flush_state(&mut state)
        .expect("non-empty assembler flushes to a Final segment");
    assert!(seg.is_final);
    assert_eq!(seg.text, "hello");
    assert!(
        engine.flush_state(&mut state).is_none(),
        "finalize resets the assembler"
    );
}

#[test]
#[ignore = "requires model"]
fn test_finish_stream_no_pending_flushes_assembler() {
    // No buffered audio and no pending samples: finish_stream skips the
    // forced decode and just flushes whatever the assembler holds.
    let engine = Engine::load_with_pool_size(&crate::model::default_model_dir(), 1)
        .expect("engine should load");
    let mut guard = engine.pool.checkout_blocking().expect("checkout");
    let mut state = engine.create_state(false);
    state.assembler.set_words(vec![word("trailing", 0.0, 0.4)]);
    let seg = engine
        .finish_stream(&mut state, &mut guard)
        .expect("finish_stream flushes the assembler");
    assert_eq!(seg.text, "trailing");
    assert!(seg.is_final);
}

#[test]
#[ignore = "requires model"]
fn test_finish_stream_empty_state_returns_none() {
    let engine = Engine::load_with_pool_size(&crate::model::default_model_dir(), 1)
        .expect("engine should load");
    let mut guard = engine.pool.checkout_blocking().expect("checkout");
    let mut state = engine.create_state(false);
    assert!(
        engine.finish_stream(&mut state, &mut guard).is_none(),
        "an idle stream finishes to None"
    );
}

#[test]
#[ignore = "requires model"]
fn test_transcribe_samples_silence_yields_empty_text() {
    // The single-pass file path on pure silence: the encoder runs, decode
    // produces no words, and the result text is empty with a correct
    // duration.
    let engine = Engine::load_with_pool_size(&crate::model::default_model_dir(), 1)
        .expect("engine should load");
    let mut guard = engine.pool.checkout_blocking().expect("checkout");
    let silence = vec![0.0f32; 16000 * 2]; // 2s
    let result = engine
        .transcribe_samples(&silence, &mut guard)
        .expect("silence transcription must not error");
    assert!(result.text.trim().is_empty(), "silence yields no text");
    assert!(result.words.is_empty());
    assert!((result.duration_s - 2.0).abs() < 1e-6);
}

#[test]
#[ignore = "requires model"]
fn test_transcribe_samples_short_sub_frame_audio() {
    // Audio shorter than a single FFT frame: the mel extractor pads to one
    // zero frame and the encoder still runs — exercising the short-input
    // single-pass branch without panicking. The decode output is whatever
    // the model emits on a lone padded frame; we only assert it doesn't
    // error and the reported duration matches the (tiny) input.
    let engine = Engine::load_with_pool_size(&crate::model::default_model_dir(), 1)
        .expect("engine should load");
    let mut guard = engine.pool.checkout_blocking().expect("checkout");
    let tiny = vec![0.0f32; 100]; // < N_FFT (320)
    let result = engine
        .transcribe_samples(&tiny, &mut guard)
        .expect("sub-frame audio must not error");
    assert!((result.duration_s - 100.0 / 16000.0).abs() < 1e-9);
}

#[test]
#[ignore = "requires model"]
fn test_transcribe_samples_below_chunk_threshold_single_pass() {
    // Just under the long-form chunk threshold (30s) takes the single-pass
    // path; pure silence still yields no words but must not error.
    let engine = Engine::load_with_pool_size(&crate::model::default_model_dir(), 1)
        .expect("engine should load");
    let mut guard = engine.pool.checkout_blocking().expect("checkout");
    let samples = vec![0.0f32; CHUNK_THRESHOLD_SAMPLES]; // exactly at threshold → single pass
    let result = engine
        .transcribe_samples(&samples, &mut guard)
        .expect("at-threshold audio must not error");
    assert!(result.words.is_empty(), "silence decodes to no words");
}

/// End-to-end proof that the Candle backend transcribes IDENTICALLY to the
/// ort backend through the full engine pipeline (mel + encoder + RNN-T
/// decode), not just per-stage tensors.
///
/// Two engines are built on the SAME model dir — one forced onto the ort CPU
/// backend, one forced onto the Candle backend (which reads the sibling
/// `candle/*.safetensors`) — and the same fixture wav is transcribed by both.
/// Per-stage parity is bit-exact, so the decoded text MUST be byte-identical.
///
/// Run with:
/// `cargo test -p gigastt-core --features candle --lib -- --ignored --nocapture candle_ort_transcription_parity`
#[cfg(feature = "candle")]
#[test]
#[ignore = "requires v3_rnnt model + candle/*.safetensors"]
fn candle_ort_transcription_parity() {
    let model_dir = crate::model::default_model_dir();
    let model_path = Path::new(&model_dir);

    let ort_engine = Engine::load_with_factory(
        model_path,
        None,
        1,
        1,
        0,
        Box::new(crate::runtime::ort::factory::OrtFactory::cpu()),
        1,
    )
    .expect("ort engine should load");
    let candle_engine = Engine::load_with_factory(
        model_path,
        None,
        1,
        1,
        0,
        Box::new(crate::runtime::candle::factory::CandleFactory::new()),
        1,
    )
    .expect("candle engine should load");

    let fixtures = [
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../gigastt/tests/fixtures/golos_00.wav"
        ),
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../gigastt/tests/fixtures/golos_01.wav"
        ),
    ];

    for fixture in fixtures {
        let mut ort_guard = ort_engine.pool.checkout_blocking().expect("ort checkout");
        let ort_text = ort_engine
            .transcribe_file(fixture, &mut ort_guard)
            .expect("ort transcription")
            .text;

        let mut candle_guard = candle_engine
            .pool
            .checkout_blocking()
            .expect("candle checkout");
        let candle_text = candle_engine
            .transcribe_file(fixture, &mut candle_guard)
            .expect("candle transcription")
            .text;

        eprintln!("fixture = {fixture}");
        eprintln!("ort    = {ort_text:?}");
        eprintln!("candle = {candle_text:?}");

        assert_eq!(
            ort_text, candle_text,
            "candle transcript diverges from ort for {fixture}:\n  ort    = {ort_text:?}\n  candle = {candle_text:?}"
        );
    }
}

/// End-to-end proof that the Candle backend produces IDENTICAL output to ort
/// through the STREAMING path (sliding-window `process_chunk` + `finish_stream`),
/// not just through the whole-file `transcribe_file` path.
///
/// Both engines receive the SAME 8 000-sample (0.5 s) chunks fed in order;
/// all returned segment texts are concatenated and compared byte-for-byte.
/// Per-stage tensor parity is bit-exact, so the streamed transcripts must match.
///
/// Run with:
/// `cargo test -p gigastt-core --features candle --lib -- --ignored --nocapture candle_ort_streaming_parity`
#[cfg(feature = "candle")]
#[test]
#[ignore = "requires v3_rnnt model + candle/*.safetensors"]
fn candle_ort_streaming_parity() {
    const CHUNK_SAMPLES: usize = 8_000; // 0.5 s at 16 kHz

    let model_dir = crate::model::default_model_dir();
    let model_path = Path::new(&model_dir);

    let ort_engine = Engine::load_with_factory(
        model_path,
        None,
        1,
        1,
        0,
        Box::new(crate::runtime::ort::factory::OrtFactory::cpu()),
        1,
    )
    .expect("ort engine should load");
    let candle_engine = Engine::load_with_factory(
        model_path,
        None,
        1,
        1,
        0,
        Box::new(crate::runtime::candle::factory::CandleFactory::new()),
        1,
    )
    .expect("candle engine should load");

    let fixtures = [
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../gigastt/tests/fixtures/golos_00.wav"
        ),
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../gigastt/tests/fixtures/golos_01.wav"
        ),
    ];

    for fixture in fixtures {
        let samples = audio::decode_audio_file(fixture)
            .unwrap_or_else(|e| panic!("failed to decode {fixture}: {e:#}"));

        // --- ort streaming ---
        let mut ort_guard = ort_engine.pool.checkout_blocking().expect("ort checkout");
        let mut ort_state = ort_engine.create_state(false);
        let mut ort_text = String::new();
        for chunk in samples.chunks(CHUNK_SAMPLES) {
            let segs = ort_engine
                .process_chunk(chunk, &mut ort_state, &mut ort_guard)
                .expect("ort process_chunk must not error");
            for seg in segs {
                if !ort_text.is_empty() {
                    ort_text.push(' ');
                }
                ort_text.push_str(seg.text.trim());
            }
        }
        if let Some(seg) = ort_engine.finish_stream(&mut ort_state, &mut ort_guard) {
            if !ort_text.is_empty() {
                ort_text.push(' ');
            }
            ort_text.push_str(seg.text.trim());
        }

        // --- candle streaming ---
        let mut candle_guard = candle_engine
            .pool
            .checkout_blocking()
            .expect("candle checkout");
        let mut candle_state = candle_engine.create_state(false);
        let mut candle_text = String::new();
        for chunk in samples.chunks(CHUNK_SAMPLES) {
            let segs = candle_engine
                .process_chunk(chunk, &mut candle_state, &mut candle_guard)
                .expect("candle process_chunk must not error");
            for seg in segs {
                if !candle_text.is_empty() {
                    candle_text.push(' ');
                }
                candle_text.push_str(seg.text.trim());
            }
        }
        if let Some(seg) = candle_engine.finish_stream(&mut candle_state, &mut candle_guard) {
            if !candle_text.is_empty() {
                candle_text.push(' ');
            }
            candle_text.push_str(seg.text.trim());
        }

        eprintln!("fixture = {fixture}");
        eprintln!("ort    (streamed) = {ort_text:?}");
        eprintln!("candle (streamed) = {candle_text:?}");

        assert_eq!(
            ort_text, candle_text,
            "candle streamed transcript diverges from ort for {fixture}:\n  ort    = {ort_text:?}\n  candle = {candle_text:?}"
        );
    }
}

/// Word-level edit distance / WER between a reference and a hypothesis
/// transcript (Levenshtein over whitespace tokens, normalized by reference
/// word count). Used by the ANE measurement harness below.
#[cfg(all(feature = "ane", target_os = "macos"))]
fn word_error_rate(reference: &str, hypothesis: &str) -> f64 {
    let r: Vec<&str> = reference.split_whitespace().collect();
    let h: Vec<&str> = hypothesis.split_whitespace().collect();
    if r.is_empty() {
        return if h.is_empty() { 0.0 } else { 1.0 };
    }
    let mut prev: Vec<usize> = (0..=h.len()).collect();
    let mut cur = vec![0usize; h.len() + 1];
    for (i, rw) in r.iter().enumerate() {
        cur[0] = i + 1;
        for (j, hw) in h.iter().enumerate() {
            let cost = if rw == hw { 0 } else { 1 };
            cur[j + 1] = (prev[j] + cost).min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[h.len()] as f64 / r.len() as f64
}

/// The 15 Golos fixtures with ground-truth references (from
/// `crates/gigastt/tests/fixtures/manifest.json`). `(path, reference)`.
#[cfg(all(feature = "ane", target_os = "macos"))]
fn golos_fixtures() -> Vec<(String, &'static str)> {
    const REFS: &[(&str, &str)] = &[
        (
            "golos_00.wav",
            "шестьдесят тысяч тенге сколько будет стоить",
        ),
        (
            "golos_01.wav",
            "покажи мне на смотрешке телеканал синергия тв",
        ),
        ("golos_02.wav", "заказать яблоки зеленые"),
        (
            "golos_03.wav",
            "алиса закажи килограммовый торт графские развалины",
        ),
        ("golos_04.wav", "ищи телеканал про бизнес на тиви"),
        ("golos_05.wav", "михаила мурадяна"),
        (
            "golos_06.wav",
            "любовницы две тысячи тринадцать пятнадцатый сезон",
        ),
        ("golos_07.wav", "найди боевики"),
        ("golos_08.wav", "гетто сезон три"),
        ("golos_09.wav", "хочу посмотреть ростов папа на телевизоре"),
        ("golos_10.wav", "сбер какое твое самое ненавистное занятие"),
        ("golos_11.wav", "афина чем платят у китайцев"),
        (
            "golos_12.wav",
            "джой как работает досрочное погашение кредита",
        ),
        ("golos_13.wav", "у тебя найдется люк кейдж"),
        ("golos_14.wav", "у тебя будет лучшая часть пинк"),
    ];
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../gigastt/tests/fixtures/");
    REFS.iter()
        .map(|(f, r)| (format!("{dir}{f}"), *r))
        .collect()
}

/// Build the two engines (ort baseline + composite ANE) on the rnnt model in
/// the default model dir. Returns `None` (with a printed SKIP) when the
/// rnnt model or the bucket-768 `.mlpackage` is absent so the `#[ignore]`d
/// measurement tests degrade cleanly on machines without the assets.
#[cfg(all(feature = "ane", target_os = "macos"))]
fn ane_measurement_engines() -> Option<(Engine, Engine)> {
    let model_dir = crate::model::default_model_dir();
    let model_path = Path::new(&model_dir);

    let ane_dir = model_path.join("ane");
    let bucket_768 = ane_dir.join(crate::model::ane_package_dir_name(768));
    if !crate::model::ane_package_complete(&bucket_768) {
        eprintln!(
            "SKIP: ANE bucket-768 package missing in {} (run `gigastt download --ane` or convert locally)",
            ane_dir.display()
        );
        return None;
    }
    if ModelVariant::detect_in_dir(model_path).is_none() {
        eprintln!("SKIP: no model files in {model_dir} (run `gigastt download`)");
        return None;
    }

    let ort_engine = Engine::load_with_factory(
        model_path,
        None,
        1,
        1,
        0,
        Box::new(crate::runtime::ort::factory::OrtFactory::cpu()),
        1,
    )
    .expect("ort engine should load");
    let ane_engine = Engine::load_with_factory(
        model_path,
        None,
        1,
        1,
        0,
        Box::new(crate::runtime::coreml::factory::AneFactory::new()),
        1,
    )
    .expect("ANE engine should load");
    Some((ort_engine, ane_engine))
}

/// Run one encoder pass directly through a checked-out triplet (mirrors
/// [`Engine::run_inference`]'s encoder setup) and return the emitted
/// `encoded_len`. Used to compare the ANE and ort encoders' length tensors
/// for the SAME mel input, pinning [`calc_output_length`] against ONNX drift.
#[cfg(all(feature = "ane", target_os = "macos"))]
fn encoder_emitted_len(engine: &Engine, features: &[f32], num_frames: usize) -> usize {
    let mut guard = engine.pool.checkout_blocking().expect("checkout");
    let triplet = &mut *guard;
    triplet.encoder_inputs[0].resize_to(Shape::new(vec![1, N_MELS, num_frames]));
    triplet.encoder_inputs[0]
        .as_f32_mut()
        .expect("encoder signal tensor is f32")
        .copy_from_slice(features);
    triplet.encoder_inputs[1]
        .as_i64_mut()
        .expect("encoder length tensor is i64")[0] = num_frames as i64;
    let outputs = triplet
        .encoder
        .run(&triplet.encoder_inputs)
        .expect("encoder run");
    match outputs[1].view().data() {
        TensorDataView::I32(v) => usize::try_from(v[0]).expect("non-negative len"),
        TensorDataView::I64(v) => usize::try_from(v[0]).expect("non-negative len"),
        _ => panic!("unexpected encoder length tensor type"),
    }
}

/// FULL-GOLOS WER + frame-count-equality measurement (Part 2a + frame pin).
///
/// For every Golos fixture, transcribes through BOTH the composite ANE engine
/// and the pure-ort baseline, records mel length `T`, the bucket fill % and
/// which PATH the ANE encoder took (ANE bucket vs ort fallback), and emits a
/// per-clip table plus aggregate WER(ANE vs ort), WER(ANE vs truth) and
/// WER(ort vs truth). Additionally asserts the ANE and ort encoders emit the
/// SAME `encoded_len` for the same mel input across all fixtures' real `T`
/// (pins [`calc_output_length`] == the ort ONNX length op against drift).
///
/// ANE parity is SOFT (mask-free FP16 pad-up is not byte-exact: cosine >= 0.94
/// at >= 50% fill, a borderline token can flip), so the per-clip ANE-vs-ort
/// gate is a small WER threshold rather than byte equality.
///
/// Run with:
/// `cargo test -p gigastt-core --features ane --lib -- --ignored --nocapture ane_ort_transcription_parity`
#[cfg(all(feature = "ane", target_os = "macos"))]
#[test]
#[ignore = "requires v3_rnnt model + ~/.gigastt/models/ane/*.mlpackage + ANE hardware"]
fn ane_ort_transcription_parity() {
    let Some((ort_engine, ane_engine)) = ane_measurement_engines() else {
        return;
    };

    // Mirror the encoder-session selection policy (select_bucket over the
    // shipped ladder) so the table can label the bucket the ANE engine took
    // without instrumenting the session itself.
    use crate::model::ANE_BUCKETS;
    use crate::runtime::coreml::encoder_session::select_bucket;
    const FILL_FLOOR: f64 = 0.5;
    // Aggregate gate: the mean WER(ANE vs ort) across all 15 clips must stay
    // small. Per-clip gate: at most ONE word may differ from ort — the
    // documented FP16-pad-up borderline-token flip (see FILL_FLOOR) is allowed
    // on a single word, but a multi-word divergence is a real regression.
    const MAX_MEAN_WER: f64 = 0.05;
    const MAX_WORD_DIFF_PER_CLIP: usize = 1;

    let fixtures = golos_fixtures();
    let mut sum_wer_ane_ort = 0.0;
    let mut sum_wer_ane_truth = 0.0;
    let mut sum_wer_ort_truth = 0.0;
    let mut frame_eq_checked: Vec<usize> = Vec::new();
    // (clip, word-diff vs ort) for the post-table per-clip gate.
    let mut clip_word_diffs: Vec<(String, usize)> = Vec::new();

    eprintln!(
        "\n{:<12} {:>5} {:>6} {:>9} {:>6} texts",
        "clip", "T", "fill%", "path", "ident"
    );
    for (path, reference) in &fixtures {
        let samples = audio::decode_audio_file(path).expect("decode fixture");
        let (features, num_frames) = ane_engine.features.compute(&samples);
        let bucket = select_bucket(num_frames, ANE_BUCKETS, FILL_FLOOR);
        let on_ane = bucket.is_some();
        let fill = match bucket {
            Some(n) => num_frames as f64 / n as f64,
            None => 0.0,
        };
        let path_label = match bucket {
            Some(n) => format!("ANE-{n}"),
            None => "ort-fb".to_string(),
        };

        // Frame-count equality: the ANE and ort encoders must emit the SAME
        // encoded_len for the same mel input. (On the ort-fallback path this
        // is trivially the same session class, but we still assert it; on the
        // ANE path it pins calc_output_length against the ONNX length op.)
        let ane_len = encoder_emitted_len(&ane_engine, &features, num_frames);
        let ort_len = encoder_emitted_len(&ort_engine, &features, num_frames);
        assert_eq!(
            ane_len, ort_len,
            "encoded_len mismatch ANE={ane_len} ort={ort_len} for T={num_frames} ({path})"
        );
        if on_ane {
            let formula = crate::runtime::coreml::encoder_session::calc_output_length(num_frames);
            assert_eq!(
                formula, ort_len,
                "calc_output_length({num_frames})={formula} != ort encoder emitted {ort_len}"
            );
            frame_eq_checked.push(num_frames);
        }

        let mut ort_guard = ort_engine.pool.checkout_blocking().expect("ort checkout");
        let ort_text = ort_engine
            .transcribe_file(path, &mut ort_guard)
            .expect("ort transcription")
            .text;
        drop(ort_guard);

        let mut ane_guard = ane_engine.pool.checkout_blocking().expect("ANE checkout");
        let ane_text = ane_engine
            .transcribe_file(path, &mut ane_guard)
            .expect("ANE transcription")
            .text;
        drop(ane_guard);

        let wer_ane_ort = word_error_rate(&ort_text, &ane_text);
        let wer_ane_truth = word_error_rate(reference, &ane_text);
        let wer_ort_truth = word_error_rate(reference, &ort_text);
        sum_wer_ane_ort += wer_ane_ort;
        sum_wer_ane_truth += wer_ane_truth;
        sum_wer_ort_truth += wer_ort_truth;

        // Absolute word-edit distance vs ort (WER * ort word count, rounded).
        let ort_words = ort_text.split_whitespace().count().max(1);
        let word_diff = (wer_ane_ort * ort_words as f64).round() as usize;

        let clip = std::path::Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(path);
        clip_word_diffs.push((clip.to_string(), word_diff));
        eprintln!(
            "{:<12} {:>5} {:>5.1}% {:>9} {:>6} ort={:?}",
            clip,
            num_frames,
            fill * 100.0,
            path_label,
            ort_text == ane_text,
            ort_text
        );
        eprintln!(
            "{:<12} {:>5} {:>6} {:>9} {:>6} ane={:?}  truth={:?}",
            "", "", "", "", "", ane_text, reference
        );
        eprintln!(
            "{:<12} WER ANE-vs-ort={wer_ane_ort:.4} (word_diff={word_diff}) ANE-vs-truth={wer_ane_truth:.4} ort-vs-truth={wer_ort_truth:.4}",
            ""
        );
    }

    let n = fixtures.len() as f64;
    let mean_ane_ort = sum_wer_ane_ort / n;
    eprintln!("\n=== AGGREGATE (n={}) ===", fixtures.len());
    eprintln!("mean WER(ANE vs ort)   = {mean_ane_ort:.4}");
    eprintln!("mean WER(ANE vs truth) = {:.4}", sum_wer_ane_truth / n);
    eprintln!("mean WER(ort vs truth) = {:.4}", sum_wer_ort_truth / n);
    eprintln!(
        "frame-count equality (ANE==ort encoded_len & ==calc_output_length) verified for T = {:?}",
        frame_eq_checked
    );

    // Gate AFTER measuring every clip (a measurement harness must not abort
    // mid-run). Aggregate parity must be tight; per clip, at most one word may
    // differ from ort (the documented single FP16-pad-up borderline flip).
    assert!(
        mean_ane_ort <= MAX_MEAN_WER,
        "mean WER(ANE vs ort) {mean_ane_ort:.4} > {MAX_MEAN_WER}"
    );
    for (clip, diff) in &clip_word_diffs {
        assert!(
            *diff <= MAX_WORD_DIFF_PER_CLIP,
            "clip {clip}: {diff} words differ from ort (> {MAX_WORD_DIFF_PER_CLIP}) — multi-word ANE divergence is a regression, not a borderline flip"
        );
    }
}

/// END-TO-END RTFx measurement (Part 2b).
///
/// For the fixtures that take the ANE path (>= 384 mel frames, >= 50% fill),
/// measures FULL-PIPELINE wall time (audio decode -> mel -> encoder -> RNN-T
/// greedy decode -> text) through the ANE engine and the ort baseline, warm
/// (first run discarded), median of >= 5. Reports RTFx (audio_secs / median_s)
/// for each engine and the speedup ratio ANE/ort. This quantifies how little
/// of the encoder-only ~230x ANE speedup survives the CPU-bound RNN-T decode
/// loop: the encoder is nearly free on the ANE, but end-to-end the pipeline is
/// decode-bound, so the realized full-pipeline speedup is only ~3.7x.
///
/// Run with:
/// `cargo test -p gigastt-core --features ane --lib -- --ignored --nocapture ane_e2e_rtfx`
#[cfg(all(feature = "ane", target_os = "macos"))]
#[test]
#[ignore = "requires v3_rnnt model + ~/.gigastt/models/ane/*.mlpackage + ANE hardware"]
fn ane_e2e_rtfx() {
    let Some((ort_engine, ane_engine)) = ane_measurement_engines() else {
        return;
    };

    use crate::model::ANE_BUCKETS;
    use crate::runtime::coreml::encoder_session::select_bucket;
    const FILL_FLOOR: f64 = 0.5;
    const WARM: usize = 1;
    const TIMED: usize = 6;

    fn median_secs(engine: &Engine, path: &str) -> f64 {
        // Warmup (discarded) + timed full-pipeline runs.
        for _ in 0..WARM {
            let mut g = engine.pool.checkout_blocking().expect("checkout");
            let _ = engine.transcribe_file(path, &mut g).expect("transcribe");
        }
        let mut times: Vec<f64> = Vec::with_capacity(TIMED);
        for _ in 0..TIMED {
            let mut g = engine.pool.checkout_blocking().expect("checkout");
            let t = std::time::Instant::now();
            let _ = engine.transcribe_file(path, &mut g).expect("transcribe");
            times.push(t.elapsed().as_secs_f64());
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        times[times.len() / 2]
    }

    eprintln!(
        "\n{:<12} {:>5} {:>6} {:>6} {:>9} {:>9} {:>9} {:>9} {:>8}",
        "clip",
        "T",
        "bucket",
        "audio_s",
        "ort_med_s",
        "ane_med_s",
        "ort_RTFx",
        "ane_RTFx",
        "speedup"
    );
    let mut any_ane = false;
    for (path, _ref) in golos_fixtures() {
        let samples = audio::decode_audio_file(&path).expect("decode fixture");
        let (_features, num_frames) = ane_engine.features.compute(&samples);
        let Some(bucket) = select_bucket(num_frames, ANE_BUCKETS, FILL_FLOOR) else {
            continue; // only clips that exercise the ANE encoder path
        };
        any_ane = true;
        let audio_s = samples.len() as f64 / 16000.0;
        let ort_med = median_secs(&ort_engine, &path);
        let ane_med = median_secs(&ane_engine, &path);
        let ort_rtfx = audio_s / ort_med;
        let ane_rtfx = audio_s / ane_med;
        let speedup = ort_med / ane_med;
        let clip = std::path::Path::new(&path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(&path)
            .to_string();
        eprintln!(
            "{:<12} {:>5} {:>6} {:>6.2} {:>9.4} {:>9.4} {:>9.1} {:>9.1} {:>7.2}x",
            clip, num_frames, bucket, audio_s, ort_med, ane_med, ort_rtfx, ane_rtfx, speedup
        );
    }
    assert!(
        any_ane,
        "no Golos fixture took the ANE path (>= 256 mel frames at >= 50% fill); cannot measure e2e RTFx"
    );
}

/// CONCURRENT-PREDICTION test (Part 1 item 2).
///
/// Builds ONE `AneEncoderSession` backed by a single shared `Arc<SharedModel>`
/// and fires concurrent `run` calls from N >= 4 threads on the SAME model,
/// asserting no crash/panic and that every thread gets the SAME output for the
/// same input (deterministic). Exercises the `unsafe impl Send/Sync` under
/// real `CPUAndNeuralEngine` multi-thread load.
///
/// Run with:
/// `cargo test -p gigastt-core --features ane --lib -- --ignored --nocapture ane_concurrent_prediction_deterministic`
#[cfg(all(feature = "ane", target_os = "macos"))]
#[test]
#[ignore = "requires ~/.gigastt/models/ane/gigaam_v3_encoder_768.mlpackage + ANE hardware"]
fn ane_concurrent_prediction_deterministic() {
    use crate::runtime::coreml::bridge;
    use crate::runtime::coreml::encoder_session::{SharedModel, pad_time};
    use std::sync::Arc;

    let model_dir = crate::model::default_model_dir();
    let pkg = Path::new(&model_dir)
        .join("ane")
        .join(crate::model::ane_package_dir_name(768));
    if !crate::model::ane_package_complete(&pkg) {
        eprintln!("SKIP: ANE bucket-768 package missing at {}", pkg.display());
        return;
    }

    // Compile + load ONCE; share across threads (the production sharing model).
    let model = Arc::new(SharedModel(
        bridge::compile_and_load(&pkg, true).expect("compile_and_load bucket-768"),
    ));

    // Deterministic-but-non-trivial mel input padded to the 768 window.
    const T: usize = 600;
    const N: usize = 768;
    let mut mel = vec![0.0f32; N_MELS * T];
    for (i, v) in mel.iter_mut().enumerate() {
        *v = ((i % 97) as f32 * 0.013 - 0.5).sin();
    }
    let padded: Arc<Vec<f32>> = Arc::new(pad_time(&mel, N_MELS, T, N));

    // Single-threaded reference output.
    let (reference, ref_shape) =
        bridge::predict_f32(&model.0, "mel", &padded, &[1, N_MELS, N], "encoded")
            .expect("reference predict");

    const THREADS: usize = 4;
    const PER_THREAD: usize = 5;
    let mut handles = Vec::with_capacity(THREADS);
    for _ in 0..THREADS {
        let model = Arc::clone(&model);
        let padded = Arc::clone(&padded);
        handles.push(std::thread::spawn(move || {
            let mut outs = Vec::with_capacity(PER_THREAD);
            for _ in 0..PER_THREAD {
                let (out, shape) =
                    bridge::predict_f32(&model.0, "mel", &padded, &[1, N_MELS, N], "encoded")
                        .expect("concurrent predict");
                outs.push((out, shape));
            }
            outs
        }));
    }

    let mut total = 0usize;
    for h in handles {
        let outs = h.join().expect("thread did not panic");
        for (out, shape) in outs {
            assert_eq!(shape, ref_shape, "concurrent output shape diverged");
            assert_eq!(
                out.len(),
                reference.len(),
                "concurrent output length diverged"
            );
            assert!(
                out.iter().all(|v| v.is_finite()),
                "concurrent output has non-finite values"
            );
            // Bit-for-bit determinism: same model + same input -> same output.
            assert_eq!(
                out, reference,
                "concurrent prediction diverged from the single-threaded reference"
            );
            total += 1;
        }
    }
    eprintln!(
        "concurrent OK: {THREADS} threads x {PER_THREAD} predicts = {total} runs, all deterministic & finite"
    );
}

mod mock_runtime_tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use crate::inference::{EndpointMode, EndpointReason, Engine, PRED_HIDDEN, WordInfo};
    use crate::runtime::mock::{MockFactory, MockSession};
    use crate::runtime::tensor::{Shape, Tensor, TensorData};

    const ENC_DIM: usize = 768;

    fn tiny_mock_engine() -> (Engine, tempfile::TempDir) {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path();

        // Empty model files are enough for variant detection; the mock
        // runtime intercepts all session loading before the filesystem is
        // read for ONNX data. Product path is INT8-only (stem keys the mock).
        std::fs::write(dir.join("v3_rnnt_encoder_int8.onnx"), b"").unwrap();
        std::fs::write(dir.join("v3_rnnt_decoder.onnx"), b"").unwrap();
        std::fs::write(dir.join("v3_rnnt_joint.onnx"), b"").unwrap();
        // vocab: index 0 = "▁hi", index 1 = "<blk>" (blank wins on ties).
        std::fs::write(dir.join("v3_vocab.txt"), "\u{2581}hi\n<blk>\n").unwrap();

        let mut sessions: HashMap<String, Arc<MockSession>> = HashMap::new();
        sessions.insert(
            "v3_rnnt_encoder_int8".into(),
            Arc::new(MockSession::new(
                vec![Shape::new(vec![1, 64, 1]), Shape::new(vec![1])],
                vec![
                    Tensor::new(
                        Shape::new(vec![1, ENC_DIM, 1]),
                        TensorData::F32(vec![0.0; ENC_DIM]),
                    )
                    .unwrap(),
                    Tensor::new(Shape::new(vec![1]), TensorData::I64(vec![1])).unwrap(),
                ],
            )),
        );
        sessions.insert(
            "v3_rnnt_decoder".into(),
            Arc::new(MockSession::new(
                vec![
                    Shape::new(vec![1, 1]),
                    Shape::new(vec![1, 1, PRED_HIDDEN]),
                    Shape::new(vec![1, 1, PRED_HIDDEN]),
                ],
                vec![
                    Tensor::new(
                        Shape::new(vec![1, 1, PRED_HIDDEN]),
                        TensorData::F32(vec![0.0; PRED_HIDDEN]),
                    )
                    .unwrap(),
                    Tensor::new(
                        Shape::new(vec![1, 1, PRED_HIDDEN]),
                        TensorData::F32(vec![0.0; PRED_HIDDEN]),
                    )
                    .unwrap(),
                    Tensor::new(
                        Shape::new(vec![1, 1, PRED_HIDDEN]),
                        TensorData::F32(vec![0.0; PRED_HIDDEN]),
                    )
                    .unwrap(),
                ],
            )),
        );
        sessions.insert(
            "v3_rnnt_joint".into(),
            Arc::new(MockSession::new(
                vec![
                    Shape::new(vec![1, ENC_DIM, 1]),
                    Shape::new(vec![1, PRED_HIDDEN, 1]),
                ],
                vec![
                    Tensor::new(Shape::new(vec![1, 1, 2]), TensorData::F32(vec![0.0; 2])).unwrap(),
                ],
            )),
        );

        let factory = Box::new(MockFactory::new(sessions));
        let engine = Engine::load_with_factory(dir, None, 1, 1, 0, factory, 1)
            .expect("engine should load with mock runtime");
        (engine, tmp)
    }

    #[test]
    fn test_engine_loads_with_mock_runtime() {
        let _ = tiny_mock_engine();
    }

    #[test]
    fn test_engine_mock_runtime_decodes_silence() {
        let (engine, _tmp) = tiny_mock_engine();
        let mut guard = engine.pool.checkout_blocking().expect("checkout");
        let samples = vec![0.0f32; 100]; // < N_FFT → one padded mel frame
        let result = engine
            .transcribe_samples(&samples, &mut guard)
            .expect("mock decode must not error");

        assert!(result.text.is_empty(), "blank-only decode yields no text");
        assert!(result.words.is_empty());
        assert!((result.duration_s - 100.0 / 16000.0).abs() < 1e-9);
    }

    #[test]
    fn test_validate_overrides_truth_table() {
        use crate::inference::{
            HotwordOverride, MAX_HOTWORD_PHRASE_CHARS, MAX_HOTWORDS_PER_REQUEST, OverrideError,
            TranscribeOverrides,
        };

        // The tiny mock engine loads with no VAD and no punctuator attached,
        // so any knob turned *on* per-request must be rejected, and any knob
        // turned *off* (or ITN in either direction) must be accepted.
        let (engine, _tmp) = tiny_mock_engine();
        assert!(!engine.has_vad(), "mock engine has no VAD");
        assert!(!engine.has_punctuator(), "mock engine has no punctuator");

        // All absent → OK (byte-unchanged default path).
        assert_eq!(
            engine.validate_overrides(&TranscribeOverrides::default()),
            Ok(())
        );

        // vad=Some(true) with no VAD → err vad_not_loaded.
        assert_eq!(
            engine.validate_overrides(&TranscribeOverrides {
                vad: Some(true),
                ..Default::default()
            }),
            Err(OverrideError::VadNotLoaded)
        );
        // vad=Some(false) is always OK (opting out never needs a resource).
        assert_eq!(
            engine.validate_overrides(&TranscribeOverrides {
                vad: Some(false),
                ..Default::default()
            }),
            Ok(())
        );

        // punctuation=Some(true) with no punctuator → err.
        assert_eq!(
            engine.validate_overrides(&TranscribeOverrides {
                punctuation: Some(true),
                ..Default::default()
            }),
            Err(OverrideError::PunctuationNotAvailable)
        );
        // punctuation=Some(false) is always OK.
        assert_eq!(
            engine.validate_overrides(&TranscribeOverrides {
                punctuation: Some(false),
                ..Default::default()
            }),
            Ok(())
        );

        // itn=Some(true) is always OK (pure code, no model to load).
        assert_eq!(
            engine.validate_overrides(&TranscribeOverrides {
                itn: Some(true),
                ..Default::default()
            }),
            Ok(())
        );
        assert_eq!(
            engine.validate_overrides(&TranscribeOverrides {
                itn: Some(false),
                ..Default::default()
            }),
            Ok(())
        );

        // Hotword DoS limits live on validate_hotwords (separate from
        // TranscribeOverrides so that type stays Copy/Eq).
        use crate::inference::HotwordError;
        assert_eq!(
            engine.validate_hotwords(&HotwordOverride::new(vec![], None)),
            Ok(())
        );
        assert_eq!(
            engine.validate_hotwords(&HotwordOverride::new(
                vec!["ok".into(), "fine".into()],
                Some(3.0),
            )),
            Ok(())
        );

        let at_cap: Vec<String> = (0..MAX_HOTWORDS_PER_REQUEST)
            .map(|i| format!("w{i}"))
            .collect();
        assert_eq!(
            engine.validate_hotwords(&HotwordOverride::new(at_cap, None)),
            Ok(())
        );
        let over_cap: Vec<String> = (0..=MAX_HOTWORDS_PER_REQUEST)
            .map(|i| format!("w{i}"))
            .collect();
        assert_eq!(
            engine.validate_hotwords(&HotwordOverride::new(over_cap, None)),
            Err(HotwordError::TooManyHotwords)
        );

        let ok_phrase: String = "а".repeat(MAX_HOTWORD_PHRASE_CHARS);
        assert_eq!(
            engine.validate_hotwords(&HotwordOverride::new(vec![ok_phrase], None)),
            Ok(())
        );
        let long_phrase: String = "а".repeat(MAX_HOTWORD_PHRASE_CHARS + 1);
        assert_eq!(
            engine.validate_hotwords(&HotwordOverride::new(vec![long_phrase], None)),
            Err(HotwordError::PhraseTooLong)
        );
    }

    #[test]
    fn test_request_hotword_biaser_semantics() {
        use crate::inference::{HotwordOverride, TranscribeOverrides};

        // Pin the three-way hotword override contract without requiring a
        // real speech utterance:
        // - None → use engine boot biaser (build_request_biaser not used)
        // - Some(empty) → force off (build_request_biaser returns None)
        // - Some(phrases) → temporary Biaser when representable
        let (engine, _tmp) = tiny_mock_engine();
        // Mock vocab is "▁hi" + blank; "hi" encodes, unknown Cyrillic does not.
        let engine = engine.with_hotwords(&[("hi".into(), 1.0)], 5.0);
        assert!(engine.has_hotwords(), "boot biaser attached");

        // Force-off: empty phrase list yields no temporary biaser.
        let off = HotwordOverride::new(vec![], None);
        assert!(
            engine.build_request_biaser(&off).is_none(),
            "empty override forces biasing off"
        );

        // Representable phrase builds a temporary biaser.
        let on = HotwordOverride::new(vec!["hi".into()], Some(7.0));
        let built = engine
            .build_request_biaser(&on)
            .expect("representable phrase should compile");
        assert_eq!(built.phrase_count(), 1);

        // Default boost path (None) still builds when phrases are present.
        let default_boost = HotwordOverride::new(vec!["hi".into()], None);
        assert!(engine.build_request_biaser(&default_boost).is_some());

        // Unrepresentable-only phrases → None (decode continues without bias).
        let junk = HotwordOverride::new(vec!["яяяя".into()], None);
        assert!(
            engine.build_request_biaser(&junk).is_none(),
            "unrepresentable phrases drop the temporary biaser"
        );

        // hotwords=None is always valid and leaves the boot biaser in place
        // (has_hotwords stays true; decode path selects self.biaser).
        assert_eq!(
            engine.validate_overrides(&TranscribeOverrides::default()),
            Ok(())
        );
        assert!(engine.has_hotwords());
    }

    #[test]
    fn test_ctc_head_attaches_a_biaser() {
        use crate::inference::HotwordOverride;
        use crate::model::ModelVariant;

        // A glossary used to be inert here: greedy CTC has no continuation
        // state, so the biaser was refused outright to stop the engine
        // claiming otherwise. The prefix beam gives it somewhere to act, so
        // both the boot biaser and the per-request one attach again.
        let (mut engine, _tmp) = tiny_mock_engine();
        engine.variant = ModelVariant::MlCtc;

        let engine = engine.with_hotwords(&[("hi".into(), 1.0)], 5.0);
        assert!(
            engine.has_hotwords(),
            "a CTC head biases through the prefix beam"
        );

        let per_request = HotwordOverride::new(vec!["hi".into()], Some(7.0));
        assert!(
            engine.build_request_biaser(&per_request).is_some(),
            "a per-request glossary applies on a CTC head too"
        );
    }

    #[test]
    fn test_transcribe_samples_with_overrides_vad_off_matches_default() {
        use crate::inference::TranscribeOverrides;

        // `?vad=false` on a VAD-less engine is a no-op relative to the default
        // path (both decode the whole buffer), so the output must be identical.
        let (engine, _tmp) = tiny_mock_engine();
        let mut guard = engine.pool.checkout_blocking().expect("checkout");
        let samples = vec![0.0f32; 100];
        let baseline = engine
            .transcribe_samples(&samples, &mut guard)
            .expect("baseline decode");
        let with_vad_off = engine
            .transcribe_samples_with_overrides(
                &samples,
                &mut guard,
                &TranscribeOverrides {
                    vad: Some(false),
                    ..Default::default()
                },
                None,
                false,
                None,
                super::super::DecodeControls::default(),
            )
            .expect("vad-off decode");
        assert_eq!(baseline.text, with_vad_off.text);
        assert_eq!(baseline.words.len(), with_vad_off.words.len());
    }

    #[test]
    fn test_diarization_requested_without_speaker_model_records_notice() {
        use crate::inference::{DiarizationOutcome, TranscribeRequest, TranscribeSource};
        use std::sync::{Arc, OnceLock};

        // A `?diarization=true` request against an engine with no speaker
        // model must record `NoSpeakerModel` in the outcome sink so the
        // server can surface it, instead of silently returning an
        // all-empty-speaker transcript. Model-free: the tiny mock engine
        // never ships a WeSpeaker model, and a build without the
        // `diarization` feature reports the same outcome via the same sink.
        let (engine, _tmp) = tiny_mock_engine();
        let mut guard = engine.pool.checkout_blocking().expect("checkout");
        let samples = vec![0.0f32; 100];
        let sink: Arc<OnceLock<DiarizationOutcome>> = Arc::new(OnceLock::new());
        let req = TranscribeRequest::new(TranscribeSource::Samples(&samples))
            .with_diarization(true)
            .with_diarization_outcome(Some(sink.clone()));
        engine
            .transcribe_request(req, &mut guard)
            .expect("decode should succeed");
        assert_eq!(
            sink.get().copied(),
            Some(DiarizationOutcome::NoSpeakerModel),
            "diarization requested without a model must be reported, not silent"
        );
    }

    #[test]
    fn test_create_state_postprocess_overrides_default_to_none() {
        // A fresh streaming state inherits the engine boot policy: both
        // per-session overrides start unset.
        let (engine, _tmp) = tiny_mock_engine();
        let state = engine.create_state(false);
        assert_eq!(state.punctuation, None);
        assert_eq!(state.itn, None);
    }

    #[test]
    fn test_flush_state_applies_itn_per_engine_default() {
        // Streaming finals go through the same ITN pass as the file path:
        // with the engine default on, number-words are digitized at flush.
        let (engine, _tmp) = tiny_mock_engine();
        let engine = engine.with_itn(true);
        let mut state = engine.create_state(false);
        state.assembler.set_words(vec![
            super::word("двадцать", 0.0, 0.4),
            super::word("один", 0.4, 0.8),
        ]);
        let seg = engine.flush_state(&mut state).expect("flush");
        assert_eq!(seg.text, "21");
        // Word payloads stay raw — only the joined text is rewritten.
        assert_eq!(seg.words[0].word, "двадцать");
        assert_eq!(seg.words[1].word, "один");
    }

    #[test]
    fn test_flush_state_session_override_disables_itn() {
        // `Configure{itn:false}` wins over an ITN-on engine boot policy.
        let (engine, _tmp) = tiny_mock_engine();
        let engine = engine.with_itn(true);
        let mut state = engine.create_state(false);
        state.itn = Some(false);
        state.assembler.set_words(vec![
            super::word("двадцать", 0.0, 0.4),
            super::word("один", 0.4, 0.8),
        ]);
        let seg = engine.flush_state(&mut state).expect("flush");
        assert_eq!(seg.text, "двадцать один");
    }

    #[test]
    fn test_flush_state_default_leaves_text_raw() {
        // Boot defaults (ITN off, no punctuator) reproduce the pre-feature
        // behaviour byte-for-byte.
        let (engine, _tmp) = tiny_mock_engine();
        let mut state = engine.create_state(false);
        state.assembler.set_words(vec![
            super::word("двадцать", 0.0, 0.4),
            super::word("один", 0.4, 0.8),
        ]);
        let seg = engine.flush_state(&mut state).expect("flush");
        assert_eq!(seg.text, "двадцать один");
    }

    #[test]
    fn test_flush_state_punctuation_request_without_punctuator_is_noop() {
        // `Configure{punctuation:true}` on a punct-less server degrades
        // gracefully: the final is emitted unchanged (streaming never
        // errors on post-processing, unlike the REST 409).
        let (engine, _tmp) = tiny_mock_engine();
        let mut state = engine.create_state(false);
        state.punctuation = Some(true);
        state
            .assembler
            .set_words(vec![super::word("hello", 0.0, 0.4)]);
        let seg = engine.flush_state(&mut state).expect("flush");
        assert_eq!(seg.text, "hello");
    }

    /// Engine whose scripted joiner emits one "▁hi" token on its first call
    /// and blanks afterwards, over a 16-frame encoder output: a single
    /// decode then crosses the decoder's blank-run endpoint threshold
    /// (15 consecutive blanks), reproducing a decoder endpoint without a
    /// model. The encoder expects exactly one decode stride of audio
    /// (12800 samples → 79 mel frames).
    fn blank_run_engine() -> (Engine, tempfile::TempDir) {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path();

        std::fs::write(dir.join("v3_rnnt_encoder_int8.onnx"), b"").unwrap();
        std::fs::write(dir.join("v3_rnnt_decoder.onnx"), b"").unwrap();
        std::fs::write(dir.join("v3_rnnt_joint.onnx"), b"").unwrap();
        // vocab: index 0 = "▁hi", index 1 = "<blk>".
        std::fs::write(dir.join("v3_vocab.txt"), "\u{2581}hi\n<blk>\n").unwrap();

        // (12800 - N_FFT) / HOP_LENGTH + 1 mel frames for one stride.
        const MEL_FRAMES: usize = 79;
        // 16 frames: frame 0 emits the token, then a blank on the same
        // frame's second joiner call; frames 1..=15 add 15 more blanks —
        // 16 consecutive blanks, one above the endpoint threshold.
        const ENC_LEN: usize = 16;

        let mut sessions: HashMap<String, Arc<MockSession>> = HashMap::new();
        sessions.insert(
            "v3_rnnt_encoder_int8".into(),
            Arc::new(MockSession::new(
                vec![Shape::new(vec![1, 64, MEL_FRAMES]), Shape::new(vec![1])],
                vec![
                    Tensor::new(
                        Shape::new(vec![1, ENC_DIM, ENC_LEN]),
                        TensorData::F32(vec![0.0; ENC_DIM * ENC_LEN]),
                    )
                    .unwrap(),
                    Tensor::new(Shape::new(vec![1]), TensorData::I64(vec![ENC_LEN as i64]))
                        .unwrap(),
                ],
            )),
        );
        sessions.insert(
            "v3_rnnt_decoder".into(),
            Arc::new(MockSession::new(
                vec![
                    Shape::new(vec![1, 1]),
                    Shape::new(vec![1, 1, PRED_HIDDEN]),
                    Shape::new(vec![1, 1, PRED_HIDDEN]),
                ],
                vec![
                    Tensor::new(
                        Shape::new(vec![1, 1, PRED_HIDDEN]),
                        TensorData::F32(vec![0.0; PRED_HIDDEN]),
                    )
                    .unwrap(),
                    Tensor::new(
                        Shape::new(vec![1, 1, PRED_HIDDEN]),
                        TensorData::F32(vec![0.0; PRED_HIDDEN]),
                    )
                    .unwrap(),
                    Tensor::new(
                        Shape::new(vec![1, 1, PRED_HIDDEN]),
                        TensorData::F32(vec![0.0; PRED_HIDDEN]),
                    )
                    .unwrap(),
                ],
            )),
        );
        sessions.insert(
            "v3_rnnt_joint".into(),
            Arc::new(
                MockSession::new(
                    vec![
                        Shape::new(vec![1, ENC_DIM, 1]),
                        Shape::new(vec![1, PRED_HIDDEN, 1]),
                    ],
                    vec![
                        Tensor::new(Shape::new(vec![1, 1, 2]), TensorData::F32(vec![0.0; 2]))
                            .unwrap(),
                    ],
                )
                .with_script(vec![
                    // First joiner call: token 0 ("▁hi") wins the argmax.
                    vec![
                        Tensor::new(Shape::new(vec![1, 1, 2]), TensorData::F32(vec![2.0, 0.0]))
                            .unwrap(),
                    ],
                    // All later calls: blank (id 1) wins.
                    vec![
                        Tensor::new(Shape::new(vec![1, 1, 2]), TensorData::F32(vec![0.0, 2.0]))
                            .unwrap(),
                    ],
                ]),
            ),
        );

        let factory = Box::new(MockFactory::new(sessions));
        let engine = Engine::load_with_factory(dir, None, 1, 1, 0, factory, 1)
            .expect("engine should load with mock runtime");
        (engine, tmp)
    }

    #[test]
    fn test_process_chunk_blank_endpoint_finalizes_without_vad() {
        // No VAD attached: the decoder's blank-run heuristic is the only
        // endpoint signal and finalizes the segment as before.
        let (engine, _tmp) = blank_run_engine();
        let mut guard = engine.pool.checkout_blocking().expect("checkout");
        let mut state = engine.create_state(false);
        let chunk = vec![0.0f32; 12800]; // one decode stride
        let segs = engine
            .process_chunk(&chunk, &mut state, &mut guard)
            .expect("mock decode must not error");
        assert_eq!(
            segs.len(),
            1,
            "blank-run endpoint must finalize the segment"
        );
        assert!(segs[0].is_final);
        assert!(segs[0].speech_final);
        assert_eq!(segs[0].endpoint_reason, Some(EndpointReason::Blank));
        assert_eq!(segs[0].text, "hi");
    }

    #[test]
    fn test_process_chunk_blank_endpoint_ignored_with_vad() {
        // With a VAD endpointer attached the VAD owns endpointing, so the
        // decoder's blank-run heuristic must NOT finalize the segment —
        // otherwise a `--vad-min-silence-ms` above ~600ms would be
        // unreachable (the decoder would cut the segment first).
        let (engine, _tmp) = blank_run_engine();
        let mut guard = engine.pool.checkout_blocking().expect("checkout");
        let mut state = engine.create_state(false);
        state.vad_endpointer = Some(crate::vad::VadEndpointer::new(
            &crate::vad::VadConfig::default(),
        ));
        let chunk = vec![0.0f32; 12800];
        let segs = engine
            .process_chunk(&chunk, &mut state, &mut guard)
            .expect("mock decode must not error");
        assert_eq!(segs.len(), 1, "decoded words still surface as a partial");
        assert!(
            !segs[0].is_final,
            "blank-run must not finalize while a VAD owns endpointing"
        );
        assert!(!segs[0].speech_final);
        assert!(segs[0].endpoint_reason.is_none());
        assert_eq!(segs[0].text, "hi");
    }

    /// Mock engine whose encoder input shape matches a full 2.5 s window
    /// (STREAM_MAX_WINDOW_SAMPLES → 249 mel frames), so we can hit `over_cap`
    /// in a single `process_chunk` without multi-stride buffering.
    fn blank_run_engine_window_cap() -> (Engine, tempfile::TempDir) {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path();
        std::fs::write(dir.join("v3_rnnt_encoder_int8.onnx"), b"").unwrap();
        std::fs::write(dir.join("v3_rnnt_decoder.onnx"), b"").unwrap();
        std::fs::write(dir.join("v3_rnnt_joint.onnx"), b"").unwrap();
        std::fs::write(dir.join("v3_vocab.txt"), "\u{2581}hi\n<blk>\n").unwrap();

        // (40000 - N_FFT) / HOP_LENGTH + 1 = 249 mel frames @ 16 kHz.
        const MEL_FRAMES: usize = 249;
        const ENC_LEN: usize = 16;

        let mut sessions: HashMap<String, Arc<MockSession>> = HashMap::new();
        sessions.insert(
            "v3_rnnt_encoder_int8".into(),
            Arc::new(MockSession::new(
                vec![Shape::new(vec![1, 64, MEL_FRAMES]), Shape::new(vec![1])],
                vec![
                    Tensor::new(
                        Shape::new(vec![1, ENC_DIM, ENC_LEN]),
                        TensorData::F32(vec![0.0; ENC_DIM * ENC_LEN]),
                    )
                    .unwrap(),
                    Tensor::new(Shape::new(vec![1]), TensorData::I64(vec![ENC_LEN as i64]))
                        .unwrap(),
                ],
            )),
        );
        sessions.insert(
            "v3_rnnt_decoder".into(),
            Arc::new(MockSession::new(
                vec![
                    Shape::new(vec![1, 1]),
                    Shape::new(vec![1, 1, PRED_HIDDEN]),
                    Shape::new(vec![1, 1, PRED_HIDDEN]),
                ],
                vec![
                    Tensor::new(
                        Shape::new(vec![1, 1, PRED_HIDDEN]),
                        TensorData::F32(vec![0.0; PRED_HIDDEN]),
                    )
                    .unwrap(),
                    Tensor::new(
                        Shape::new(vec![1, 1, PRED_HIDDEN]),
                        TensorData::F32(vec![0.0; PRED_HIDDEN]),
                    )
                    .unwrap(),
                    Tensor::new(
                        Shape::new(vec![1, 1, PRED_HIDDEN]),
                        TensorData::F32(vec![0.0; PRED_HIDDEN]),
                    )
                    .unwrap(),
                ],
            )),
        );
        sessions.insert(
            "v3_rnnt_joint".into(),
            Arc::new(
                MockSession::new(
                    vec![
                        Shape::new(vec![1, ENC_DIM, 1]),
                        Shape::new(vec![1, PRED_HIDDEN, 1]),
                    ],
                    vec![
                        Tensor::new(Shape::new(vec![1, 1, 2]), TensorData::F32(vec![0.0; 2]))
                            .unwrap(),
                    ],
                )
                .with_script(vec![
                    vec![
                        Tensor::new(Shape::new(vec![1, 1, 2]), TensorData::F32(vec![2.0, 0.0]))
                            .unwrap(),
                    ],
                    vec![
                        Tensor::new(Shape::new(vec![1, 1, 2]), TensorData::F32(vec![0.0, 2.0]))
                            .unwrap(),
                    ],
                ]),
            ),
        );
        let factory = Box::new(MockFactory::new(sessions));
        let engine = Engine::load_with_factory(dir, None, 1, 1, 0, factory, 1)
            .expect("engine should load with mock runtime");
        (engine, tmp)
    }

    #[test]
    fn test_process_chunk_window_cap_emits_partial_not_final() {
        // Hitting STREAM_MAX_WINDOW must commit a stable prefix and emit a
        // non-final partial — never speech_final. Voice assistants treat
        // every final as a command; cap-as-final was the Irene bug.
        let (engine, _tmp) = blank_run_engine_window_cap();
        let mut guard = engine.pool.checkout_blocking().expect("checkout");
        let mut state = engine.create_state(false);
        // Attach a VAD endpointer so the blank-run heuristic does not fire
        // first; the cap path is what we exercise.
        state.vad_endpointer = Some(crate::vad::VadEndpointer::new(
            &crate::vad::VadConfig::default(),
        ));
        // One chunk at the window cap (2.5 s @ 16 kHz).
        let chunk = vec![0.0f32; 16000 * 5 / 2];
        let segs = engine
            .process_chunk(&chunk, &mut state, &mut guard)
            .expect("mock decode must not error");
        assert_eq!(segs.len(), 1, "cap must still surface decoded text");
        assert!(
            !segs[0].is_final,
            "window cap must not emit is_final (got final text={:?})",
            segs[0].text
        );
        assert!(!segs[0].speech_final);
        assert!(segs[0].endpoint_reason.is_none());
        assert_eq!(segs[0].text, "hi");
        // Committed prefix survives after cap commit + slide.
        assert!(
            !state.assembler.is_empty(),
            "stable prefix must remain after cap commit"
        );
    }

    #[test]
    fn test_process_chunk_assistant_mode_ignores_blank_without_vad() {
        let (engine, _tmp) = blank_run_engine();
        let mut guard = engine.pool.checkout_blocking().expect("checkout");
        let mut state = engine.create_state(false);
        state.endpoint_mode = EndpointMode::Assistant;
        let chunk = vec![0.0f32; 12800];
        let segs = engine
            .process_chunk(&chunk, &mut state, &mut guard)
            .expect("mock decode must not error");
        assert_eq!(segs.len(), 1);
        assert!(
            !segs[0].is_final,
            "assistant mode must not finalize on blank-run alone"
        );
        assert_eq!(segs[0].text, "hi");
    }

    #[test]
    fn test_process_chunk_manual_mode_never_auto_finalizes() {
        let (engine, _tmp) = blank_run_engine();
        let mut guard = engine.pool.checkout_blocking().expect("checkout");
        let mut state = engine.create_state(false);
        state.endpoint_mode = EndpointMode::Manual;
        // Even with a synthetic VAD fire, Manual must not close.
        // We only have blank-run here; Manual ignores it too.
        let chunk = vec![0.0f32; 12800];
        let segs = engine
            .process_chunk(&chunk, &mut state, &mut guard)
            .expect("mock decode must not error");
        assert_eq!(segs.len(), 1);
        assert!(!segs[0].is_final);
    }

    #[test]
    fn test_flush_state_marks_stop_endpoint_reason() {
        let (engine, _tmp) = blank_run_engine();
        let mut state = engine.create_state(false);
        state
            .assembler
            .append(vec![WordInfo::new("bye", 0.0, 0.3, 0.9, None)]);
        let seg = engine.flush_state(&mut state).expect("flush");
        assert!(seg.is_final);
        assert!(seg.speech_final);
        assert_eq!(seg.endpoint_reason, Some(EndpointReason::Stop));
        assert_eq!(seg.text, "bye");
    }
}
