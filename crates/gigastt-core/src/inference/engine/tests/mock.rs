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
            vec![Tensor::new(Shape::new(vec![1, 1, 2]), TensorData::F32(vec![0.0; 2])).unwrap()],
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
                Tensor::new(Shape::new(vec![1]), TensorData::I64(vec![ENC_LEN as i64])).unwrap(),
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
                    Tensor::new(Shape::new(vec![1, 1, 2]), TensorData::F32(vec![0.0; 2])).unwrap(),
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
                Tensor::new(Shape::new(vec![1]), TensorData::I64(vec![ENC_LEN as i64])).unwrap(),
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
                    Tensor::new(Shape::new(vec![1, 1, 2]), TensorData::F32(vec![0.0; 2])).unwrap(),
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
