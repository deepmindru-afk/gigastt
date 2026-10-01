use super::*;
use crate::inference::TranscriptSnapshot;
use crate::runtime::{
    RuntimeError,
    session::RuntimeSession,
    tensor::{Shape, Tensor, TensorData},
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

/// Trigger cancellation from inside a runtime call, after real tokens have
/// already reached the engine. No sleeps or scheduler timing assumptions.
struct AbortingJoiner {
    abort: Arc<AtomicBool>,
    calls: AtomicUsize,
    abort_after: usize,
}

impl RuntimeSession for AbortingJoiner {
    fn run(&self, _: &[Tensor]) -> Result<Vec<Tensor>, RuntimeError> {
        if self.calls.fetch_add(1, Ordering::Relaxed) + 1 == self.abort_after {
            self.abort.store(true, Ordering::Relaxed);
        }
        Ok(vec![Tensor::new_checked(
            Shape::new(vec![1, 1, 2]),
            TensorData::F32(vec![10.0, 0.0]),
        )])
    }
}

fn install_abort(triplet: &mut SessionTriplet, flag: &Arc<AtomicBool>, after: usize) {
    triplet.joiner = Some(Box::new(AbortingJoiner {
        abort: flag.clone(),
        calls: AtomicUsize::new(0),
        abort_after: after,
    }));
}

#[test]
fn test_cancelled_request_keeps_words_from_interrupted_decode() {
    // 800_000 samples is past the 30 s single-pass ceiling, so a cancelled
    // redecode must keep the previous window. That mock decode does not finish
    // inside the nightly Miri budget; the short case still checks the partial.
    let cases: &[(usize, usize)] = if cfg!(miri) {
        &[(8_000, 3)]
    } else {
        &[(8_000, 3), (800_000, 14)]
    };
    for &(samples, after) in cases {
        let (engine, _tmp) = crate::test_support::rnnt_engine();
        let flag = Arc::new(AtomicBool::new(false));
        let partial = Arc::new(TranscriptSnapshot::default());
        let mut guard = engine.pool.checkout_blocking().unwrap();
        install_abort(&mut guard, &flag, after);
        let audio = vec![0.0; samples];
        let request = TranscribeRequest::new(TranscribeSource::Samples(&audio))
            .with_abort(Some(flag))
            .with_partial(Some(partial.clone()));
        assert!(matches!(
            engine.transcribe_request(request, &mut guard),
            Err(GigasttError::Cancelled)
        ));
        let snapshot = partial.get().expect("readable partial after cancellation");
        assert!(!snapshot.text.is_empty());
        assert!(snapshot.words.len() >= 3);
        if samples > 8000 {
            assert!(
                snapshot.words.first().unwrap().start < 1.0,
                "previous window was lost"
            );
        }
        assert!(!snapshot.is_final);
        drop(guard);
        assert!(engine.pool.try_checkout().unwrap().is_some());
    }
}

#[test]
fn test_cancelled_stream_is_terminal_and_partial_stays_readable() {
    let (engine, _tmp) = crate::test_support::rnnt_engine();
    let flag = Arc::new(AtomicBool::new(false));
    let partial = Arc::new(TranscriptSnapshot::default());
    let mut guard = engine.pool.checkout_blocking().unwrap();
    install_abort(&mut guard, &flag, 3);
    let mut state = engine.create_state(false);
    state.abort = Some(flag.clone());
    state.partial = Some(partial.clone());
    assert!(matches!(
        engine.process_chunk(&[0.0; 16000], &mut state, &mut guard),
        Err(GigasttError::Cancelled)
    ));
    assert!(state.is_failed());
    let text = partial.get().unwrap().text;
    assert!(!text.is_empty());
    assert_eq!(state.assembler.partial(0.0).text, text);
    flag.store(false, Ordering::Relaxed);
    assert!(matches!(
        engine.process_chunk(&[0.0; 16000], &mut state, &mut guard),
        Err(GigasttError::Cancelled)
    ));
    assert_eq!(
        engine.finish_stream(&mut state, &mut guard).unwrap().text,
        text
    );
    assert_eq!(engine.flush_state(&mut state).unwrap().text, text);
    assert_eq!(state.assembler.partial(0.0).text, text);
}

#[test]
fn test_cancelled_redecode_keeps_previous_live_and_committed_words() {
    let (engine, _tmp) = crate::test_support::rnnt_engine();
    let flag = Arc::new(AtomicBool::new(false));
    let mut guard = engine.pool.checkout_blocking().unwrap();
    install_abort(&mut guard, &flag, 1);
    let mut state = engine.create_state(false);
    state.abort = Some(flag);
    state.assembler.append(vec![word("already", 0.0, 0.1)]);
    state.assembler.commit_live();
    state
        .assembler
        .append(vec![word("readable", 0.1, 0.2), word("tail", 0.2, 0.3)]);
    assert!(matches!(
        engine.process_chunk(&[0.0; 16000], &mut state, &mut guard),
        Err(GigasttError::Cancelled)
    ));
    assert_eq!(state.assembler.partial(0.0).text, "already readable tail");
}

#[test]
fn test_cancelled_second_channel_keeps_first_channel_text() {
    let (engine, _tmp) = crate::test_support::rnnt_engine();
    let flag = Arc::new(AtomicBool::new(false));
    let partial = Arc::new(TranscriptSnapshot::default());
    let mut guard = engine.pool.checkout_blocking().unwrap();
    install_abort(&mut guard, &flag, 14);
    let channels = vec![vec![0.0; 16000]; 2];
    let request = TranscribeRequest::new(TranscribeSource::Channels(&channels))
        .with_abort(Some(flag))
        .with_partial(Some(partial.clone()));
    assert!(matches!(
        engine.transcribe_request(request, &mut guard),
        Err(GigasttError::Cancelled)
    ));
    let words = partial.get().unwrap().words;
    assert!(words.iter().any(|word| word.speaker == Some(0)));
    assert!(words.iter().any(|word| word.speaker == Some(1)));
}

#[test]
#[cfg(feature = "diarization")]
fn test_cancel_after_recognition_skips_lazy_speaker_load() {
    let (mut engine, tmp) = crate::test_support::rnnt_engine();
    std::fs::write(tmp.path().join("wespeaker_resnet34.onnx"), b"invalid model").unwrap();
    engine.speaker_encoder = super::super::diarization::probe_speaker_encoder(tmp.path());
    let mut guard = engine.pool.checkout_blocking().unwrap();
    let flag = AtomicBool::new(false);
    let abort = || flag.load(Ordering::Relaxed);
    let report = |_| flag.store(true, Ordering::Relaxed);
    let outcome = std::sync::OnceLock::new();
    let result = engine.transcribe_samples_with_overrides(
        &[0.0; 320],
        &mut guard,
        &TranscribeOverrides::default(),
        None,
        true,
        Some(&outcome),
        DecodeControls {
            abort: Some(&abort),
            on_progress: Some(&report),
            ..Default::default()
        },
    );
    assert!(matches!(result, Err(GigasttError::Cancelled)));
    assert!(outcome.get().is_none());
    // A failed load is cached permanently; pending proves the load never began.
    assert!(engine.speaker_encoder.as_ref().unwrap().is_pending());
}

#[test]
fn test_cancelled_window_decode_does_not_pull_source() {
    struct UnreadSource;
    impl PcmWindows for UnreadSource {
        fn next_window(&mut self) -> Result<Option<PcmWindow<'_>>, GigasttError> {
            panic!("cancelled request pulled a new source window")
        }
    }
    for concurrency in [1, 2] {
        let (engine, _tmp) = crate::test_support::rnnt_engine();
        let engine = engine.with_file_window_concurrency(concurrency);
        let mut guard = engine.pool.checkout_blocking().unwrap();
        let result = engine.decode_words_streaming(
            &mut UnreadSource,
            &mut guard,
            None,
            DecodeControls {
                abort: Some(&|| true),
                ..Default::default()
            },
        );
        assert!(matches!(result, Err(GigasttError::Cancelled)));
    }
}

#[test]
fn test_postprocess_cancellation_between_stages_rejects_final_text() {
    let (engine, _tmp) = crate::test_support::rnnt_engine();
    let checks = AtomicUsize::new(0);
    let abort = || checks.fetch_add(1, Ordering::Relaxed) >= 1;
    let result = engine.apply_text_postprocess(
        "двадцать один".into(),
        true,
        true,
        DecodeControls {
            abort: Some(&abort),
            ..Default::default()
        },
    );
    assert!(matches!(result, Err(GigasttError::Cancelled)));
    assert_eq!(checks.load(Ordering::Relaxed), 2);
}
