use super::*;
use crate::runtime::{RuntimeError, RuntimeSession, Tensor};

struct FailingTail {
    panic: bool,
}
impl RuntimeSession for FailingTail {
    fn run(&self, _: &[Tensor]) -> Result<Vec<Tensor>, RuntimeError> {
        assert!(!self.panic, "injected final decode panic");
        Err(RuntimeError::InferenceFailed(
            "injected final decode failure".into(),
        ))
    }
}

#[test]
fn test_finish_stream_tail_failure_never_returns_successful_final() {
    for encoder in [false, true] {
        let (engine, _tmp) = crate::test_support::rnnt_engine();
        let mut guard = engine.pool.checkout_blocking().unwrap();
        let mut state = engine.create_state(false);
        state.assembler.append(vec![word("retained", 0.0, 0.1)]);
        assert!(
            engine
                .process_chunk(&[0.0; 1600], &mut state, &mut guard)
                .unwrap()
                .is_empty()
        );
        if encoder {
            guard.encoder = Box::new(FailingTail { panic: false });
        } else {
            guard.joiner = Some(Box::new(FailingTail { panic: false }));
        }
        let partial = engine.finish_stream(&mut state, &mut guard).unwrap();
        assert!(
            !partial.is_final,
            "tail failure must not become a successful final"
        );
        assert_eq!(partial.text, "retained");
        assert!(state.is_failed());
    }
}

#[test]
fn test_try_finish_stream_tail_failures_keep_partial_and_poison_state() {
    for encoder in [false, true] {
        for prior_text in [false, true] {
            for panic in [false, true] {
                let (engine, _tmp) = crate::test_support::rnnt_engine();
                let mut guard = engine.pool.checkout_blocking().unwrap();
                let mut state = engine.create_state(false);
                let snapshot = std::sync::Arc::new(crate::inference::TranscriptSnapshot::default());
                state.partial = Some(snapshot.clone());
                if prior_text {
                    state.assembler.append(vec![word("retained", 0.0, 0.1)]);
                }
                assert!(
                    engine
                        .process_chunk(&[0.0; 1600], &mut state, &mut guard)
                        .unwrap()
                        .is_empty()
                );
                if encoder {
                    guard.encoder = Box::new(FailingTail { panic });
                } else {
                    guard.joiner = Some(Box::new(FailingTail { panic }));
                }
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    engine.try_finish_stream(&mut state, &mut guard)
                }));
                if panic {
                    assert!(result.is_err());
                } else {
                    assert!(matches!(
                        result.unwrap(),
                        Err(GigasttError::Inference { .. })
                    ));
                }
                assert!(state.is_failed());
                let partial = engine.flush_state(&mut state);
                if prior_text {
                    let partial = partial.unwrap();
                    assert_eq!(partial.text, "retained");
                    assert!(!partial.is_final);
                    assert_eq!(snapshot.get().unwrap().text, "retained");
                } else {
                    assert!(partial.is_none());
                }
                assert!(matches!(
                    engine.try_finish_stream(&mut state, &mut guard),
                    Err(GigasttError::Cancelled)
                ));
                assert!(matches!(
                    engine.process_chunk(&[0.0; 16000], &mut state, &mut guard),
                    Err(GigasttError::Cancelled)
                ));
            }
        }
    }
}

#[test]
fn test_successful_finish_does_not_redecode_tail_and_remains_reusable() {
    let (engine, _tmp) = crate::test_support::rnnt_engine();
    let mut guard = engine.pool.checkout_blocking().unwrap();
    let mut state = engine.create_state(false);
    engine
        .process_chunk(&[0.0; 1600], &mut state, &mut guard)
        .unwrap();
    engine.try_finish_stream(&mut state, &mut guard).unwrap();
    assert!(!state.is_failed());
    assert_eq!(state.pending_samples, 0);
    let encoder = std::mem::replace(&mut guard.encoder, Box::new(FailingTail { panic: true }));
    assert!(
        engine
            .try_finish_stream(&mut state, &mut guard)
            .unwrap()
            .is_none()
    );
    guard.encoder = encoder;
    assert!(
        engine
            .process_chunk(&[0.0; 16000], &mut state, &mut guard)
            .is_ok()
    );
}

#[test]
fn test_successful_finish_keeps_sub_frame_audio_for_next_chunk() {
    let (engine, _tmp) = crate::test_support::rnnt_engine();
    let mut guard = engine.pool.checkout_blocking().unwrap();
    let mut state = engine.create_state(false);
    engine
        .process_chunk(&[0.25; N_FFT / 2], &mut state, &mut guard)
        .unwrap();
    assert!(
        engine
            .try_finish_stream(&mut state, &mut guard)
            .unwrap()
            .is_none()
    );
    assert_eq!(state.pending_samples, N_FFT / 2);
    assert_eq!(state.context_samples, 0);
    assert_eq!(state.audio_buffer, vec![0.25; N_FFT / 2]);
    engine
        .process_chunk(&[0.5; N_FFT / 2], &mut state, &mut guard)
        .unwrap();
    guard.encoder = Box::new(FailingTail { panic: false });
    assert!(matches!(
        engine.try_finish_stream(&mut state, &mut guard),
        Err(GigasttError::Inference { .. })
    ));
    assert_eq!(state.audio_buffer.len(), N_FFT);
}

#[test]
fn test_successful_finish_then_new_chunk_emits_only_new_words() {
    use crate::runtime::{Shape, TensorData};
    let tmp = tempfile::tempdir().unwrap();
    crate::test_support::write_rnnt_layout(tmp.path()).unwrap();
    let engine = crate::test_support::load_rnnt_engine_with_factory(
        tmp.path(),
        1,
        Box::new(crate::test_support::FailingStreamFactory::new(false, false)),
    )
    .unwrap();
    let mut guard = engine.pool.checkout_blocking().unwrap();
    let mut state = engine.create_state(false);
    engine
        .process_chunk(&[0.0; 1600], &mut state, &mut guard)
        .unwrap();
    let first = engine
        .try_finish_stream(&mut state, &mut guard)
        .unwrap()
        .unwrap();
    assert!(first.is_final);
    assert!(first.text.contains("hi"));
    // A longer next window contains both the old context and new words.
    guard.encoder = Box::new(crate::runtime::mock::MockSession::unconstrained(vec![
        Tensor::new(
            Shape::new(vec![1, 768, 32]),
            TensorData::F32(vec![0.0; 768 * 32]),
        )
        .unwrap(),
        Tensor::new(Shape::new(vec![1]), TensorData::I64(vec![32])).unwrap(),
    ]));
    engine
        .process_chunk(&[0.0; 1600], &mut state, &mut guard)
        .unwrap();
    let second = engine
        .try_finish_stream(&mut state, &mut guard)
        .unwrap()
        .unwrap();
    assert!(second.is_final);
    assert!(second.text.contains("hi"));
    assert!(!second.words.is_empty());
    assert!(
        second.words.iter().all(|word| word.start >= 0.1),
        "{:?}",
        second.words
    );
    assert!(
        engine
            .try_finish_stream(&mut state, &mut guard)
            .unwrap()
            .is_none()
    );
}
