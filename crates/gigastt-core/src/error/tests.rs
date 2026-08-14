use super::*;

#[test]
fn test_error_code_maps_variants() {
    assert_eq!(
        GigasttError::Inference {
            source: "boom".into()
        }
        .code(),
        "inference_error"
    );
    assert_eq!(
        GigasttError::InvalidAudio {
            reason: "bad".into()
        }
        .code(),
        "invalid_audio"
    );
    assert_eq!(
        GigasttError::ModelLoad {
            path: "x".into(),
            source: None
        }
        .code(),
        "model_load_error"
    );
    assert_eq!(
        GigasttError::Io(std::io::Error::other("x")).code(),
        "io_error"
    );
    assert_eq!(
        GigasttError::InvalidInput {
            message: "bad format".into()
        }
        .code(),
        "invalid_input"
    );
    assert_eq!(GigasttError::Cancelled.code(), "cancelled");
    assert_eq!(
        GigasttError::AudioTooLong {
            observed_secs: 4000.0,
            limit_secs: 1800.0,
        }
        .code(),
        "audio_too_long"
    );
}

#[test]
fn test_cancelled_display() {
    assert_eq!(GigasttError::Cancelled.to_string(), "cancelled");
}

#[test]
fn test_audio_too_long_display_rounds_seconds() {
    let e = GigasttError::AudioTooLong {
        observed_secs: 3661.4,
        limit_secs: 1800.0,
    };
    assert_eq!(
        e.to_string(),
        "audio too long: 3661s exceeds the maximum of 1800s"
    );
}

#[test]
fn test_audio_too_long_survives_anyhow_downcast() {
    // The decode layer bails through `anyhow`; the engine seam downcasts the
    // typed variant back out. This guards that round-trip.
    let err: anyhow::Error = GigasttError::AudioTooLong {
        observed_secs: 5000.0,
        limit_secs: 1800.0,
    }
    .into();
    match err.downcast::<GigasttError>() {
        Ok(GigasttError::AudioTooLong { limit_secs, .. }) => {
            assert_eq!(limit_secs, 1800.0);
        }
        other => panic!("expected AudioTooLong, got {other:?}"),
    }
}

#[test]
fn test_display_invalid_input() {
    let e = GigasttError::InvalidInput {
        message: "unsupported format".into(),
    };
    assert_eq!(e.to_string(), "invalid input: unsupported format");
}

#[test]
fn test_model_path_rejects_empty() {
    assert!(ModelPath::new("").is_err());
}

#[test]
fn test_model_path_accepts_valid() {
    let p = ModelPath::new("encoder.onnx").unwrap();
    assert_eq!(p.as_str(), "encoder.onnx");
}

#[test]
fn test_reason_rejects_empty() {
    assert!(Reason::new("").is_err());
}

#[test]
fn test_reason_accepts_valid() {
    let r = Reason::new("too long").unwrap();
    assert_eq!(r.as_str(), "too long");
}

#[test]
fn test_display_model_load() {
    let e = GigasttError::ModelLoad {
        path: "encoder.onnx".into(),
        source: Some(Box::new(std::io::Error::other("missing weights"))),
    };
    assert!(e.to_string().contains("encoder.onnx"));
}

#[test]
fn test_display_inference() {
    let e = GigasttError::Inference {
        source: Box::new(std::io::Error::other("decoder failed")),
    };
    assert_eq!(e.to_string(), "inference failed");
}

#[test]
fn test_display_invalid_audio() {
    let e = GigasttError::InvalidAudio {
        reason: "too long".into(),
    };
    assert_eq!(e.to_string(), "invalid audio: too long");
}

#[test]
fn test_display_io() {
    let e = GigasttError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "gone"));
    assert!(e.to_string().contains("gone"));
}

#[test]
fn test_from_io_error() {
    let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
    let e: GigasttError = io_err.into();
    assert!(matches!(e, GigasttError::Io(_)));
}

#[test]
fn test_error_source_io() {
    let e = GigasttError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "x"));
    assert!(std::error::Error::source(&e).is_none());
}

#[test]
fn test_into_anyhow() {
    // Verify GigasttError works with ? in anyhow::Result contexts
    fn returns_anyhow() -> anyhow::Result<()> {
        Err(GigasttError::Inference {
            source: Box::new(std::io::Error::other("test")),
        })?;
        Ok(())
    }
    assert!(returns_anyhow().is_err());
}
