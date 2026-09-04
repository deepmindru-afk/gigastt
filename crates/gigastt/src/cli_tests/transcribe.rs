use super::*;

#[test]
fn test_cli_transcribe_encoder_intra_threads_flag() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _restore = EnvRestore(
        "GIGASTT_ENCODER_INTRA_THREADS",
        std::env::var("GIGASTT_ENCODER_INTRA_THREADS").ok(),
    );
    unsafe {
        std::env::remove_var("GIGASTT_ENCODER_INTRA_THREADS");
    }
    let cli = Cli::parse_from([
        "gigastt",
        "transcribe",
        "audio.wav",
        "--encoder-intra-threads",
        "3",
    ]);
    match cli.command {
        Commands::Transcribe {
            encoder_intra_threads,
            ..
        } => assert_eq!(encoder_intra_threads, Some(3)),
        _ => panic!("expected Transcribe"),
    }
}

#[test]
fn test_cli_transcribe_file_window_concurrency_default() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _restore = EnvRestore(
        "GIGASTT_FILE_WINDOW_CONCURRENCY",
        std::env::var("GIGASTT_FILE_WINDOW_CONCURRENCY").ok(),
    );
    unsafe {
        std::env::remove_var("GIGASTT_FILE_WINDOW_CONCURRENCY");
    }
    let cli = Cli::parse_from(["gigastt", "transcribe", "audio.wav"]);
    match cli.command {
        Commands::Transcribe {
            file_window_concurrency,
            ..
        } => assert_eq!(file_window_concurrency, 1),
        _ => panic!("expected Transcribe"),
    }
}

#[test]
fn test_cli_transcribe_file_window_concurrency_flag() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _restore = EnvRestore(
        "GIGASTT_FILE_WINDOW_CONCURRENCY",
        std::env::var("GIGASTT_FILE_WINDOW_CONCURRENCY").ok(),
    );
    unsafe {
        std::env::remove_var("GIGASTT_FILE_WINDOW_CONCURRENCY");
    }
    let cli = Cli::parse_from([
        "gigastt",
        "transcribe",
        "audio.wav",
        "--file-window-concurrency",
        "2",
    ]);
    match cli.command {
        Commands::Transcribe {
            file_window_concurrency,
            ..
        } => assert_eq!(file_window_concurrency, 2),
        _ => panic!("expected Transcribe"),
    }
}

#[test]
fn test_cli_transcribe_parsing() {
    let cli = Cli::parse_from(["gigastt", "transcribe", "audio.wav"]);
    match cli.command {
        Commands::Transcribe {
            file,
            model_variant,
            format,
            output,
            ..
        } => {
            assert_eq!(file, "audio.wav");
            // No --model-variant → None (auto-detect from disk).
            assert_eq!(model_variant, None);
            assert_eq!(format, "txt");
            assert!(output.is_none());
        }
        _ => panic!("expected Transcribe"),
    }
}

#[test]
fn test_cli_transcribe_format_and_output() {
    let cli = Cli::parse_from([
        "gigastt",
        "transcribe",
        "audio.wav",
        "--format",
        "srt",
        "-o",
        "out.srt",
    ]);
    match cli.command {
        Commands::Transcribe {
            file,
            format,
            output,
            ..
        } => {
            assert_eq!(file, "audio.wav");
            assert_eq!(format, "srt");
            assert_eq!(output, Some("out.srt".to_string()));
        }
        _ => panic!("expected Transcribe"),
    }
}

#[test]
fn test_cli_transcribe_subtitle_options() {
    let cli = Cli::parse_from([
        "gigastt",
        "transcribe",
        "audio.wav",
        "--format",
        "vtt",
        "--max-chars-per-line",
        "60",
        "--max-words-per-line",
        "10",
        "--word-timestamps",
    ]);
    match cli.command {
        Commands::Transcribe {
            format,
            max_chars_per_line,
            max_words_per_line,
            word_timestamps,
            ..
        } => {
            assert_eq!(format, "vtt");
            assert_eq!(max_chars_per_line, Some(60));
            assert_eq!(max_words_per_line, Some(10));
            assert!(word_timestamps);
        }
        _ => panic!("expected Transcribe"),
    }
}

#[test]
fn test_cli_transcribe_stereo_speakers_flag() {
    let cli = Cli::parse_from(["gigastt", "transcribe", "audio.wav", "--stereo-speakers"]);
    match cli.command {
        Commands::Transcribe {
            stereo_speakers, ..
        } => {
            assert!(stereo_speakers);
        }
        _ => panic!("expected Transcribe"),
    }
}

#[test]
fn test_cli_transcribe_stereo_speakers_defaults_off() {
    let cli = Cli::parse_from(["gigastt", "transcribe", "audio.wav"]);
    match cli.command {
        Commands::Transcribe {
            stereo_speakers, ..
        } => {
            assert!(!stereo_speakers);
        }
        _ => panic!("expected Transcribe"),
    }
}

#[test]
fn test_cli_transcribe_codec_flags() {
    let cli = Cli::parse_from([
        "gigastt",
        "transcribe",
        "call.ulaw",
        "--codec",
        "pcmu",
        "--sample-rate",
        "8000",
    ]);
    match cli.command {
        Commands::Transcribe {
            codec, sample_rate, ..
        } => {
            assert_eq!(codec.as_deref(), Some("pcmu"));
            assert_eq!(sample_rate, Some(8000));
        }
        _ => panic!("expected Transcribe"),
    }
}

#[test]
fn test_cli_transcribe_codec_requires_sample_rate() {
    // clap must reject `--codec` without `--sample-rate` before any engine
    // work happens.
    let result = Cli::try_parse_from(["gigastt", "transcribe", "call.ulaw", "--codec", "pcmu"]);
    assert!(result.is_err(), "--codec without --sample-rate must fail");
}

#[test]
fn test_cli_transcribe_sample_rate_alone_is_allowed() {
    // `--sample-rate` without `--codec` parses (it is simply unused), so
    // scripts can always append both flags uniformly.
    let cli = Cli::parse_from([
        "gigastt",
        "transcribe",
        "audio.wav",
        "--sample-rate",
        "8000",
    ]);
    match cli.command {
        Commands::Transcribe { codec, .. } => assert!(codec.is_none()),
        _ => panic!("expected Transcribe"),
    }
}

#[test]
fn test_cli_transcribe_punctuation_off() {
    let cli = Cli::parse_from(["gigastt", "transcribe", "a.wav", "--punctuation", "off"]);
    match cli.command {
        Commands::Transcribe { punctuation, .. } => {
            assert_eq!(punctuation, PunctuationMode::Off);
        }
        _ => panic!("expected Transcribe"),
    }
}

#[test]
fn test_cli_transcribe_itn_override() {
    let cli = Cli::parse_from(["gigastt", "transcribe", "a.wav", "--itn", "on"]);
    match cli.command {
        Commands::Transcribe { itn, .. } => assert_eq!(itn, ItnMode::On),
        _ => panic!("expected Transcribe"),
    }
}

#[test]
fn test_cli_transcribe_hotwords_flags() {
    let cli = Cli::parse_from([
        "gigastt",
        "transcribe",
        "a.wav",
        "--hotwords-file",
        "hw.txt",
    ]);
    match cli.command {
        Commands::Transcribe {
            hotwords_file,
            hotwords_default,
            ..
        } => {
            assert_eq!(hotwords_file, Some("hw.txt".to_string()));
            assert!(!hotwords_default);
        }
        _ => panic!("expected Transcribe"),
    }
}

#[test]
fn test_cli_transcribe_vad_and_itn_flags() {
    let _guard = ENV_LOCK.lock().unwrap();
    let restores: Vec<EnvRestore> = ["GIGASTT_VAD", "GIGASTT_ITN", "GIGASTT_VAD_THRESHOLD"]
        .iter()
        .map(|k| {
            let r = EnvRestore(k, std::env::var(k).ok());
            unsafe {
                std::env::remove_var(k);
            }
            r
        })
        .collect();
    let cli = Cli::parse_from([
        "gigastt",
        "transcribe",
        "a.wav",
        "--vad",
        "--vad-threshold",
        "0.6",
        "--itn",
        "off",
    ]);
    match cli.command {
        Commands::Transcribe {
            vad,
            vad_threshold,
            itn,
            ..
        } => {
            assert!(vad);
            assert_eq!(vad_threshold, Some(0.6));
            assert_eq!(itn, ItnMode::Off);
        }
        _ => panic!("expected Transcribe"),
    }
    drop(restores);
}
