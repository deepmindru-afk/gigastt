use super::*;

#[test]
fn test_cli_transcribe_batch_defaults() {
    let _guard = ENV_LOCK.lock().unwrap();
    let restores: Vec<EnvRestore> = ["GIGASTT_FORMAT", "GIGASTT_BATCH_RETRIES"]
        .iter()
        .map(|k| {
            let r = EnvRestore(k, std::env::var(k).ok());
            unsafe {
                std::env::remove_var(k);
            }
            r
        })
        .collect();
    let cli = Cli::parse_from(["gigastt", "transcribe-batch", "samples/", "out/"]);
    match cli.command {
        Commands::TranscribeBatch {
            input_dir,
            output_dir,
            engine,
            output,
        } => {
            assert_eq!(input_dir, "samples/");
            assert_eq!(output_dir, "out/");
            assert_eq!(engine.pool_size, 2);
            assert_eq!(engine.model_variant, None);
            assert_eq!(output.format, "txt,json");
            assert_eq!(output.move_to, None);
            assert!(!output.delete_source);
            assert_eq!(output.retries, None);
        }
        _ => panic!("expected TranscribeBatch"),
    }
    drop(restores);
}

#[test]
fn test_cli_transcribe_batch_flags() {
    let cli = Cli::parse_from([
        "gigastt",
        "transcribe-batch",
        "in/",
        "out/",
        "--format",
        "md,srt",
        "--move-to",
        "in/done",
        "--pool-size",
        "4",
        "--retries",
        "1",
    ]);
    match cli.command {
        Commands::TranscribeBatch { engine, output, .. } => {
            assert_eq!(engine.pool_size, 4);
            assert_eq!(output.format, "md,srt");
            assert_eq!(output.move_to, Some("in/done".to_string()));
            assert_eq!(output.retries, Some(1));
        }
        _ => panic!("expected TranscribeBatch"),
    }
}

#[test]
fn test_cli_transcribe_batch_move_to_conflicts_with_delete_source() {
    let res = Cli::try_parse_from([
        "gigastt",
        "transcribe-batch",
        "in/",
        "out/",
        "--move-to",
        "done/",
        "--delete-source",
    ]);
    assert!(res.is_err(), "--move-to and --delete-source must conflict");
}

#[test]
fn test_cli_watch_defaults() {
    let _guard = ENV_LOCK.lock().unwrap();
    let restores: Vec<EnvRestore> = [
        "GIGASTT_WATCH_POLL_INTERVAL_MS",
        "GIGASTT_WATCH_SETTLE_POLLS",
        "GIGASTT_FORMAT",
    ]
    .iter()
    .map(|k| {
        let r = EnvRestore(k, std::env::var(k).ok());
        unsafe {
            std::env::remove_var(k);
        }
        r
    })
    .collect();
    let cli = Cli::parse_from(["gigastt", "watch", "in/", "out/"]);
    match cli.command {
        Commands::Watch {
            input_dir,
            output_dir,
            poll_interval_ms,
            settle_polls,
            engine,
            output,
        } => {
            assert_eq!(input_dir, "in/");
            assert_eq!(output_dir, "out/");
            assert_eq!(poll_interval_ms, 1000);
            assert_eq!(settle_polls, 2);
            assert_eq!(engine.pool_size, 2);
            assert_eq!(output.format, "txt,json");
        }
        _ => panic!("expected Watch"),
    }
    drop(restores);
}

#[test]
fn test_cli_watch_flags() {
    let cli = Cli::parse_from([
        "gigastt",
        "watch",
        "in/",
        "out/",
        "--poll-interval-ms",
        "250",
        "--settle-polls",
        "4",
        "--delete-source",
    ]);
    match cli.command {
        Commands::Watch {
            poll_interval_ms,
            settle_polls,
            output,
            ..
        } => {
            assert_eq!(poll_interval_ms, 250);
            assert_eq!(settle_polls, 4);
            assert!(output.delete_source);
        }
        _ => panic!("expected Watch"),
    }
}
