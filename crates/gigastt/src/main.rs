use clap::{CommandFactory, FromArgMatches, Parser, Subcommand};
use gigastt::boot::{ItnMode, PunctuationMode, parse_itn_mode, parse_punctuation_mode};
use gigastt_core::model;
use gigastt_core::model::{ModelVariant, ProgressMode};
use tracing_subscriber::EnvFilter;

mod packaging;
mod serve;
mod transcribe_cmd;
use packaging::{run_cache_gc, run_download, run_quantize};
use serve::{ServeArgs, run_serve};
use transcribe_cmd::{
    BatchOutputArgs, OfflineEngineArgs, run_transcribe, run_transcribe_batch, run_watch,
};

#[derive(Parser)]
#[command(
    name = "gigastt",
    version,
    about = "Local STT server powered by GigaAM v3",
    after_long_help = "Engine and post-processing options (--model-variant, --punctuation, --itn, --vad, ...) are defined on the subcommands, not at the top level.\nSee `gigastt serve --help` or `gigastt transcribe --help` for the full list."
)]
struct Cli {
    /// Log level [default: info]
    #[arg(long, global = true, default_value = "info")]
    log_level: String,

    /// Air-gapped mode: refuse every network fetch (model download, punctuation /
    /// diarization / VAD auto-fetch) with an instruction naming the missing file
    /// instead of a connect timeout. Equivalent to GIGASTT_OFFLINE=1.
    #[arg(long, global = true)]
    offline: bool,

    #[command(subcommand)]
    command: Commands,
}

#[allow(clippy::large_enum_variant)]
#[derive(Subcommand)]
enum Commands {
    /// Start WebSocket STT server (auto-downloads model if missing)
    Serve(ServeArgs),

    /// Download model without starting server
    Download {
        /// Model directory
        #[arg(long, default_value_t = model::default_model_dir())]
        model_dir: String,

        /// Recognition head to download: `rnnt` (default — lower WER, bare
        /// lowercase), `e2e_rnnt` (punctuation / casing / ITN), or `ml_ctc` /
        /// `ml_ctc_large` (GigaAM Multilingual charwise CTC, 220M / 600M —
        /// ru/en/kk/ky/uz, pre-quantized INT8 fetched directly).
        #[arg(
            long,
            env = "GIGASTT_MODEL_VARIANT",
            default_value = "rnnt",
            value_parser = parse_model_variant
        )]
        model_variant: ModelVariant,

        /// Skip downloading the speaker diarization model
        #[cfg(feature = "diarization")]
        #[arg(long, default_value_t = false)]
        skip_diarization: bool,

        /// Progress reporting format: `human` (default — interactive `\r`
        /// progress on stderr) or `json` (NDJSON events on stdout, one object
        /// per line, for sidecar integrators; human progress and tracing logs
        /// stay off stdout in this mode).
        #[arg(
            long,
            env = "GIGASTT_DOWNLOAD_PROGRESS",
            default_value = "human",
            value_parser = parse_progress_mode
        )]
        progress: ProgressMode,

        /// Also fetch the per-bucket palettized ANE (Core ML) encoder packages
        /// into `~/.gigastt/models/ane/` for the macOS Neural Engine backend.
        /// Requires a published ANE release.
        #[cfg(feature = "ane")]
        #[arg(long, default_value_t = false)]
        ane: bool,
    },

    /// Packaging: quantize a local **FP32** encoder ONNX to INT8.
    /// Runtime inference never uses FP32 — prefer `gigastt download` for INT8.
    Quantize {
        /// Model directory holding the FP32 encoder source
        #[arg(long, default_value_t = model::default_model_dir())]
        model_dir: String,

        /// Force re-quantization even if INT8 model exists
        #[arg(long)]
        force: bool,
    },

    /// Prune stale ONNX Runtime optimized graphs and stale CoreML compiled-model
    /// caches (and optionally hardlink exact duplicate files) under the model
    /// directory. Reclaims disk on multi-head / FP32-polluted installs and after
    /// an ONNX Runtime upgrade, without changing accuracy.
    CacheGc {
        /// Model directory
        #[arg(long, default_value_t = model::default_model_dir())]
        model_dir: String,

        /// Report reclaimable files without deleting or hardlinking
        #[arg(long, default_value_t = false)]
        dry_run: bool,

        /// Also hardlink content-identical files under the model dir
        /// (SHA-256 groups). Off by default — optimized_cache prune always runs.
        #[arg(long, default_value_t = false)]
        dedupe: bool,
    },

    /// Transcribe an audio file (offline)
    Transcribe {
        /// Path to audio file (WAV, M4A, MP3, OGG, FLAC)
        file: String,

        /// Model directory
        #[arg(long, default_value_t = model::default_model_dir())]
        model_dir: String,

        /// Recognition head to use. Omit to auto-detect from the model
        /// directory (existing install used as-is; only downloads if empty).
        /// `rnnt` (lower WER, bare lowercase), `e2e_rnnt` (punctuation /
        /// casing / ITN), or `ml_ctc` / `ml_ctc_large` (GigaAM Multilingual
        /// charwise CTC, 220M / 600M — ru/en/kk/ky/uz, bare lowercase).
        /// Env: GIGASTT_MODEL_VARIANT.
        #[arg(
            long,
            env = "GIGASTT_MODEL_VARIANT",
            value_parser = parse_model_variant
        )]
        model_variant: Option<ModelVariant>,

        /// Punctuation + capitalization restoration: `on`, `off`, or `auto`.
        /// `auto` (default) enables it for `rnnt`, disables it for `e2e_rnnt`.
        /// Env: GIGASTT_PUNCTUATION.
        #[arg(
            long,
            env = "GIGASTT_PUNCTUATION",
            default_value = "auto",
            value_parser = parse_punctuation_mode
        )]
        punctuation: PunctuationMode,

        /// Directory holding the optional punctuation model. Defaults to
        /// `~/.gigastt/models/punct/`. Auto-downloaded from
        /// `ekhodzitsky/rupunct-small-onnx` when enabled and absent.
        /// Env: GIGASTT_PUNCT_MODEL_DIR.
        #[arg(
            long,
            env = "GIGASTT_PUNCT_MODEL_DIR",
            default_value_t = model::default_punct_model_dir()
        )]
        punct_model_dir: String,

        /// Inverse text normalization (Russian number-words → digits):
        /// `on`, `off`, or `auto`. `auto` (default) enables it for `rnnt`,
        /// disables it for `e2e_rnnt`. Runs before punctuation. Env: GIGASTT_ITN.
        #[arg(
            long,
            env = "GIGASTT_ITN",
            default_value = "auto",
            value_parser = parse_itn_mode
        )]
        itn: ItnMode,

        /// Contextual hotword biasing: path to a file of phrases to boost during
        /// recognition (one phrase per line, optional `\t<weight>` suffix; blank
        /// lines and `#` comments ignored). Off when unset. Env:
        /// GIGASTT_HOTWORDS_FILE.
        #[arg(long, env = "GIGASTT_HOTWORDS_FILE")]
        hotwords_file: Option<String>,

        /// Also bias the built-in Russian brand/acronym lexicon. Combined with
        /// any `--hotwords-file` phrases. Env: GIGASTT_HOTWORDS_DEFAULT.
        #[arg(long, env = "GIGASTT_HOTWORDS_DEFAULT", default_value_t = false)]
        hotwords_default: bool,

        /// Additive logit boost applied to hotword continuation tokens during
        /// greedy decode [default: 5.0]. Higher = stronger bias. No effect
        /// unless hotwords are configured. Env: GIGASTT_HOTWORDS_BOOST.
        #[arg(long, env = "GIGASTT_HOTWORDS_BOOST")]
        hotwords_boost: Option<f32>,

        /// Voice activity detection: skip silence before decoding. Off by
        /// default; downloads the Silero VAD model (MIT) on first use. Env:
        /// GIGASTT_VAD.
        #[arg(long, env = "GIGASTT_VAD", default_value_t = false)]
        vad: bool,

        /// VAD speech-probability threshold in [0,1] [default: 0.5]. Higher =
        /// stricter. No effect unless `--vad`. Env: GIGASTT_VAD_THRESHOLD.
        #[arg(long, env = "GIGASTT_VAD_THRESHOLD")]
        vad_threshold: Option<f32>,

        /// Minimum trailing silence (ms) to close a speech region [default: 500].
        /// No effect unless `--vad`. Env: GIGASTT_VAD_MIN_SILENCE_MS.
        #[arg(long, env = "GIGASTT_VAD_MIN_SILENCE_MS")]
        vad_min_silence_ms: Option<u32>,

        /// Directory holding the Silero VAD model (`silero_vad.onnx`). Defaults
        /// to `~/.gigastt/models/vad/`. Auto-downloaded when `--vad` is set and
        /// the model is absent. Env: GIGASTT_VAD_MODEL_DIR.
        #[arg(long, env = "GIGASTT_VAD_MODEL_DIR", default_value_t = model::default_vad_model_dir())]
        vad_model_dir: String,

        /// Intra-op thread count for the encoder session on the CPU build. The
        /// encoder dominates the single-utterance cost, so more threads speed up
        /// long single-file jobs on weak CPUs. When unset, defaults to the logical
        /// CPU count (offline transcription runs a single triplet). Do not set
        /// `1` on multi-core hosts unless debugging — it is ~3× slower than auto.
        /// An explicit value (flag or env, including `1`) is still honoured as-is
        /// for debug passthrough. No effect on CoreML / CUDA builds.
        #[arg(long, env = "GIGASTT_ENCODER_INTRA_THREADS")]
        encoder_intra_threads: Option<usize>,

        /// Export format: json, txt, srt, vtt, md [default: txt]
        #[arg(short, long, env = "GIGASTT_FORMAT", default_value = "txt")]
        format: String,

        /// Output file. When omitted, prints to stdout.
        #[arg(short, long, env = "GIGASTT_OUTPUT")]
        output: Option<String>,

        /// Maximum characters per subtitle/caption line (SRT/VTT) [default: 80]
        #[arg(long, env = "GIGASTT_MAX_CHARS_PER_LINE")]
        max_chars_per_line: Option<usize>,

        /// Maximum words per subtitle/caption line (SRT/VTT) [default: 14]
        #[arg(long, env = "GIGASTT_MAX_WORDS_PER_LINE")]
        max_words_per_line: Option<usize>,

        /// Include per-word timestamps in Markdown output
        #[arg(long, env = "GIGASTT_WORD_TIMESTAMPS", default_value_t = false)]
        word_timestamps: bool,

        /// Transcribe left/right channels as separate speakers (channel 0 = speaker_0,
        /// channel 1 = speaker_1). Falls back to mono for mono files, dual-mono stereo
        /// files, and files with more than two channels. Env: GIGASTT_STEREO_SPEAKERS.
        #[arg(long, env = "GIGASTT_STEREO_SPEAKERS", default_value_t = false)]
        stereo_speakers: bool,

        /// Raw headerless telephony codec of the input file: `pcmu` (alias
        /// `ulaw`), `pcma` (alias `alaw`), or `g722`. When set, the file is
        /// decoded as a raw byte stream (RTP dump, Asterisk Monitor raw)
        /// instead of sniffing a container. Requires `--sample-rate`.
        /// Env: GIGASTT_CODEC.
        #[arg(long, env = "GIGASTT_CODEC", requires = "sample_rate")]
        codec: Option<String>,

        /// Sample rate (Hz) of a raw `--codec` input (typical telephony: 8000).
        /// G.722 decodes to its native 16 kHz; both 8000 (the SDP clock-rate
        /// convention) and 16000 are accepted for it. Env: GIGASTT_SAMPLE_RATE.
        #[arg(long, env = "GIGASTT_SAMPLE_RATE")]
        sample_rate: Option<u32>,
    },

    /// Transcribe every audio file in a directory (offline, one-shot)
    TranscribeBatch {
        /// Directory scanned recursively for audio files (WAV, MP3, M4A, OGG, FLAC)
        input_dir: String,

        /// Directory the `<stem>.<ext>` transcripts are written into
        output_dir: String,

        #[command(flatten)]
        engine: OfflineEngineArgs,

        #[command(flatten)]
        output: BatchOutputArgs,
    },

    /// Watch a directory and transcribe new/changed audio files as they appear
    Watch {
        /// Directory polled for audio files (WAV, MP3, M4A, OGG, FLAC). Files
        /// already present at startup are registered but not transcribed.
        input_dir: String,

        /// Directory the `<stem>.<ext>` transcripts are written into
        output_dir: String,

        #[command(flatten)]
        engine: OfflineEngineArgs,

        #[command(flatten)]
        output: BatchOutputArgs,

        /// Poll interval in milliseconds [default: 1000]. Polling with a
        /// stability check keeps the watcher dependency-free and handles
        /// files still being copied into the directory.
        /// Env: GIGASTT_WATCH_POLL_INTERVAL_MS.
        #[arg(long, env = "GIGASTT_WATCH_POLL_INTERVAL_MS", default_value_t = 1000)]
        poll_interval_ms: u64,

        /// Consecutive polls with an identical size+mtime required before a
        /// file is scheduled [default: 2]. Env: GIGASTT_WATCH_SETTLE_POLLS.
        #[arg(long, env = "GIGASTT_WATCH_SETTLE_POLLS", default_value_t = 2)]
        settle_polls: u32,
    },
}

/// clap value parser for `--model-variant`. Accepts `rnnt` / `e2e_rnnt` /
/// `ml_ctc` / `ml_ctc_large` (case-insensitive); see [`ModelVariant::from_str`].
fn parse_model_variant(s: &str) -> Result<ModelVariant, String> {
    s.parse()
}

/// Parse the `download --progress` value (`human` | `json`).
fn parse_progress_mode(s: &str) -> Result<ProgressMode, String> {
    s.parse()
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let matches = Cli::command().get_matches();
    let cli = Cli::from_arg_matches(&matches).unwrap_or_else(|e| e.exit());

    if cli.offline {
        // Translate the flag into the env var the download guard in
        // gigastt-core reads, so both spellings behave identically.
        // Safety: this is the first statement after argument parsing — nothing
        // has read or written the process environment concurrently yet (the
        // only env readers live further down this same call path).
        unsafe { std::env::set_var("GIGASTT_OFFLINE", "1") };
    }
    // NDJSON download progress owns stdout: in `--progress=json` mode the
    // tracing writer moves to stderr so stdout carries nothing but event
    // lines (the default writer is stdout).
    let json_progress = matches!(
        &cli.command,
        Commands::Download { progress, .. } if *progress == ProgressMode::Json
    );

    let directive = format!("gigastt={}", cli.log_level);
    let filter = EnvFilter::from_default_env().add_directive(directive.parse()?);
    if json_progress {
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(std::io::stderr)
            .init();
    } else {
        tracing_subscriber::fmt().with_env_filter(filter).init();
    }

    match cli.command {
        Commands::Serve(args) => {
            run_serve(&matches, args).await?;
        }
        Commands::Download {
            model_dir,
            model_variant,
            #[cfg(feature = "diarization")]
            skip_diarization,
            progress,
            #[cfg(feature = "ane")]
            ane,
        } => {
            run_download(
                model_dir,
                model_variant,
                #[cfg(feature = "diarization")]
                skip_diarization,
                progress,
                #[cfg(feature = "ane")]
                ane,
            )
            .await?;
        }
        Commands::Quantize { model_dir, force } => {
            run_quantize(model_dir, force)?;
        }
        Commands::CacheGc {
            model_dir,
            dry_run,
            dedupe,
        } => {
            run_cache_gc(model_dir, dry_run, dedupe)?;
        }
        Commands::Transcribe {
            file,
            model_dir,
            model_variant,
            punctuation,
            punct_model_dir,
            itn,
            hotwords_file,
            hotwords_default,
            hotwords_boost,
            vad,
            vad_threshold,
            vad_min_silence_ms,
            vad_model_dir,
            encoder_intra_threads,
            format,
            output,
            max_chars_per_line,
            max_words_per_line,
            word_timestamps,
            stereo_speakers,
            codec,
            sample_rate,
        } => {
            run_transcribe(
                file,
                model_dir,
                model_variant,
                punctuation,
                punct_model_dir,
                itn,
                hotwords_file,
                hotwords_default,
                hotwords_boost,
                vad,
                vad_threshold,
                vad_min_silence_ms,
                vad_model_dir,
                encoder_intra_threads,
                format,
                output,
                max_chars_per_line,
                max_words_per_line,
                word_timestamps,
                stereo_speakers,
                codec,
                sample_rate,
            )
            .await?;
        }
        Commands::TranscribeBatch {
            input_dir,
            output_dir,
            engine: eng,
            output: out,
        } => {
            run_transcribe_batch(input_dir, output_dir, eng, out).await?;
        }
        Commands::Watch {
            input_dir,
            output_dir,
            engine: eng,
            output: out,
            poll_interval_ms,
            settle_polls,
        } => {
            run_watch(
                input_dir,
                output_dir,
                eng,
                out,
                poll_interval_ms,
                settle_polls,
            )
            .await?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::serve::{
        ServeProfile, build_limits, build_server_config, ensure_bind_allowed,
        ensure_metrics_bind_allowed, is_loopback_host,
    };
    use super::*;
    use gigastt::server;
    use gigastt::server::RuntimeLimits;

    // Serialize tests that mutate process env vars to avoid races under
    // cargo test's default multi-threaded harness (used by tarpaulin).
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn test_is_loopback_host_recognises_common_forms() {
        assert!(is_loopback_host("127.0.0.1"));
        assert!(is_loopback_host("localhost"));
        assert!(is_loopback_host("::1"));
        assert!(is_loopback_host("[::1]"));
        assert!(is_loopback_host("127.0.0.2")); // loopback /8
        assert!(!is_loopback_host("0.0.0.0"));
        assert!(!is_loopback_host("192.168.1.10"));
        assert!(!is_loopback_host("example.com"));
    }

    #[test]
    fn test_ensure_bind_allowed_loopback_ok() {
        ensure_bind_allowed("127.0.0.1", false).expect("loopback must be allowed");
        ensure_bind_allowed("localhost", false).expect("localhost must be allowed");
    }

    #[test]
    fn test_ensure_bind_allowed_non_loopback_requires_flag() {
        let _guard = ENV_LOCK.lock().unwrap();
        let previous = std::env::var("GIGASTT_ALLOW_BIND_ANY").ok();
        unsafe {
            std::env::remove_var("GIGASTT_ALLOW_BIND_ANY");
        }
        let result = ensure_bind_allowed("0.0.0.0", false);
        if let Some(v) = previous {
            unsafe {
                std::env::set_var("GIGASTT_ALLOW_BIND_ANY", v);
            }
        }
        assert!(
            result.is_err(),
            "0.0.0.0 without --bind-all must be rejected"
        );
    }

    #[test]
    fn test_ensure_bind_allowed_explicit_flag_ok() {
        ensure_bind_allowed("0.0.0.0", true).expect("explicit --bind-all must pass");
    }

    #[test]
    fn test_ensure_metrics_bind_allowed_disabled_skips_gate() {
        // Metrics off: no listener is bound, so even a wildcard address needs
        // no consent.
        let addr = "0.0.0.0:9090".parse().unwrap();
        ensure_metrics_bind_allowed(false, &addr, false)
            .expect("disabled metrics listener must skip the gate");
    }

    #[test]
    fn test_ensure_metrics_bind_allowed_loopback_ok() {
        let _guard = ENV_LOCK.lock().unwrap();
        let addr = "127.0.0.1:9090".parse().unwrap();
        ensure_metrics_bind_allowed(true, &addr, false)
            .expect("loopback metrics bind must be allowed");
    }

    #[test]
    fn test_ensure_metrics_bind_allowed_non_loopback_requires_flag() {
        let _guard = ENV_LOCK.lock().unwrap();
        let previous = std::env::var("GIGASTT_ALLOW_BIND_ANY").ok();
        unsafe {
            std::env::remove_var("GIGASTT_ALLOW_BIND_ANY");
        }
        let addr = "0.0.0.0:9090".parse().unwrap();
        let result = ensure_metrics_bind_allowed(true, &addr, false);
        if let Some(v) = previous {
            unsafe {
                std::env::set_var("GIGASTT_ALLOW_BIND_ANY", v);
            }
        }
        assert!(
            result.is_err(),
            "0.0.0.0 metrics bind without --bind-all must be rejected"
        );
    }

    #[test]
    fn test_ensure_metrics_bind_allowed_explicit_flag_ok() {
        let _guard = ENV_LOCK.lock().unwrap();
        let addr = "0.0.0.0:9090".parse().unwrap();
        ensure_metrics_bind_allowed(true, &addr, true)
            .expect("explicit --bind-all must allow the metrics bind");
    }

    #[test]
    fn test_cli_serve_parsing() {
        let cli = Cli::parse_from(["gigastt", "serve", "--port", "1234", "--bind-all"]);
        match cli.command {
            Commands::Serve(ServeArgs {
                port,
                bind_all,
                metrics,
                model_variant,
                ..
            }) => {
                assert_eq!(port, 1234);
                assert!(bind_all);
                assert!(!metrics);
                // No --model-variant → None (auto-detect from disk).
                assert_eq!(model_variant, None);
            }
            _ => panic!("expected Serve"),
        }
    }

    // Restore a captured env value when dropped, so an env-mutating test never
    // leaks `GIGASTT_ENCODER_INTRA_THREADS` to a sibling test (clap reads the
    // process environment). Paired with `ENV_LOCK` to serialize these tests.
    struct EnvRestore(&'static str, Option<String>);
    impl Drop for EnvRestore {
        fn drop(&mut self) {
            match &self.1 {
                Some(v) => unsafe { std::env::set_var(self.0, v) },
                None => unsafe { std::env::remove_var(self.0) },
            }
        }
    }

    #[test]
    fn test_cli_serve_profile_edge_defaults_pool_and_vad_flags() {
        // Parsing only: profile field is Edge; runtime applies pool/vad in main.
        let cli = Cli::try_parse_from(["gigastt", "serve", "--profile", "edge"]).expect("parse");
        match cli.command {
            Commands::Serve(ServeArgs {
                profile,
                pool_size,
                vad,
                ..
            }) => {
                assert_eq!(profile, ServeProfile::Edge);
                // clap defaults before profile application:
                assert_eq!(pool_size, 2);
                assert!(!vad);
            }
            _ => panic!("expected Serve"),
        }
    }

    #[test]
    fn test_cli_serve_encoder_intra_threads_default() {
        // Unset → None, so the default resolves from the pool size at load time.
        let _guard = ENV_LOCK.lock().unwrap();
        let _restore = EnvRestore(
            "GIGASTT_ENCODER_INTRA_THREADS",
            std::env::var("GIGASTT_ENCODER_INTRA_THREADS").ok(),
        );
        unsafe {
            std::env::remove_var("GIGASTT_ENCODER_INTRA_THREADS");
        }
        let cli = Cli::parse_from(["gigastt", "serve"]);
        match cli.command {
            Commands::Serve(ServeArgs {
                encoder_intra_threads,
                ..
            }) => assert_eq!(encoder_intra_threads, None),
            _ => panic!("expected Serve"),
        }
    }

    #[test]
    fn test_cli_serve_encoder_intra_threads_flag() {
        // The explicit flag wins over any inherited env value.
        let _guard = ENV_LOCK.lock().unwrap();
        let _restore = EnvRestore(
            "GIGASTT_ENCODER_INTRA_THREADS",
            std::env::var("GIGASTT_ENCODER_INTRA_THREADS").ok(),
        );
        unsafe {
            std::env::remove_var("GIGASTT_ENCODER_INTRA_THREADS");
        }
        let cli = Cli::parse_from(["gigastt", "serve", "--encoder-intra-threads", "4"]);
        match cli.command {
            Commands::Serve(ServeArgs {
                encoder_intra_threads,
                ..
            }) => assert_eq!(encoder_intra_threads, Some(4)),
            _ => panic!("expected Serve"),
        }
    }

    #[test]
    fn test_cli_serve_encoder_intra_threads_env() {
        // The flag is wired to GIGASTT_ENCODER_INTRA_THREADS; clap reads the
        // process environment, so serialize against other env-mutating tests.
        let _guard = ENV_LOCK.lock().unwrap();
        let _restore = EnvRestore(
            "GIGASTT_ENCODER_INTRA_THREADS",
            std::env::var("GIGASTT_ENCODER_INTRA_THREADS").ok(),
        );
        unsafe {
            std::env::set_var("GIGASTT_ENCODER_INTRA_THREADS", "6");
        }
        let cli = Cli::parse_from(["gigastt", "serve"]);
        match cli.command {
            Commands::Serve(ServeArgs {
                encoder_intra_threads,
                ..
            }) => assert_eq!(encoder_intra_threads, Some(6)),
            _ => panic!("expected Serve"),
        }
    }

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
    fn test_cli_serve_model_variant_override() {
        let cli = Cli::parse_from(["gigastt", "serve", "--model-variant", "e2e_rnnt"]);
        match cli.command {
            Commands::Serve(ServeArgs { model_variant, .. }) => {
                assert_eq!(model_variant, Some(ModelVariant::E2eRnnt));
            }
            _ => panic!("expected Serve"),
        }
    }

    #[test]
    fn test_cli_serve_model_variant_explicit_rnnt() {
        let cli = Cli::parse_from(["gigastt", "serve", "--model-variant", "rnnt"]);
        match cli.command {
            Commands::Serve(ServeArgs { model_variant, .. }) => {
                assert_eq!(model_variant, Some(ModelVariant::Rnnt));
            }
            _ => panic!("expected Serve"),
        }
    }

    #[test]
    fn test_cli_download_parsing() {
        let cli = Cli::parse_from(["gigastt", "download", "--model-dir", "/tmp/models"]);
        match cli.command {
            Commands::Download {
                model_dir,
                model_variant,
                ..
            } => {
                assert_eq!(model_dir, "/tmp/models");
                assert_eq!(model_variant, ModelVariant::Rnnt);
            }
            _ => panic!("expected Download"),
        }
    }

    #[test]
    fn test_cli_download_model_variant_override() {
        let cli = Cli::parse_from(["gigastt", "download", "--model-variant", "e2e_rnnt"]);
        match cli.command {
            Commands::Download { model_variant, .. } => {
                assert_eq!(model_variant, ModelVariant::E2eRnnt);
            }
            _ => panic!("expected Download"),
        }
    }

    #[test]
    fn test_cli_cache_gc_parsing() {
        let cli = Cli::try_parse_from([
            "gigastt",
            "cache-gc",
            "--model-dir",
            "/tmp/models",
            "--dry-run",
            "--dedupe",
        ])
        .expect("parse cache-gc");
        match cli.command {
            Commands::CacheGc {
                model_dir,
                dry_run,
                dedupe,
            } => {
                assert_eq!(model_dir, "/tmp/models");
                assert!(dry_run);
                assert!(dedupe);
            }
            _ => panic!("expected CacheGc"),
        }
    }

    #[test]
    fn test_cli_quantize_parsing() {
        let cli = Cli::parse_from(["gigastt", "quantize", "--force"]);
        match cli.command {
            Commands::Quantize { force, .. } => {
                assert!(force);
            }
            _ => panic!("expected Quantize"),
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
    fn test_cli_serve_rejects_unknown_model_variant() {
        let res = Cli::try_parse_from(["gigastt", "serve", "--model-variant", "whisper"]);
        assert!(res.is_err(), "unknown variant must be rejected by clap");
    }

    #[test]
    fn test_cli_serve_punctuation_defaults_auto() {
        let cli = Cli::parse_from(["gigastt", "serve"]);
        match cli.command {
            Commands::Serve(ServeArgs {
                punctuation,
                punct_model_dir,
                ..
            }) => {
                assert_eq!(punctuation, PunctuationMode::Auto);
                assert!(punct_model_dir.contains("punct"));
            }
            _ => panic!("expected Serve"),
        }
    }

    #[test]
    fn test_cli_serve_punctuation_override() {
        let cli = Cli::parse_from([
            "gigastt",
            "serve",
            "--punctuation",
            "on",
            "--punct-model-dir",
            "/tmp/punct",
        ]);
        match cli.command {
            Commands::Serve(ServeArgs {
                punctuation,
                punct_model_dir,
                ..
            }) => {
                assert_eq!(punctuation, PunctuationMode::On);
                assert_eq!(punct_model_dir, "/tmp/punct");
            }
            _ => panic!("expected Serve"),
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
    fn test_cli_serve_itn_defaults_auto() {
        let cli = Cli::parse_from(["gigastt", "serve"]);
        match cli.command {
            Commands::Serve(ServeArgs { itn, .. }) => assert_eq!(itn, ItnMode::Auto),
            _ => panic!("expected Serve"),
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
    fn test_cli_serve_hotwords_flags() {
        let cli = Cli::parse_from([
            "gigastt",
            "serve",
            "--hotwords-file",
            "/tmp/hw.txt",
            "--hotwords-default",
            "--hotwords-boost",
            "8.5",
        ]);
        match cli.command {
            Commands::Serve(ServeArgs {
                hotwords_file,
                hotwords_default,
                hotwords_boost,
                ..
            }) => {
                assert_eq!(hotwords_file, Some("/tmp/hw.txt".to_string()));
                assert!(hotwords_default);
                assert_eq!(hotwords_boost, Some(8.5));
            }
            _ => panic!("expected Serve"),
        }
    }

    #[test]
    fn test_cli_serve_hotwords_default_off() {
        let cli = Cli::parse_from(["gigastt", "serve"]);
        match cli.command {
            Commands::Serve(ServeArgs {
                hotwords_file,
                hotwords_default,
                hotwords_boost,
                ..
            }) => {
                assert_eq!(hotwords_file, None);
                assert!(!hotwords_default);
                assert_eq!(hotwords_boost, None);
            }
            _ => panic!("expected Serve"),
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
    fn test_cli_serve_with_metrics() {
        let cli = Cli::parse_from(["gigastt", "serve", "--metrics"]);
        match cli.command {
            Commands::Serve(ServeArgs {
                metrics,
                metrics_listen,
                ..
            }) => {
                assert!(metrics);
                // Unset → resolved to the loopback default downstream.
                assert!(metrics_listen.is_none());
            }
            _ => panic!("expected Serve"),
        }
    }

    #[test]
    fn test_cli_serve_metrics_listen_override() {
        let cli = Cli::parse_from([
            "gigastt",
            "serve",
            "--metrics",
            "--metrics-listen",
            "127.0.0.1:9123",
        ]);
        match cli.command {
            Commands::Serve(ServeArgs { metrics_listen, .. }) => {
                let addr = metrics_listen.expect("--metrics-listen must parse");
                assert_eq!(addr.port(), 9123);
                assert!(addr.ip().is_loopback());
            }
            _ => panic!("expected Serve"),
        }
        // Default when omitted resolves to 127.0.0.1:9090.
        assert_eq!(server::config::default_metrics_listen().port(), 9090);
    }

    #[test]
    fn test_is_loopback_host_ipv6_bracketed() {
        assert!(is_loopback_host("[::1]"));
        assert!(!is_loopback_host("[2001:db8::1]"));
    }

    #[test]
    fn test_ensure_bind_allowed_env_opt_in() {
        let _guard = ENV_LOCK.lock().unwrap();
        let previous = std::env::var("GIGASTT_ALLOW_BIND_ANY").ok();
        unsafe {
            std::env::set_var("GIGASTT_ALLOW_BIND_ANY", "1");
        }
        let result = ensure_bind_allowed("0.0.0.0", false);
        if let Some(v) = previous {
            unsafe {
                std::env::set_var("GIGASTT_ALLOW_BIND_ANY", v);
            }
        } else {
            unsafe {
                std::env::remove_var("GIGASTT_ALLOW_BIND_ANY");
            }
        }
        assert!(result.is_ok(), "env opt-in must allow non-loopback bind");
    }

    #[test]
    fn test_build_limits_defaults_when_no_config() {
        let limits = build_limits(
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            None,
        )
        .unwrap();
        assert_eq!(limits.idle_timeout_secs, 300);
        assert_eq!(limits.ws_frame_max_bytes, 512 * 1024);
    }

    #[test]
    fn test_build_limits_job_overrides() {
        let limits = build_limits(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(true),
            Some(7200),
            Some(50),
            Some(2 * 1024 * 1024),
            Some(5),
        )
        .unwrap();
        assert!(limits.jobs_enabled);
        assert_eq!(limits.jobs_ttl_secs, 7200);
        assert_eq!(limits.jobs_max, 50);
        assert_eq!(limits.jobs_max_bytes, 2 * 1024 * 1024);
        assert_eq!(limits.jobs_retry, 5);
    }

    #[test]
    fn test_cli_serve_jobs_flags() {
        let cli = Cli::parse_from([
            "gigastt",
            "serve",
            "--enable-jobs",
            "--jobs-ttl-secs",
            "7200",
            "--jobs-max",
            "50",
            "--jobs-max-bytes",
            "1048576",
            "--jobs-retry",
            "5",
        ]);
        match cli.command {
            Commands::Serve(ServeArgs {
                enable_jobs,
                jobs_ttl_secs,
                jobs_max,
                jobs_max_bytes,
                jobs_retry,
                ..
            }) => {
                assert!(enable_jobs);
                assert_eq!(jobs_ttl_secs, Some(7200));
                assert_eq!(jobs_max, Some(50));
                assert_eq!(jobs_max_bytes, Some(1048576));
                assert_eq!(jobs_retry, Some(5));
            }
            _ => panic!("expected Serve"),
        }
    }

    #[test]
    fn test_cli_serve_jobs_defaults_off() {
        let cli = Cli::parse_from(["gigastt", "serve"]);
        match cli.command {
            Commands::Serve(ServeArgs {
                enable_jobs,
                jobs_ttl_secs,
                jobs_max,
                jobs_retry,
                ..
            }) => {
                assert!(!enable_jobs);
                assert_eq!(jobs_ttl_secs, None);
                assert_eq!(jobs_max, None);
                assert_eq!(jobs_retry, None);
            }
            _ => panic!("expected Serve"),
        }
    }

    #[test]
    fn test_build_limits_applies_overrides() {
        let limits = build_limits(
            None,
            Some(600),
            Some(1024),
            Some(10 * 1024 * 1024),
            Some(60),
            Some(20),
            Some(1800),
            Some(5),
            Some(15),
            Some(45),
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        assert_eq!(limits.idle_timeout_secs, 600);
        assert_eq!(limits.ws_frame_max_bytes, 1024);
        assert_eq!(limits.body_limit_bytes, 10 * 1024 * 1024);
        assert_eq!(limits.rate_limit_per_minute, 60);
        assert_eq!(limits.rate_limit_burst, 20);
        assert_eq!(limits.max_session_secs, 1800);
        assert_eq!(limits.shutdown_drain_secs, 5);
        assert_eq!(limits.pool_checkout_timeout_secs, 15);
        assert_eq!(limits.inference_timeout_secs, 45);
    }

    #[test]
    fn test_build_limits_with_valid_config_file() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"idle_timeout_secs = 123\n").unwrap();
        let limits = build_limits(
            Some(tmp.path().to_str().unwrap()),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        assert_eq!(limits.idle_timeout_secs, 123);
    }

    #[test]
    fn test_build_limits_with_invalid_config_file() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"not valid toml {{{").unwrap();
        let result = build_limits(
            Some(tmp.path().to_str().unwrap()),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_build_limits_rejects_zero_burst_with_nonzero_rpm() {
        let result = build_limits(
            None,
            None,
            None,
            None,
            Some(30),
            Some(0),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("rate-limit-burst"));
    }

    #[test]
    fn test_build_limits_allows_zero_rpm() {
        let limits = build_limits(
            None,
            None,
            None,
            None,
            Some(0),
            Some(0),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        assert_eq!(limits.rate_limit_per_minute, 0);
        assert_eq!(limits.rate_limit_burst, 0);
    }

    #[test]
    fn test_build_server_config() {
        let limits = RuntimeLimits::default();
        let cfg = build_server_config(
            1234,
            "127.0.0.1".into(),
            vec!["https://app.example.com".into()],
            false,
            limits.clone(),
            true,
            "127.0.0.1:9099".parse().unwrap(),
            true,
            Some("/tmp/config.toml".into()),
            2,
        );
        assert_eq!(cfg.port, 1234);
        assert_eq!(cfg.metrics_listen.port(), 9099);
        assert_eq!(cfg.host, "127.0.0.1");
        assert_eq!(cfg.batch_pool_size, 2);
        assert_eq!(cfg.origin_policy.allowed_origins.len(), 1);
        assert!(!cfg.origin_policy.allow_any);
        assert!(cfg.metrics_enabled);
        assert!(cfg.trust_proxy);
        assert_eq!(
            cfg.config_path,
            Some(std::path::PathBuf::from("/tmp/config.toml"))
        );
        assert_eq!(cfg.limits.idle_timeout_secs, limits.idle_timeout_secs);
    }

    #[test]
    fn test_parse_model_variant_valid_and_invalid() {
        assert_eq!(parse_model_variant("rnnt").unwrap(), ModelVariant::Rnnt);
        assert_eq!(
            parse_model_variant("e2e_rnnt").unwrap(),
            ModelVariant::E2eRnnt
        );
        assert!(parse_model_variant("whisper").is_err());
    }

    #[test]
    fn test_cli_serve_vad_flags() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _restore_vad = EnvRestore("GIGASTT_VAD", std::env::var("GIGASTT_VAD").ok());
        let _restore_threshold = EnvRestore(
            "GIGASTT_VAD_THRESHOLD",
            std::env::var("GIGASTT_VAD_THRESHOLD").ok(),
        );
        let _restore_sil = EnvRestore(
            "GIGASTT_VAD_MIN_SILENCE_MS",
            std::env::var("GIGASTT_VAD_MIN_SILENCE_MS").ok(),
        );
        let _restore_dir = EnvRestore(
            "GIGASTT_VAD_MODEL_DIR",
            std::env::var("GIGASTT_VAD_MODEL_DIR").ok(),
        );
        unsafe {
            std::env::remove_var("GIGASTT_VAD");
            std::env::remove_var("GIGASTT_VAD_THRESHOLD");
            std::env::remove_var("GIGASTT_VAD_MIN_SILENCE_MS");
            std::env::remove_var("GIGASTT_VAD_MODEL_DIR");
        }
        let cli = Cli::parse_from([
            "gigastt",
            "serve",
            "--vad",
            "--vad-threshold",
            "0.8",
            "--vad-min-silence-ms",
            "700",
            "--vad-model-dir",
            "/tmp/vad",
        ]);
        match cli.command {
            Commands::Serve(ServeArgs {
                vad,
                vad_threshold,
                vad_min_silence_ms,
                vad_model_dir,
                ..
            }) => {
                assert!(vad);
                assert_eq!(vad_threshold, Some(0.8));
                assert_eq!(vad_min_silence_ms, Some(700));
                assert_eq!(vad_model_dir, "/tmp/vad");
            }
            _ => panic!("expected Serve"),
        }
    }

    #[test]
    fn test_cli_serve_vad_defaults_off() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _restore_vad = EnvRestore("GIGASTT_VAD", std::env::var("GIGASTT_VAD").ok());
        let _restore_threshold = EnvRestore(
            "GIGASTT_VAD_THRESHOLD",
            std::env::var("GIGASTT_VAD_THRESHOLD").ok(),
        );
        let _restore_sil = EnvRestore(
            "GIGASTT_VAD_MIN_SILENCE_MS",
            std::env::var("GIGASTT_VAD_MIN_SILENCE_MS").ok(),
        );
        unsafe {
            std::env::remove_var("GIGASTT_VAD");
            std::env::remove_var("GIGASTT_VAD_THRESHOLD");
            std::env::remove_var("GIGASTT_VAD_MIN_SILENCE_MS");
        }
        let cli = Cli::parse_from(["gigastt", "serve"]);
        match cli.command {
            Commands::Serve(ServeArgs {
                vad,
                vad_threshold,
                vad_min_silence_ms,
                endpoint_mode,
                ..
            }) => {
                assert!(!vad);
                assert_eq!(vad_threshold, None);
                assert_eq!(vad_min_silence_ms, None);
                assert_eq!(endpoint_mode, "auto");
            }
            _ => panic!("expected Serve"),
        }
    }

    #[test]
    fn test_cli_serve_endpoint_mode_assistant() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _restore = EnvRestore(
            "GIGASTT_ENDPOINT_MODE",
            std::env::var("GIGASTT_ENDPOINT_MODE").ok(),
        );
        unsafe {
            std::env::remove_var("GIGASTT_ENDPOINT_MODE");
        }
        let cli = Cli::parse_from(["gigastt", "serve", "--endpoint-mode", "assistant"]);
        match cli.command {
            Commands::Serve(ServeArgs { endpoint_mode, .. }) => {
                assert_eq!(endpoint_mode, "assistant");
            }
            _ => panic!("expected Serve"),
        }
    }

    #[test]
    fn test_cli_serve_pool_and_thread_flags() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _restore_min = EnvRestore(
            "GIGASTT_POOL_MIN_SIZE",
            std::env::var("GIGASTT_POOL_MIN_SIZE").ok(),
        );
        let _restore_batch = EnvRestore(
            "GIGASTT_BATCH_POOL_SIZE",
            std::env::var("GIGASTT_BATCH_POOL_SIZE").ok(),
        );
        let _restore_threads = EnvRestore(
            "GIGASTT_ENCODER_INTRA_THREADS",
            std::env::var("GIGASTT_ENCODER_INTRA_THREADS").ok(),
        );
        unsafe {
            std::env::remove_var("GIGASTT_POOL_MIN_SIZE");
            std::env::remove_var("GIGASTT_BATCH_POOL_SIZE");
            std::env::remove_var("GIGASTT_ENCODER_INTRA_THREADS");
        }
        let cli = Cli::parse_from([
            "gigastt",
            "serve",
            "--pool-size",
            "8",
            "--pool-min-size",
            "3",
            "--batch-pool-size",
            "2",
        ]);
        match cli.command {
            Commands::Serve(ServeArgs {
                pool_size,
                pool_min_size,
                batch_pool_size,
                ..
            }) => {
                assert_eq!(pool_size, 8);
                assert_eq!(pool_min_size, 3);
                assert_eq!(batch_pool_size, 2);
            }
            _ => panic!("expected Serve"),
        }
    }

    #[test]
    fn test_cli_serve_runtime_limit_flags() {
        let _guard = ENV_LOCK.lock().unwrap();
        // These flags read env vars; clear them so the explicit args win.
        let restores: Vec<EnvRestore> = [
            "GIGASTT_IDLE_TIMEOUT_SECS",
            "GIGASTT_WS_FRAME_MAX_BYTES",
            "GIGASTT_BODY_LIMIT_BYTES",
            "GIGASTT_RATE_LIMIT_PER_MINUTE",
            "GIGASTT_RATE_LIMIT_BURST",
            "GIGASTT_MAX_SESSION_SECS",
            "GIGASTT_SHUTDOWN_DRAIN_SECS",
            "GIGASTT_POOL_CHECKOUT_TIMEOUT_SECS",
            "GIGASTT_INFERENCE_TIMEOUT_SECS",
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
        let cli = Cli::parse_from([
            "gigastt",
            "serve",
            "--idle-timeout-secs",
            "120",
            "--ws-frame-max-bytes",
            "4096",
            "--body-limit-bytes",
            "8192",
            "--rate-limit-per-minute",
            "90",
            "--rate-limit-burst",
            "15",
            "--max-session-secs",
            "777",
            "--shutdown-drain-secs",
            "7",
            "--pool-checkout-timeout-secs",
            "11",
            "--inference-timeout-secs",
            "300",
            "--trust-proxy",
        ]);
        match cli.command {
            Commands::Serve(ServeArgs {
                idle_timeout_secs,
                ws_frame_max_bytes,
                body_limit_bytes,
                rate_limit_per_minute,
                rate_limit_burst,
                max_session_secs,
                shutdown_drain_secs,
                pool_checkout_timeout_secs,
                inference_timeout_secs,
                trust_proxy,
                ..
            }) => {
                assert_eq!(idle_timeout_secs, Some(120));
                assert_eq!(ws_frame_max_bytes, Some(4096));
                assert_eq!(body_limit_bytes, Some(8192));
                assert_eq!(rate_limit_per_minute, Some(90));
                assert_eq!(rate_limit_burst, Some(15));
                assert_eq!(max_session_secs, Some(777));
                assert_eq!(shutdown_drain_secs, Some(7));
                assert_eq!(pool_checkout_timeout_secs, Some(11));
                assert_eq!(inference_timeout_secs, Some(300));
                assert!(trust_proxy);
            }
            _ => panic!("expected Serve"),
        }
        drop(restores);
    }

    #[test]
    fn test_cli_serve_config_flag() {
        let cli = Cli::parse_from(["gigastt", "serve", "--config", "/tmp/limits.toml"]);
        match cli.command {
            Commands::Serve(ServeArgs { config, .. }) => {
                assert_eq!(config, Some("/tmp/limits.toml".to_string()));
            }
            _ => panic!("expected Serve"),
        }
    }

    #[test]
    fn test_cli_serve_cors_and_origin_flags() {
        let cli = Cli::parse_from([
            "gigastt",
            "serve",
            "--allow-origin",
            "https://a.example.com",
            "--allow-origin",
            "https://b.example.com",
            "--cors-allow-any",
        ]);
        match cli.command {
            Commands::Serve(ServeArgs {
                allow_origin,
                cors_allow_any,
                ..
            }) => {
                assert_eq!(allow_origin.len(), 2);
                assert_eq!(allow_origin[0], "https://a.example.com");
                assert!(cors_allow_any);
            }
            _ => panic!("expected Serve"),
        }
    }

    #[test]
    fn test_cli_download_parses() {
        let cli = Cli::try_parse_from(["gigastt", "download"]).expect("parse");
        match cli.command {
            Commands::Download { model_variant, .. } => {
                assert_eq!(model_variant, ModelVariant::Rnnt);
            }
            _ => panic!("expected Download"),
        }
    }

    #[test]
    fn test_cli_download_rejects_fp32_flag() {
        let err = match Cli::try_parse_from(["gigastt", "download", "--fp32"]) {
            Ok(_) => panic!("fp32 flag must be removed from download"),
            Err(e) => e,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("unexpected") || msg.contains("fp32") || msg.contains("unknown"),
            "expected clap to reject --fp32, got: {msg}"
        );
    }

    #[test]
    fn test_cli_download_progress_defaults_to_human() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _restore = EnvRestore(
            "GIGASTT_DOWNLOAD_PROGRESS",
            std::env::var("GIGASTT_DOWNLOAD_PROGRESS").ok(),
        );
        unsafe {
            std::env::remove_var("GIGASTT_DOWNLOAD_PROGRESS");
        }
        let cli = Cli::parse_from(["gigastt", "download"]);
        match cli.command {
            Commands::Download { progress, .. } => assert_eq!(progress, ProgressMode::Human),
            _ => panic!("expected Download"),
        }
    }

    #[test]
    fn test_cli_download_progress_json_flag() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _restore = EnvRestore(
            "GIGASTT_DOWNLOAD_PROGRESS",
            std::env::var("GIGASTT_DOWNLOAD_PROGRESS").ok(),
        );
        unsafe {
            std::env::remove_var("GIGASTT_DOWNLOAD_PROGRESS");
        }
        let cli = Cli::parse_from(["gigastt", "download", "--progress", "json"]);
        match cli.command {
            Commands::Download { progress, .. } => assert_eq!(progress, ProgressMode::Json),
            _ => panic!("expected Download"),
        }
    }

    #[test]
    fn test_cli_download_progress_env_var() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _restore = EnvRestore(
            "GIGASTT_DOWNLOAD_PROGRESS",
            std::env::var("GIGASTT_DOWNLOAD_PROGRESS").ok(),
        );
        unsafe {
            std::env::set_var("GIGASTT_DOWNLOAD_PROGRESS", "json");
        }
        let cli = Cli::parse_from(["gigastt", "download"]);
        match cli.command {
            Commands::Download { progress, .. } => assert_eq!(progress, ProgressMode::Json),
            _ => panic!("expected Download"),
        }
    }

    #[test]
    fn test_parse_progress_mode_value_parser() {
        assert_eq!(parse_progress_mode("human").unwrap(), ProgressMode::Human);
        assert_eq!(parse_progress_mode("json").unwrap(), ProgressMode::Json);
        assert!(parse_progress_mode("xml").is_err());
    }

    #[cfg(feature = "ane")]
    #[test]
    fn test_cli_download_ane_flag() {
        let cli = Cli::parse_from(["gigastt", "download", "--ane"]);
        match cli.command {
            Commands::Download { ane, .. } => assert!(ane),
            _ => panic!("expected Download"),
        }
        // Absent by default.
        let cli = Cli::parse_from(["gigastt", "download"]);
        match cli.command {
            Commands::Download { ane, .. } => assert!(!ane),
            _ => panic!("expected Download"),
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

    #[test]
    fn test_cli_rejects_unknown_subcommand() {
        let res = Cli::try_parse_from(["gigastt", "bogus"]);
        assert!(res.is_err(), "unknown subcommand must be rejected");
    }

    #[test]
    fn test_cli_top_level_long_help_points_to_subcommand_engine_flags() {
        use clap::CommandFactory;
        let help = Cli::command().render_long_help().to_string();
        for needle in [
            "serve --help",
            "--punctuation",
            "--itn",
            "--vad",
            "--model-variant",
        ] {
            assert!(
                help.contains(needle),
                "top-level long help must mention `{needle}`:\n{help}"
            );
        }
    }

    #[test]
    fn test_cli_serve_rejects_bad_punctuation_value() {
        let res = Cli::try_parse_from(["gigastt", "serve", "--punctuation", "sometimes"]);
        assert!(res.is_err(), "invalid punctuation mode must be rejected");
    }

    #[test]
    fn test_cli_serve_rejects_bad_itn_value() {
        let res = Cli::try_parse_from(["gigastt", "serve", "--itn", "sometimes"]);
        assert!(res.is_err(), "invalid itn mode must be rejected");
    }

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
}
