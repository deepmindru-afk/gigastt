//! Offline `transcribe` / `transcribe-batch` / `watch` command bodies.

use anyhow::Context;
use clap::Args;
use gigastt::batch;
use gigastt::boot::{
    EngineRecipe, ItnMode, PunctuationMode, parse_itn_mode, parse_punctuation_mode,
};
use gigastt_core::export::{ExportFormat, RenderOpts};
use gigastt_core::inference;
use gigastt_core::model;
use gigastt_core::model::ModelVariant;
use std::str::FromStr;

/// Engine / post-processing flags shared by the offline directory commands
/// (`transcribe-batch`, `watch`). Mirrors the corresponding `transcribe` flags.
#[derive(Args)]
pub(crate) struct OfflineEngineArgs {
    /// Model directory
    #[arg(long, default_value_t = model::default_model_dir())]
    pub(crate) model_dir: String,

    /// Recognition head to use. Omit to auto-detect from the model directory.
    /// Env: GIGASTT_MODEL_VARIANT.
    #[arg(
        long,
        env = "GIGASTT_MODEL_VARIANT",
        value_parser = crate::parse_model_variant
    )]
    pub(crate) model_variant: Option<ModelVariant>,

    /// Punctuation + capitalization restoration: `on`, `off`, or `auto`.
    /// Env: GIGASTT_PUNCTUATION.
    #[arg(
        long,
        env = "GIGASTT_PUNCTUATION",
        default_value = "auto",
        value_parser = parse_punctuation_mode
    )]
    pub(crate) punctuation: PunctuationMode,

    /// Directory holding the optional punctuation model.
    /// Env: GIGASTT_PUNCT_MODEL_DIR.
    #[arg(
        long,
        env = "GIGASTT_PUNCT_MODEL_DIR",
        default_value_t = model::default_punct_model_dir()
    )]
    pub(crate) punct_model_dir: String,

    /// Inverse text normalization (Russian number-words → digits):
    /// `on`, `off`, or `auto`. Env: GIGASTT_ITN.
    #[arg(
        long,
        env = "GIGASTT_ITN",
        default_value = "auto",
        value_parser = parse_itn_mode
    )]
    pub(crate) itn: ItnMode,

    /// Contextual hotword biasing: path to a file of phrases to boost (one
    /// phrase per line, optional `\t<weight>` suffix). Env: GIGASTT_HOTWORDS_FILE.
    #[arg(long, env = "GIGASTT_HOTWORDS_FILE")]
    pub(crate) hotwords_file: Option<String>,

    /// Also bias the built-in Russian brand/acronym lexicon.
    /// Env: GIGASTT_HOTWORDS_DEFAULT.
    #[arg(long, env = "GIGASTT_HOTWORDS_DEFAULT", default_value_t = false)]
    pub(crate) hotwords_default: bool,

    /// Additive logit boost applied to hotword continuation tokens [default: 5.0].
    /// Env: GIGASTT_HOTWORDS_BOOST.
    #[arg(long, env = "GIGASTT_HOTWORDS_BOOST")]
    pub(crate) hotwords_boost: Option<f32>,

    /// Voice activity detection: skip silence before decoding. Env: GIGASTT_VAD.
    #[arg(long, env = "GIGASTT_VAD", default_value_t = false)]
    pub(crate) vad: bool,

    /// VAD speech-probability threshold in [0,1] [default: 0.5].
    /// Env: GIGASTT_VAD_THRESHOLD.
    #[arg(long, env = "GIGASTT_VAD_THRESHOLD")]
    pub(crate) vad_threshold: Option<f32>,

    /// Minimum trailing silence (ms) to close a speech region [default: 500].
    /// Env: GIGASTT_VAD_MIN_SILENCE_MS.
    #[arg(long, env = "GIGASTT_VAD_MIN_SILENCE_MS")]
    pub(crate) vad_min_silence_ms: Option<u32>,

    /// Directory holding the Silero VAD model (`silero_vad.onnx`).
    /// Env: GIGASTT_VAD_MODEL_DIR.
    #[arg(long, env = "GIGASTT_VAD_MODEL_DIR", default_value_t = model::default_vad_model_dir())]
    pub(crate) vad_model_dir: String,

    /// Intra-op thread count for the encoder session on the CPU build. When
    /// unset, defaults to the logical CPU count divided across the pool.
    /// Do not set `1` on multi-core hosts unless debugging — it is ~3× slower
    /// than auto. Explicit `1` is still honoured for debug passthrough.
    /// Env: GIGASTT_ENCODER_INTRA_THREADS.
    #[arg(long, env = "GIGASTT_ENCODER_INTRA_THREADS")]
    pub(crate) encoder_intra_threads: Option<usize>,

    /// Number of concurrent transcription workers (engine session pool). Each
    /// session loads its own encoder copy (~0.4 GB resident for the INT8
    /// encoder). Default 2 suits multi-file hosts; use `--pool-size 1` on
    /// edge / low-RAM (~400 MB RSS). Pool > 1 costs RAM and can cost ~10–20%
    /// single-job RTF (threads split across slots).
    #[arg(long, default_value_t = 2)]
    pub(crate) pool_size: usize,
}

/// Output / source-policy flags shared by `transcribe-batch` and `watch`.
#[derive(Args)]
pub(crate) struct BatchOutputArgs {
    /// Export formats, comma-separated: txt, json, md, srt, vtt.
    /// One `<stem>.<ext>` file is written per format. Env: GIGASTT_FORMAT.
    #[arg(short, long, env = "GIGASTT_FORMAT", default_value = "txt,json")]
    pub(crate) format: String,

    /// Move each successfully transcribed source file into this directory
    /// (e.g. `--move-to done/`). Mutually exclusive with `--delete-source`.
    /// Env: GIGASTT_BATCH_MOVE_TO.
    #[arg(long, env = "GIGASTT_BATCH_MOVE_TO", conflicts_with = "delete_source")]
    pub(crate) move_to: Option<String>,

    /// Delete each successfully transcribed source file. Failed files are
    /// always left in place. Env: GIGASTT_BATCH_DELETE_SOURCE.
    #[arg(long, env = "GIGASTT_BATCH_DELETE_SOURCE", default_value_t = false)]
    pub(crate) delete_source: bool,

    /// Extra attempts per file after a failure, with a short backoff
    /// [default: 0 for transcribe-batch, 2 for watch]. Env: GIGASTT_BATCH_RETRIES.
    #[arg(long, env = "GIGASTT_BATCH_RETRIES")]
    pub(crate) retries: Option<u32>,

    /// Maximum characters per subtitle/caption line (SRT/VTT) [default: 80]
    #[arg(long, env = "GIGASTT_MAX_CHARS_PER_LINE")]
    pub(crate) max_chars_per_line: Option<usize>,

    /// Maximum words per subtitle/caption line (SRT/VTT) [default: 14]
    #[arg(long, env = "GIGASTT_MAX_WORDS_PER_LINE")]
    pub(crate) max_words_per_line: Option<usize>,

    /// Include per-word timestamps in Markdown output
    #[arg(long, env = "GIGASTT_WORD_TIMESTAMPS", default_value_t = false)]
    pub(crate) word_timestamps: bool,
}

/// Build the synchronous transcribe closure injected into the batch / watch
/// runners: check out a pool triplet (blocking — the runner calls it from a
/// blocking thread) and transcribe one file.
pub(crate) fn make_transcribe_fn(engine: std::sync::Arc<inference::Engine>) -> batch::TranscribeFn {
    std::sync::Arc::new(move |path: std::path::PathBuf| {
        let path_str = path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("non-UTF-8 path: {}", path.display()))?;
        let mut guard = engine
            .pool
            .checkout_blocking()
            .map_err(|e| anyhow::anyhow!("session pool closed: {e}"))?;
        Ok(engine.transcribe_file(path_str, &mut guard)?)
    })
}

/// Cancellation token fired on SIGINT, shared by the batch / watch runners for
/// graceful shutdown (finish in-flight files, stop scheduling new ones).
pub(crate) fn ctrl_c_token() -> tokio_util::sync::CancellationToken {
    let token = tokio_util::sync::CancellationToken::new();
    let inner = token.clone();
    tokio::spawn(async move {
        match tokio::signal::ctrl_c().await {
            Ok(()) => inner.cancel(),
            Err(e) => tracing::warn!("Failed to listen for Ctrl-C: {e}"),
        }
    });
    token
}

/// Build the [`batch::BatchOptions`] shared by `transcribe-batch` and `watch`
/// from the parsed CLI flags.
pub(crate) fn build_batch_options(
    input_dir: &str,
    output_dir: &str,
    pool_size: usize,
    retries: u32,
    out: &BatchOutputArgs,
) -> anyhow::Result<batch::BatchOptions> {
    Ok(batch::BatchOptions {
        input_dir: std::path::PathBuf::from(input_dir),
        output_dir: std::path::PathBuf::from(output_dir),
        formats: batch::parse_formats(&out.format).map_err(|e| anyhow::anyhow!("{e}"))?,
        render_opts: RenderOpts {
            max_chars_per_line: out.max_chars_per_line.unwrap_or(80),
            max_words_per_line: out.max_words_per_line.unwrap_or(14),
            include_word_timestamps: out.word_timestamps,
        },
        move_to: out.move_to.as_deref().map(std::path::PathBuf::from),
        delete_source: out.delete_source,
        concurrency: pool_size,
        retries,
    })
}

/// Offline single-file transcription (telephony / stereo / mono).
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_transcribe(
    file: String,
    model_dir: String,
    model_variant: Option<ModelVariant>,
    punctuation: PunctuationMode,
    punct_model_dir: String,
    itn: ItnMode,
    hotwords_file: Option<String>,
    hotwords_default: bool,
    hotwords_boost: Option<f32>,
    vad: bool,
    vad_threshold: Option<f32>,
    vad_min_silence_ms: Option<u32>,
    vad_model_dir: String,
    encoder_intra_threads: Option<usize>,
    format: String,
    output: Option<String>,
    max_chars_per_line: Option<usize>,
    max_words_per_line: Option<usize>,
    word_timestamps: bool,
    stereo_speakers: bool,
    codec: Option<String>,
    sample_rate: Option<u32>,
) -> anyhow::Result<()> {
    // Single-triplet pool for offline file transcription; when the
    // thread count is unset it defaults to every logical CPU (one
    // running triplet), else the explicit value is used as-is.
    let engine = EngineRecipe::offline(
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
        1,
    )
    .load_offline_engine()
    .await?;
    let mut guard = engine.pool.checkout().await?;
    let result = if let Some(codec_name) = codec.as_deref() {
        // Raw headerless telephony input: decode via the codec tables
        // straight to mono 16 kHz f32 and hand the samples to the engine.
        let telephony_codec =
            inference::audio::TelephonyCodec::from_name(codec_name).ok_or_else(|| {
                anyhow::anyhow!("unsupported codec '{codec_name}' (supported: pcmu, pcma, g722)")
            })?;
        // clap enforces `--sample-rate` when `--codec` is given; keep a
        // graceful error instead of an unwrap in case that ever changes.
        let rate =
            sample_rate.ok_or_else(|| anyhow::anyhow!("--sample-rate is required with --codec"))?;
        telephony_codec
            .validate_sample_rate(rate)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let raw =
            std::fs::read(&file).with_context(|| format!("Failed to open audio file: {file}"))?;
        let mut samples = inference::audio::decode_telephony_raw(&raw, telephony_codec, rate)?;
        // NOT a no-op: preserve transcript byte-identity. The former path
        // encoded these samples into an in-memory WAV and let the engine
        // decode it back, which snapped every value to 16-bit PCM. Passing
        // the raw f32 straight through would change the transcript (measured:
        // 153 token edits on a 9-minute call), so reproduce that quantization
        // in place — clamp to [-1, 1], round via 32767, normalise via 32768,
        // matching `encode_wav_pcm16` and the PCM decode path. Whether
        // full-precision f32 is better is an open WER question tracked in
        // roadmap/ (telephony precision); do not silently drop this snap.
        for s in samples.iter_mut() {
            let v = if s.is_finite() {
                s.clamp(-1.0, 1.0)
            } else {
                0.0
            };
            *s = (v * 32767.0).round() as i16 as f32 / 32768.0;
        }
        engine.transcribe_request(
            inference::TranscribeRequest::new(inference::TranscribeSource::Samples(&samples)),
            &mut guard,
        )
    } else if stereo_speakers {
        let channels = inference::audio::load_audio_channels(&file)?;
        let fallback_reason = match channels.len() {
            0 => Some("no channels"),
            1 => Some("mono audio"),
            2 if inference::audio::is_dual_mono(&channels) => Some("dual-mono audio"),
            n if n > 2 => Some("more than two channels"),
            _ => None,
        };
        if let Some(reason) = fallback_reason {
            tracing::warn!(
                "--stereo-speakers requested but {reason} detected; falling back to mono transcription"
            );
            engine.transcribe_file(&file, &mut guard)
        } else {
            engine.transcribe_channels(&channels, &mut guard)
        }
    } else {
        engine.transcribe_file(&file, &mut guard)
    };
    drop(guard);
    let result = result?;

    let format = ExportFormat::from_str(&format).map_err(|e| anyhow::anyhow!("{e}"))?;
    let opts = RenderOpts {
        max_chars_per_line: max_chars_per_line.unwrap_or(80),
        max_words_per_line: max_words_per_line.unwrap_or(14),
        include_word_timestamps: word_timestamps,
    };
    let rendered = format.render(&result, &opts);

    match output {
        Some(path) => {
            std::fs::write(&path, rendered).with_context(|| format!("failed to write {path}"))?;
            tracing::info!("Wrote {} export to {path}", format);
        }
        None => println!("{rendered}"),
    }
    Ok(())
}

pub(crate) async fn run_transcribe_batch(
    input_dir: String,
    output_dir: String,
    eng: OfflineEngineArgs,
    out: BatchOutputArgs,
) -> anyhow::Result<()> {
    let engine = EngineRecipe::offline(
        eng.model_dir,
        eng.model_variant,
        eng.punctuation,
        eng.punct_model_dir,
        eng.itn,
        eng.hotwords_file,
        eng.hotwords_default,
        eng.hotwords_boost,
        eng.vad,
        eng.vad_threshold,
        eng.vad_min_silence_ms,
        eng.vad_model_dir,
        eng.encoder_intra_threads,
        eng.pool_size,
    )
    .load_offline_engine()
    .await?;
    let opts = build_batch_options(
        &input_dir,
        &output_dir,
        eng.pool_size,
        out.retries.unwrap_or(0),
        &out,
    )?;
    let summary = batch::run_batch(
        &opts,
        make_transcribe_fn(std::sync::Arc::new(engine)),
        ctrl_c_token(),
    )
    .await?;
    tracing::info!(
        processed = summary.processed,
        failed = summary.failed,
        skipped = summary.skipped,
        "batch finished"
    );
    if summary.interrupted {
        // Same contract as `download`: SIGINT exits 130.
        std::process::exit(130);
    }
    if summary.failed > 0 {
        std::process::exit(1);
    }
    Ok(())
}

pub(crate) async fn run_watch(
    input_dir: String,
    output_dir: String,
    eng: OfflineEngineArgs,
    out: BatchOutputArgs,
    poll_interval_ms: u64,
    settle_polls: u32,
) -> anyhow::Result<()> {
    let engine = EngineRecipe::offline(
        eng.model_dir,
        eng.model_variant,
        eng.punctuation,
        eng.punct_model_dir,
        eng.itn,
        eng.hotwords_file,
        eng.hotwords_default,
        eng.hotwords_boost,
        eng.vad,
        eng.vad_threshold,
        eng.vad_min_silence_ms,
        eng.vad_model_dir,
        eng.encoder_intra_threads,
        eng.pool_size,
    )
    .load_offline_engine()
    .await?;
    let opts = batch::WatchOptions {
        batch: build_batch_options(
            &input_dir,
            &output_dir,
            eng.pool_size,
            out.retries.unwrap_or(2),
            &out,
        )?,
        poll_interval: std::time::Duration::from_millis(poll_interval_ms),
        settle_polls,
    };
    let summary = batch::run_watch(
        &opts,
        make_transcribe_fn(std::sync::Arc::new(engine)),
        ctrl_c_token(),
    )
    .await?;
    tracing::info!(
        processed = summary.processed,
        failed = summary.failed,
        "watch stopped"
    );
    if summary.failed > 0 {
        std::process::exit(1);
    }
    Ok(())
}
