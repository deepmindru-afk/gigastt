//! `impl Engine` methods — split out of the former god-file.
use super::*;
impl Engine {
    /// Unified file-transcription entry point. Prefer this over the
    /// combinatorial `transcribe_*` wrappers when building new call sites.
    pub fn transcribe_request(
        &self,
        req: TranscribeRequest<'_>,
        triplet: &mut SessionTriplet,
    ) -> Result<TranscribeResult, GigasttError> {
        use std::sync::atomic::Ordering::Relaxed;
        // Bridge the request's shared cancellation / progress handles into the
        // lightweight closures the decode chain consults. Both own a clone of
        // the `Arc`, so they outlive the borrow of `req` and stay valid for the
        // whole call. Absent handles leave `ctl` all-`None`, and every decode
        // function then runs its historical, byte-identical path.
        let abort_fn: Option<Box<dyn Fn() -> bool>> = req.abort.as_ref().map(|flag| {
            let flag = flag.clone();
            Box::new(move || flag.load(Relaxed)) as Box<dyn Fn() -> bool>
        });
        let progress_fn: Option<Box<dyn Fn(u64)>> = req.progress.as_ref().map(|counter| {
            let counter = counter.clone();
            Box::new(move |n: u64| counter.store(n, Relaxed)) as Box<dyn Fn(u64)>
        });
        let ctl = DecodeControls {
            abort: abort_fn.as_deref(),
            on_progress: progress_fn.as_deref(),
        };

        // Opt-in operator length limit (`--max-audio-secs`); `None` = unlimited.
        // The streaming path honors it verbatim; the whole-buffer decoders clamp
        // it down to their fixed safety ceiling.
        let max_audio_secs = req.max_audio_secs;

        match req.source {
            #[cfg(feature = "file-decode")]
            TranscribeSource::Path(path) => {
                if self.stream_eligible(req.diarization) {
                    self.transcribe_stream_file(
                        |spec| {
                            audio::FileWindows::open(path, spec, max_audio_secs)
                                .map_err(audio::decode_error)
                        },
                        triplet,
                        &req.overrides,
                        req.hotwords,
                        ctl,
                    )
                } else {
                    let float_samples = audio::decode_audio_file_bounded(path, max_audio_secs)
                        .map_err(audio::decode_error)?;
                    self.transcribe_samples_with_overrides(
                        &float_samples,
                        triplet,
                        &req.overrides,
                        req.hotwords,
                        req.diarization,
                        req.diarization_outcome.as_deref(),
                        ctl,
                    )
                }
            }
            #[cfg(feature = "file-decode")]
            TranscribeSource::Bytes(data) => {
                if self.stream_eligible(req.diarization) {
                    self.transcribe_stream_file(
                        |spec| {
                            audio::FileWindows::from_bytes(data.clone(), spec, max_audio_secs)
                                .map_err(audio::decode_error)
                        },
                        triplet,
                        &req.overrides,
                        req.hotwords,
                        ctl,
                    )
                } else {
                    let float_samples =
                        audio::decode_audio_bytes_shared_bounded(data, max_audio_secs)
                            .map_err(audio::decode_error)?;
                    self.transcribe_samples_with_overrides(
                        &float_samples,
                        triplet,
                        &req.overrides,
                        req.hotwords,
                        req.diarization,
                        req.diarization_outcome.as_deref(),
                        ctl,
                    )
                }
            }
            TranscribeSource::Samples(samples) => self.transcribe_samples_with_overrides(
                samples,
                triplet,
                &req.overrides,
                req.hotwords,
                req.diarization,
                req.diarization_outcome.as_deref(),
                ctl,
            ),
            #[cfg(feature = "file-decode")]
            TranscribeSource::ChannelStreams { data, channels } => self.transcribe_channel_streams(
                data,
                channels,
                max_audio_secs,
                triplet,
                &req.overrides,
                req.hotwords,
                ctl.abort_only(),
            ),
            TranscribeSource::Channels(channels) => self.transcribe_channels_inner(
                channels,
                triplet,
                &req.overrides,
                req.hotwords,
                ctl.abort_only(),
            ),
        }
    }

    /// True when a `Path` / `Bytes` request can be served by the windowed
    /// streaming decode, whose peak audio memory is O(one window) rather than
    /// O(file).
    ///
    /// Diarization embeds the whole clip in a second pass, so it cannot run
    /// against a stream that is never fully resident: it still forces the
    /// whole-buffer decode, and with it the duration ceiling. VAD does not — it
    /// is causal, and [`VadWindows`](audio::VadWindows) runs it inside the
    /// stream. `diarize` only matters with the `diarization` feature compiled
    /// in.
    #[cfg(feature = "file-decode")]
    pub(crate) fn stream_eligible(&self, diarize: bool) -> bool {
        !(cfg!(feature = "diarization") && diarize)
    }

    /// Streaming mono file transcription, with the VAD stage in front when one
    /// is attached and the request did not opt out.
    ///
    /// `open` builds a fresh window source for a given geometry. It is called
    /// once, or twice when the VAD stage declines — the model failed mid-stream,
    /// or the scan found no speech at all in a non-empty clip — in which case
    /// the clip is decoded whole, exactly as the batch path did.
    #[cfg(feature = "file-decode")]
    pub(crate) fn transcribe_stream_file(
        &self,
        open: impl Fn(WindowSpec) -> Result<audio::FileWindows, GigasttError>,
        triplet: &mut SessionTriplet,
        overrides: &TranscribeOverrides,
        hotwords: Option<&HotwordOverride>,
        ctl: DecodeControls,
    ) -> Result<TranscribeResult, GigasttError> {
        let use_vad = self.vad.is_some() && overrides.vad.unwrap_or(true);
        if let (true, Some(vad)) = (use_vad, &self.vad) {
            let wall_start = std::time::Instant::now();
            let request_biaser = hotwords.and_then(|hw| self.build_request_biaser(hw));
            let biaser = self.select_biaser(hotwords, &request_biaser);
            let mut windows = audio::VadWindows::new(
                open(audio::VadWindows::pull_spec())?,
                vad,
                &self.vad_config,
                window_spec(self.ane_encoder, self.variant.is_ctc()),
                ctl.abort,
            );
            let mut words = self.decode_words_streaming(&mut windows, triplet, biaser, ctl)?;
            if !windows.needs_fallback() {
                // Words are decoded on the compressed (silence-removed)
                // timeline; put them back on the clip's own.
                let regions = windows.regions();
                for w in &mut words {
                    w.start = crate::vad::remap_compressed_seconds(w.start, regions, 16000.0);
                    w.end = crate::vad::remap_compressed_seconds(w.end, regions, 16000.0);
                }
                let duration_s = windows.total_16k_samples() as f64 / 16000.0;
                let wall_s = wall_start.elapsed().as_secs_f64();
                tracing::info!(
                    audio_s = format_args!("{duration_s:.2}"),
                    wall_s = format_args!("{wall_s:.2}"),
                    rtf = format_args!(
                        "{:.3}",
                        if duration_s > 0.0 {
                            wall_s / duration_s
                        } else {
                            0.0
                        }
                    ),
                    regions = regions.len(),
                    "transcribe complete (streaming windows, vad)"
                );
                return Ok(self.finish_transcribe_result(words, duration_s, overrides));
            }
            // Either the VAD found no speech at all — tone or continuous speech
            // against a bad threshold — or the model failed mid-stream (already
            // logged with the cause). Both re-read the clip and decode it whole
            // rather than returning an empty transcript.
            tracing::warn!("VAD produced no usable speech regions; decoding full audio");
        }
        self.transcribe_stream_mono(
            open(window_spec(self.ane_encoder, self.variant.is_ctc()))?,
            triplet,
            overrides,
            hotwords,
            ctl,
        )
    }

    /// Mono file-transcription tail that pulls windows straight from the
    /// container instead of decoding the whole file first.
    ///
    /// Equivalent to [`Engine::transcribe_samples_with_overrides`] on the
    /// no-VAD, no-diarization path: [`FileWindows`](audio::FileWindows) yields
    /// exactly the geometry [`SliceWindows`] would over the fully-decoded buffer
    /// (one window for a single-pass-length stream, overlapping windows beyond),
    /// so [`Engine::decode_words_streaming`] produces the same words — but peak
    /// audio memory no longer scales with duration. The duration cap and its
    /// exact error string are enforced inside the decode as before.
    #[cfg(feature = "file-decode")]
    pub(crate) fn transcribe_stream_mono(
        &self,
        mut windows: audio::FileWindows,
        triplet: &mut SessionTriplet,
        overrides: &TranscribeOverrides,
        hotwords: Option<&HotwordOverride>,
        ctl: DecodeControls,
    ) -> Result<TranscribeResult, GigasttError> {
        let wall_start = std::time::Instant::now();

        // Hotword biaser: engine boot biaser, temporary per-request, or off.
        let request_biaser = hotwords.and_then(|hw| self.build_request_biaser(hw));
        let biaser = self.select_biaser(hotwords, &request_biaser);

        let words = self.decode_words_streaming(&mut windows, triplet, biaser, ctl)?;
        // Exact once every window is consumed (the loop above drains to EOF).
        let duration_s = windows.total_16k_samples() as f64 / 16000.0;
        let result = self.finish_transcribe_result(words, duration_s, overrides);

        let wall_s = wall_start.elapsed().as_secs_f64();
        let rtf = if duration_s > 0.0 {
            wall_s / duration_s
        } else {
            0.0
        };
        tracing::info!(
            audio_s = format_args!("{duration_s:.2}"),
            wall_s = format_args!("{wall_s:.2}"),
            rtf = format_args!("{rtf:.3}"),
            "transcribe complete (streaming windows)"
        );

        Ok(result)
    }

    /// Run the full mel + encoder + RNN-T decode pipeline on an already-decoded
    /// 16 kHz f32 sample buffer. Shared tail of [`Engine::transcribe_request`]
    /// for mono sources (and unit tests).
    pub(crate) fn transcribe_samples(
        &self,
        float_samples: &[f32],
        triplet: &mut SessionTriplet,
    ) -> Result<TranscribeResult, GigasttError> {
        self.transcribe_request(
            TranscribeRequest::new(TranscribeSource::Samples(float_samples)),
            triplet,
        )
    }

    /// Override-aware tail of the file-transcription pipeline. With
    /// `TranscribeOverrides::default()` (all `None`) it is byte-for-byte the
    /// engine-default path; each `Some(_)` field flips the corresponding
    /// post-processing knob for this call only. `overrides` is assumed already
    /// validated by [`Engine::validate_overrides`] — an on-request with the
    /// resource missing degrades gracefully (VAD absent → whole-buffer decode)
    /// rather than erroring here.
    /// Override-aware tail of the file-transcription pipeline. With
    /// `TranscribeOverrides::default()` (all `None`) it is byte-for-byte the
    /// engine-default path; each `Some(_)` field flips the corresponding
    /// post-processing knob for this call only. `overrides` is assumed already
    /// validated by [`Engine::validate_overrides`] — an on-request with the
    /// resource missing degrades gracefully (VAD absent → whole-buffer decode)
    /// rather than erroring here.
    #[allow(clippy::too_many_arguments)] // overrides + diarization sink + controls; bundle later if it grows again
    pub(crate) fn transcribe_samples_with_overrides(
        &self,
        float_samples: &[f32],
        triplet: &mut SessionTriplet,
        overrides: &TranscribeOverrides,
        hotwords: Option<&HotwordOverride>,
        diarize: bool,
        diar_sink: Option<&std::sync::OnceLock<DiarizationOutcome>>,
        ctl: DecodeControls,
    ) -> Result<TranscribeResult, GigasttError> {
        // `diarize` is opt-in per request: offline speaker diarization only runs
        // when the caller asked for it (REST `?diarization=true`). A plain
        // transcript — and the `channels=split` dual-mono fallback — must carry no
        // speaker labels, so the default paths pass `false`.
        let wall_start = std::time::Instant::now();
        let duration_s = float_samples.len() as f64 / 16000.0;

        #[cfg_attr(not(feature = "diarization"), allow(unused_mut))]
        let mut words =
            self.decode_words_for_samples(float_samples, triplet, overrides, hotwords, ctl)?;

        // Record *why* speakers were or were not labeled into the caller's sink
        // so a `?diarization=true` request that produced no labels can be
        // surfaced with a reason instead of an all-empty-speaker transcript.
        #[cfg(feature = "diarization")]
        if diarize {
            let outcome = match self
                .speaker_encoder
                .as_ref()
                .and_then(|lazy| lazy.get_or_load())
            {
                None => DiarizationOutcome::NoSpeakerModel,
                Some(enc) => match diarization::run_offline(&enc, float_samples) {
                    Ok(turns) => {
                        diarization::assign_speakers_by_midpoint(&turns, &mut words);
                        DiarizationOutcome::Applied
                    }
                    Err(declined) => declined,
                },
            };
            if let Some(sink) = diar_sink {
                let _ = sink.set(outcome);
            }
        }
        // A build compiled without the `diarization` feature can never label
        // speakers; report that per request rather than silently dropping the flag.
        #[cfg(not(feature = "diarization"))]
        if diarize && let Some(sink) = diar_sink {
            let _ = sink.set(DiarizationOutcome::NoSpeakerModel);
        }

        let result = self.finish_transcribe_result(words, duration_s, overrides);

        let wall_s = wall_start.elapsed().as_secs_f64();
        let rtf = if duration_s > 0.0 {
            wall_s / duration_s
        } else {
            0.0
        };
        let encoder_label = if self.int8 { "int8" } else { "fp32" };
        let backend_label = if cfg!(feature = "candle") {
            "candle"
        } else if cfg!(feature = "ane") {
            "ane"
        } else if cfg!(feature = "coreml") {
            "coreml"
        } else if cfg!(feature = "cuda") {
            "cuda"
        } else {
            "cpu"
        };
        tracing::info!(
            audio_s = format_args!("{duration_s:.2}"),
            wall_s = format_args!("{wall_s:.2}"),
            rtf = format_args!("{rtf:.3}"),
            encoder = format_args!("{encoder_label}/{backend_label}"),
            "transcribe complete"
        );

        Ok(result)
    }

    /// Build the final [`TranscribeResult`] from raw words: join text, apply ITN,
    /// and apply punctuation restoration. Word-level timing is left untouched.
    /// Per-request overrides win over engine defaults; a `None` override keeps
    /// the boot policy.
    pub(crate) fn finish_transcribe_result(
        &self,
        words: Vec<WordInfo>,
        duration_s: f64,
        overrides: &TranscribeOverrides,
    ) -> TranscribeResult {
        let text: String = words
            .iter()
            .map(|w| w.word.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        // Optional ITN then punctuation. Per-request override wins over the
        // engine default; `None` keeps the boot policy. Word-level timing is
        // left untouched — only the joined `text` is rewritten.
        let text = self.apply_text_postprocess(
            text,
            overrides.itn.unwrap_or(self.itn),
            overrides.punctuation.unwrap_or(true),
        );

        TranscribeResult {
            text,
            confidence: aggregate_confidence(&words),
            words,
            duration_s,
        }
    }
}
