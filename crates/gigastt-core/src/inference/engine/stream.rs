//! `impl Engine` methods — split out of the former god-file.
use super::*;
impl Engine {
    /// Return `true` if a speaker model file was present at boot and diarization
    /// can be requested. The ONNX session may still be unloaded until the first
    /// diarization request (lazy load).
    #[cfg(feature = "diarization")]
    pub fn has_speaker_encoder(&self) -> bool {
        self.speaker_encoder.is_some()
    }

    /// Create a fresh streaming state for a new connection.
    ///
    /// Pass `diarization_enabled = true` to activate speaker diarization for
    /// this session. Without the `diarization` feature or a speaker model file,
    /// the flag is silently ignored (a `warn!` is emitted when the caller asked
    /// for diarization but the build does not support it, so the contract
    /// mismatch is visible in logs). Enabling diarization loads the speaker
    /// encoder on first use if it was only probed at boot.
    pub fn create_state(&self, diarization_enabled: bool) -> StreamingState {
        #[cfg(feature = "diarization")]
        let diarization_state = if diarization_enabled {
            match self.speaker_encoder.as_ref() {
                Some(lazy) => lazy
                    .get_or_load()
                    .and_then(|enc| diarization::open_streaming(&enc)),
                None => {
                    tracing::warn!(
                        "diarization_enabled=true ignored: wespeaker model not present at engine boot"
                    );
                    None
                }
            }
        } else {
            None
        };

        #[cfg(not(feature = "diarization"))]
        if diarization_enabled {
            tracing::warn!(
                "diarization_enabled=true ignored: build lacks the `diarization` feature"
            );
        }

        StreamingState {
            decoder: DecoderState::new(self.tokenizer.blank_id()),
            audio_buffer: Vec::new(),
            assembler: TranscriptAssembler::new(),
            window_start_samples: 0,
            context_samples: 0,
            pending_samples: 0,
            resampler: None,
            mel_fft_input: Vec::new(),
            mel_power: Vec::new(),
            mel_output: Vec::new(),
            resample_output_buf: Vec::new(),
            vad_endpointer: self
                .vad
                .as_ref()
                .map(|_| crate::vad::VadEndpointer::new(&self.vad_config)),
            punctuation: None,
            itn: None,
            endpoint_mode: self.endpoint_mode,
            #[cfg(feature = "diarization")]
            diarization_state,
        }
    }

    /// Process a chunk of 16kHz f32 audio samples and return any new transcript segments.
    ///
    /// Returns [`TranscriptSegment`] with `is_final == false` during speech (Partial),
    /// and `is_final == true` only on a **true utterance endpoint**:
    ///
    /// - decoder blank-run (~600 ms) when no VAD and [`EndpointMode::Auto`];
    /// - VAD trailing silence when a VAD is attached (and mode is not `Manual`);
    /// - never on the ~2.5 s encoder window cap (cap commits a stable prefix and
    ///   emits a non-final partial so voice assistants do not treat a slide as
    ///   "command complete").
    ///
    /// Streaming state (LSTM hidden/cell, leftover audio, accumulated text) is
    /// maintained in `state`.
    ///
    /// # Errors
    ///
    /// Returns [`GigasttError::Inference`] if the ONNX runtime fails.
    pub fn process_chunk(
        &self,
        samples: &[f32],
        state: &mut StreamingState,
        triplet: &mut SessionTriplet,
    ) -> Result<Vec<TranscriptSegment>, GigasttError> {
        if samples.is_empty() {
            return Ok(vec![]);
        }

        // Diarization tracks speakers continuously, so feed every chunk's audio
        // even when this chunk doesn't trigger a decode (see the stride gate).
        #[cfg(feature = "diarization")]
        if let Some(dia) = state.diarization_state.as_mut() {
            diarization::feed_chunk(dia, samples);
        }

        // Sliding-window streaming: accumulate audio; the encoder re-runs on the
        // whole retained window so the offline Conformer always has left context
        // (an isolated ~100ms chunk decodes to garbage). Re-decoding is the cost,
        // so we only decode once STREAM_DECODE_STRIDE_SAMPLES of NEW audio have
        // arrived (or the window hit its cap) — this keeps the engine real-time.
        // The window is bounded by `self.stream_max_window_samples` (default
        // 2.5s, configurable at serve time); on endpoint or cap we finalize the
        // tail and slide, retaining STREAM_LEFT_CONTEXT_SAMPLES.
        state.audio_buffer.extend_from_slice(samples);
        state.pending_samples += samples.len();

        // Feed the VAD on every chunk so trailing silence is tracked
        // continuously (independent of the decode stride). A VAD endpoint forces
        // a decode + finalize this chunk even if the stride gate wouldn't fire.
        // VAD is non-blocking: an inference error is logged and ignored, leaving
        // the window cap as the only backstop until the VAD recovers. With no
        // VAD attached `vad_endpoint` is always false and the decoder's
        // blank-run heuristic keeps owning endpointing, byte-for-byte unchanged.
        let mut vad_endpoint = false;
        if let (Some(vad), Some(ep)) = (self.vad.as_ref(), state.vad_endpointer.as_mut()) {
            match ep.push(vad, samples) {
                Ok(fired) => vad_endpoint = fired,
                Err(e) => tracing::warn!("VAD endpoint detection failed: {e:#}"),
            }
        }

        let over_cap = state.audio_buffer.len() >= self.stream_max_window_samples;
        // Stride gate on NEW audio since the last decode (not since the last
        // slide): otherwise a non-finalizing partial would leave the counter
        // high and decode on every subsequent chunk. A VAD endpoint overrides
        // the gate so the utterance finalizes promptly.
        if state.pending_samples < STREAM_DECODE_STRIDE_SAMPLES && !over_cap && !vad_endpoint {
            return Ok(vec![]);
        }
        // Too little audio to extract a frame. Skip — but never when finalizing:
        // a fired VAD endpoint (or cap) must still flush the assembler below,
        // even though `decode_window` will add no new words from a sub-frame
        // buffer. (In practice a VAD endpoint needs ≥ min_silence_ms of trailing
        // audio, so the buffer is always ≫ N_FFT here; this guards the edge.)
        if state.audio_buffer.len() < N_FFT && !vad_endpoint && !over_cap {
            return Ok(vec![]);
        }

        let endpoint = self
            .decode_window(state, triplet)
            .map_err(|e| GigasttError::Inference { source: e.into() })?;
        state.pending_samples = 0;
        let ts = now_timestamp();

        // True utterance endpoints only — never the encoder window cap.
        // Cap used to emit `final` as a "backstop", which voice assistants
        // (Irene) treated as "command complete" mid-phrase; it now commits a
        // stable prefix and emits a non-final partial instead.
        let speech_endpoint = Self::speech_endpoint(
            state.endpoint_mode,
            endpoint,
            vad_endpoint,
            state.vad_endpointer.is_some(),
        );

        if speech_endpoint {
            let reason = if vad_endpoint {
                EndpointReason::Vad
            } else {
                EndpointReason::Blank
            };
            let mut seg = state.assembler.finalize_with_reason(ts, reason);
            self.enrich_final_segment(&mut seg, state);
            Self::slide_streaming_window(state);
            if seg.text.trim().is_empty() {
                return Ok(vec![]);
            }
            return Ok(vec![seg]);
        }

        if over_cap {
            // Encoder cost bound: commit live words so they are not lost when
            // the window slides, but do **not** end the utterance.
            let live = state.assembler.live_word_count();
            tracing::debug!(
                committed = live,
                window_samples = state.audio_buffer.len(),
                "stream window cap: committed stable prefix, sliding"
            );
            state.assembler.commit_live();
            Self::slide_streaming_window(state);
            if state.assembler.is_empty() {
                return Ok(vec![]);
            }
            return Ok(vec![state.assembler.partial(ts)]);
        }

        if state.assembler.is_empty() {
            return Ok(vec![]);
        }
        Ok(vec![state.assembler.partial(ts)])
    }

    /// Whether this decode should close the utterance (`final` / `speech_final`).
    ///
    /// Pure helper so endpoint policy is unit-testable without ONNX.
    pub(crate) fn speech_endpoint(
        mode: EndpointMode,
        decoder_blank_endpoint: bool,
        vad_endpoint: bool,
        vad_attached: bool,
    ) -> bool {
        let decoder_endpoint = decoder_blank_endpoint && !vad_attached;
        match mode {
            EndpointMode::Auto => decoder_endpoint || vad_endpoint,
            // Assistant: only VAD silence (when attached). Blank-run alone is too
            // aggressive for multi-word voice commands.
            EndpointMode::Assistant => vad_endpoint,
            EndpointMode::Manual => false,
        }
    }

    /// Slide the streaming audio window, retaining left context for the next decode.
    pub(crate) fn slide_streaming_window(state: &mut StreamingState) {
        let keep = STREAM_LEFT_CONTEXT_SAMPLES.min(state.audio_buffer.len());
        let slide_off = state.audio_buffer.len() - keep;
        if slide_off > 0 {
            audio::consume_audio_buffer(&mut state.audio_buffer, slide_off);
            state.window_start_samples += slide_off;
        }
        state.context_samples = keep;
    }

    /// Re-decode the whole retained window from a fresh decoder state and update
    /// the assembler with the context-suppressed tail. Returns whether the
    /// decoder detected an endpoint. Shared by [`Engine::process_chunk`]
    /// (strided) and [`Engine::finish_stream`] (forced at end of stream).
    pub(crate) fn decode_window(
        &self,
        state: &mut StreamingState,
        triplet: &mut SessionTriplet,
    ) -> anyhow::Result<bool> {
        let mel_start = std::time::Instant::now();
        let num_frames = self.features.compute_mel(
            &state.audio_buffer,
            &mut state.mel_fft_input,
            &mut state.mel_power,
            &mut state.mel_output,
        );
        tracing::debug!(
            elapsed_us = mel_start.elapsed().as_micros() as u64,
            "mel_compute"
        );
        if num_frames == 0 {
            return Ok(false);
        }

        // Encoder-frame offset of the window start (drift-free: a single division
        // over the cumulative slid-off sample count).
        let frame_offset = state.window_start_samples / (HOP_LENGTH * ENCODER_SUBSAMPLING);

        // The window overlaps the previous one, so persisting the LSTM state
        // would double-condition the prediction network — decode fresh.
        let mut decoder_state = DecoderState::new(self.tokenizer.blank_id());
        // Streaming always uses the engine boot biaser (per-request hotwords
        // apply to file-transcription paths that carry TranscribeOverrides).
        let (all_words, endpoint) = self.run_inference(
            triplet,
            &state.mel_output[..],
            num_frames,
            &mut decoder_state,
            frame_offset,
            true, // streaming: ANE low-latency pad floor when available
            self.biaser.as_ref(),
        )?;

        // Suppress words inside the already-emitted left context so a slid
        // window does not re-emit committed words.
        let window_start_s = frame_offset as f64 * SECONDS_PER_FRAME;
        let context_boundary_s = window_start_s + state.context_samples as f64 / 16000.0;
        let decoded = all_words.len();
        #[cfg_attr(not(feature = "diarization"), allow(unused_mut))]
        let mut tail: Vec<WordInfo> = all_words
            .into_iter()
            .filter(|w| w.start + f64::EPSILON >= context_boundary_s)
            .collect();

        // Per-pass visibility for stream-vs-file divergence analysis:
        // decoded = full window hypothesis, suppressed = dropped context words,
        // replaced = previous live tail this hypothesis overwrites.
        tracing::debug!(
            decoded,
            suppressed = decoded - tail.len(),
            replaced = state.assembler.live_word_count(),
            live = tail.len(),
            "stream decode window"
        );

        #[cfg(feature = "diarization")]
        if let Some(dia) = state.diarization_state.as_mut()
            && let Some(speaker) = diarization::last_turn_speaker(dia)
        {
            for w in &mut tail {
                w.speaker = Some(speaker);
            }
        }

        state.assembler.set_words(tail);
        Ok(endpoint)
    }

    /// Decode any audio buffered since the last strided decode, then finalize.
    /// Call when the stream ends (Stop / EOF) so the decode-stride batching does
    /// not drop trailing words. Best-effort: on decode failure, falls back to a
    /// plain flush of whatever the assembler already holds.
    pub fn finish_stream(
        &self,
        state: &mut StreamingState,
        triplet: &mut SessionTriplet,
    ) -> Option<TranscriptSegment> {
        let has_pending = state.pending_samples > 0 && state.audio_buffer.len() >= N_FFT;
        if has_pending && let Err(e) = self.decode_window(state, triplet) {
            tracing::warn!("finish_stream decode failed: {e:#}");
        }
        self.flush_state(state)
    }

    /// Flush accumulated text as a Final segment (called on Stop/Close).
    pub fn flush_state(&self, state: &mut StreamingState) -> Option<TranscriptSegment> {
        if state.assembler.is_empty() {
            return None;
        }
        let mut seg = state
            .assembler
            .finalize_with_reason(now_timestamp(), EndpointReason::Stop);
        self.enrich_final_segment(&mut seg, state);
        Some(seg)
    }

    /// Post-process a finalized streaming segment: ITN, then punctuation/casing
    /// restoration on the joined `text`. Mirrors
    /// [`Engine::finish_transcribe_result`]'s policy exactly — the per-session
    /// override wins over the engine boot default (`None` keeps it), and the
    /// `punctuator` guard makes the pass a graceful no-op when no punct model
    /// is attached. Word payloads keep the raw decoder output, exactly like the
    /// file path. Runs only at finalization boundaries (endpoint flush /
    /// Stop-flush), so `partial` payloads are never rewritten and live previews
    /// don't flicker between hypotheses.
    ///
    /// Latency: measured via `punctuation::tests::test_restore_latency_short_segments`
    /// (debug build, Apple Silicon), `restore` costs p95 ≈ 0.45–1.0 ms on 1–10
    /// word segments — roughly two orders of magnitude below the 100 ms budget
    /// that would force a segment-length gate, so enrichment always runs
    /// regardless of segment length.
    pub(crate) fn enrich_final_segment(&self, seg: &mut TranscriptSegment, state: &StreamingState) {
        let text = std::mem::take(&mut seg.text);
        seg.text = self.apply_text_postprocess(
            text,
            state.itn.unwrap_or(self.itn),
            state.punctuation.unwrap_or(true),
        );
    }
}
