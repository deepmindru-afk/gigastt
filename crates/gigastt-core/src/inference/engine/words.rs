//! Windowed / VAD-region word decode for [`Engine`].

use super::*;

impl Engine {
    /// Decode a 16 kHz f32 buffer to words: single-pass for short inputs (one
    /// encoder Run over the whole buffer), chunked overlapping windows for long
    /// inputs so encoder activation memory stays O(chunk), not O(file). Both
    /// paths produce the same `Vec<WordInfo>` shape. This is the no-VAD path and
    /// the per-region decode used by [`Engine::decode_speech_regions`].
    ///
    /// `biaser` is the effective hotword biaser for this call (engine boot
    /// biaser, a temporary per-request biaser, or `None` when forced off).
    pub(crate) fn decode_words(
        &self,
        samples: &[f32],
        triplet: &mut SessionTriplet,
        biaser: Option<&bias::Biaser>,
        ctl: DecodeControls,
    ) -> Result<Vec<WordInfo>, GigasttError> {
        let spec = window_spec(self.ane_encoder, self.variant.is_ctc());
        if spec.is_single_pass(samples.len()) {
            // A single-pass decode is one encoder Run with no window loop to
            // check between, so honour cancellation before starting it. The
            // caller's watchdog treats this whole clip as one "window".
            if ctl.aborted() {
                return Err(GigasttError::Cancelled);
            }
            let (features, num_frames) = self.features.compute(samples);
            tracing::info!("Extracted {} mel frames", num_frames);
            let mut decoder_state = DecoderState::new(self.tokenizer.blank_id());
            let words = self
                .run_inference(
                    triplet,
                    &features,
                    num_frames,
                    &mut decoder_state,
                    0,
                    false, // file-mode fill floor
                    biaser,
                )
                .map_err(|e| GigasttError::Inference { source: e.into() })?
                .0;
            ctl.report(samples.len() as u64);
            Ok(words)
        } else {
            tracing::info!(
                "Long-form chunked decode: {:.1}s in ~{}s windows ({}s overlap, ane={})",
                samples.len() as f64 / 16000.0,
                spec.window() / 16000,
                spec.overlap() / 16000,
                self.ane_encoder,
            );
            self.decode_words_streaming(&mut SliceWindows::new(samples, spec), triplet, biaser, ctl)
        }
    }

    /// Decode only the VAD-detected speech `regions` of `float_samples`: copy the
    /// speech spans into one silence-free buffer, decode it, then remap each
    /// word's start/end from the compressed (silence-removed) timeline back to
    /// the original timeline via [`crate::vad::remap_compressed_seconds`]. Empty
    /// `regions` (no speech) yields no words.
    pub(crate) fn decode_speech_regions(
        &self,
        float_samples: &[f32],
        regions: &[(usize, usize)],
        triplet: &mut SessionTriplet,
        biaser: Option<&bias::Biaser>,
        ctl: DecodeControls,
    ) -> Result<Vec<WordInfo>, GigasttError> {
        if regions.is_empty() {
            tracing::info!("VAD found no speech; skipping decode");
            return Ok(Vec::new());
        }
        let speech_len: usize = regions.iter().map(|(s, e)| e - s).sum();
        let mut speech = Vec::with_capacity(speech_len);
        for &(s, e) in regions {
            speech.extend_from_slice(&float_samples[s..e]);
        }
        tracing::info!(
            "VAD kept {}/{} samples ({} speech region(s))",
            speech_len,
            float_samples.len(),
            regions.len()
        );
        let mut words = self.decode_words(&speech, triplet, biaser, ctl)?;
        for w in &mut words {
            w.start = crate::vad::remap_compressed_seconds(w.start, regions, 16000.0);
            w.end = crate::vad::remap_compressed_seconds(w.end, regions, 16000.0);
        }
        Ok(words)
    }

    /// Long-form decode: pull overlapping windows from `windows`, encode and
    /// decode each independently with a fresh [`DecoderState`], offset each
    /// chunk's word timestamps by the chunk's absolute start, then stitch the
    /// per-chunk word lists with overlap de-dup via [`stitch_chunk_words`].
    ///
    /// Peak encoder activation memory is bounded by the source's window length
    /// (24s ort / 30s ANE, see [`super::windows::chunk_window_samples`]) rather than the full
    /// file length. Window starts are aligned to encoder frame boundaries
    /// (multiples of `HOP_LENGTH * ENCODER_SUBSAMPLING`) so the per-chunk frame
    /// offset is exact, matching the streaming path's math.
    ///
    /// Takes the PCM as a [`PcmWindows`] source rather than a `&[f32]`, so the
    /// loop makes no assumption that the whole file is in memory.
    pub(crate) fn decode_words_streaming(
        &self,
        windows: &mut dyn PcmWindows,
        triplet: &mut SessionTriplet,
        biaser: Option<&bias::Biaser>,
        ctl: DecodeControls,
    ) -> Result<Vec<WordInfo>, GigasttError> {
        let overlap = windows.spec().overlap();
        let frame_samples = HOP_LENGTH * ENCODER_SUBSAMPLING;

        let mut merged: Vec<WordInfo> = Vec::new();
        while let Some(window) = windows.next_window()? {
            // Cooperative cancellation checkpoint: a flipped abort flag ends the
            // run at this window boundary, so a cancelled request (client
            // disconnect, `DELETE /v1/jobs/{id}`, shutdown, or the no-progress
            // watchdog) frees its pooled session within one window instead of
            // decoding the rest of the file.
            if ctl.aborted() {
                return Err(GigasttError::Cancelled);
            }
            let start = window.start_sample;
            // Window ends advance monotonically, so this doubles as the
            // cumulative processed-sample count reported below.
            let win_end = start + window.samples.len();
            let (features, num_frames) = self.features.compute(window.samples);
            let frame_offset = start / frame_samples;
            let mut decoder_state = DecoderState::new(self.tokenizer.blank_id());
            let (chunk_words, _endpoint) = self
                .run_inference(
                    triplet,
                    &features,
                    num_frames,
                    &mut decoder_state,
                    frame_offset,
                    false, // file-mode fill floor
                    biaser,
                )
                .map_err(|e| GigasttError::Inference { source: e.into() })?;

            // Seam between the previous chunk's window and this one falls at the
            // midpoint of their overlap region, in absolute seconds.
            let overlap_mid_s = (start as f64 + overlap as f64 / 2.0) / 16000.0;
            merged = stitch_chunk_words(merged, chunk_words, overlap_mid_s);

            // A completed window is one unit of real progress: it resets the
            // server's no-progress deadline and advances a job's bar by the
            // seconds of audio just decoded (`win_end / 16000`).
            ctl.report(win_end as u64);
        }
        Ok(merged)
    }

    /// Decode a 16 kHz f32 buffer to words, applying VAD if configured.
    ///
    /// The VAD path is taken only when a VAD is attached AND the caller hasn't
    /// opted out via `overrides.vad`. `?vad=false` forces whole-buffer decode
    /// even on a VAD-enabled engine; `None` uses the engine default (VAD path
    /// iff a VAD is attached). `vad = Some(true)` on a VAD-less engine can't
    /// reach here — callers should validate overrides first — but the
    /// `self.vad.is_some()` guard keeps this correct regardless.
    ///
    /// Hotword selection via `hotwords`:
    /// - `None` → engine boot biaser
    /// - `Some(empty)` → force biasing off
    /// - `Some(phrases)` → temporary biaser built for this request only
    pub(crate) fn decode_words_for_samples(
        &self,
        float_samples: &[f32],
        triplet: &mut SessionTriplet,
        overrides: &TranscribeOverrides,
        hotwords: Option<&HotwordOverride>,
        ctl: DecodeControls,
    ) -> Result<Vec<WordInfo>, GigasttError> {
        // Temporary biaser only when the request supplies hotwords. Owned
        // here so the `Option<&Biaser>` passed into decode stays valid for
        // the whole call without cloning the engine's boot biaser.
        let request_biaser = hotwords.and_then(|hw| self.build_request_biaser(hw));
        let biaser = self.select_biaser(hotwords, &request_biaser);

        let use_vad = self.vad.is_some() && overrides.vad.unwrap_or(true);
        match (use_vad, &self.vad) {
            (true, Some(vad)) => {
                match vad.speech_regions_with_abort(float_samples, &self.vad_config, ctl.abort) {
                    Ok(regions) if regions.is_empty() => {
                        // Tone / continuous speech can yield zero regions on a bad
                        // threshold; fall back to fixed-window / full decode rather
                        // than returning an empty transcript.
                        tracing::warn!(
                            "VAD found no speech regions; falling back to full/chunked decode"
                        );
                        self.decode_words(float_samples, triplet, biaser, ctl)
                    }
                    Ok(regions) => {
                        self.decode_speech_regions(float_samples, &regions, triplet, biaser, ctl)
                    }
                    Err(e) => {
                        // A flipped abort flag stops the VAD scan too; surface it
                        // as cancellation instead of silently re-decoding the whole
                        // file, which would ignore the cancel and keep the triplet.
                        if ctl.aborted() {
                            return Err(GigasttError::Cancelled);
                        }
                        tracing::warn!("VAD failed, decoding full audio: {e:#}");
                        self.decode_words(float_samples, triplet, biaser, ctl)
                    }
                }
            }
            _ => self.decode_words(float_samples, triplet, biaser, ctl),
        }
    }
}
