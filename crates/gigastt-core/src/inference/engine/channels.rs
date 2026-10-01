//! Stereo / `channels=split` transcription.

use super::*;

impl Engine {
    /// Transcribe a multi-channel recording with one speaker label per channel.
    ///
    /// Runs the engine once per channel sequentially on the supplied triplet. The
    /// caller is responsible for deciding whether to use this mode (e.g. after
    /// checking for dual-mono). Channel 0 becomes `speaker_0`, channel 1
    /// `speaker_1`, and so on. Results are merged into a single chronologically
    /// ordered transcript.
    ///
    /// Per-channel `text` fields are ignored: the merged transcript's text is
    /// rebuilt from the merged words after the final ITN/punctuation pass.
    ///
    /// Thin wrapper over [`Engine::transcribe_request`] with default overrides
    /// and no hotwords.
    #[cfg(feature = "file-decode")]
    pub fn transcribe_channels(
        &self,
        channels: &[Vec<f32>],
        triplet: &mut SessionTriplet,
    ) -> Result<TranscribeResult, GigasttError> {
        self.transcribe_request(
            TranscribeRequest::new(TranscribeSource::Channels(channels)),
            triplet,
        )
    }

    /// Per-channel decode + merge used by [`TranscribeSource::Channels`].
    pub(crate) fn transcribe_channels_inner(
        &self,
        channels: &[Vec<f32>],
        triplet: &mut SessionTriplet,
        overrides: &TranscribeOverrides,
        hotwords: Option<&HotwordOverride>,
        ctl: DecodeControls,
    ) -> Result<TranscribeResult, GigasttError> {
        if channels.is_empty() {
            return Ok(TranscribeResult {
                text: String::new(),
                words: Vec::new(),
                duration_s: 0.0,
                confidence: None,
            });
        }

        let mut per_channel = Vec::with_capacity(channels.len());
        let mut completed_samples = 0u64;
        for (channel, channel_samples) in channels.iter().enumerate() {
            let publish = |update: WordUpdate| {
                ctl.publish_update(WordUpdate {
                    channel: Some(channel),
                    ..update
                })
            };
            let report = |n| ctl.report(completed_samples.saturating_add(n));
            let channel_ctl = DecodeControls {
                on_progress: ctl.on_progress.map(|_| &report as &dyn Fn(u64)),
                on_partial: ctl.on_partial.map(|_| &publish as &dyn Fn(WordUpdate)),
                ..ctl
            };
            let words = self.decode_words_for_samples(
                channel_samples,
                triplet,
                overrides,
                hotwords,
                channel_ctl,
            )?;
            channel_ctl.check_abort()?;
            completed_samples = completed_samples.saturating_add(channel_samples.len() as u64);
            ctl.report(completed_samples);
            let duration_s = channel_samples.len() as f64 / 16000.0;
            per_channel.push(TranscribeResult {
                confidence: aggregate_confidence(&words),
                text: String::new(),
                words,
                duration_s,
            });
        }

        let merged = merge_channel_results(per_channel);
        Ok(self.finish_transcribe_result(merged.words, merged.duration_s, overrides))
    }

    /// Streaming twin of the `channels=split` decode.
    ///
    /// Each channel of `data` runs through the windowed loop the mono file path
    /// uses, so
    /// peak audio memory is one window instead of every channel of the whole
    /// file — which is what kept channel splitting on a duration ceiling. The
    /// per-channel results are merged exactly as the whole-buffer twin merges
    /// them.
    ///
    /// The caller decides *whether* to split; use
    /// [`scan_channels`](crate::inference::audio::scan_channels), which answers
    /// that in one pass without materializing anything.
    ///
    /// Progress sums decoded samples across channels. Each channel's local clock
    /// is offset by the completed channels, so the watchdog never sees a reset.
    // Bundling these behind `&TranscribeRequest` is what one would want, but the
    // caller's match moves `data` out of `req.source`, so the request cannot be
    // borrowed whole afterwards. Same shape as `run_inference` above.
    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "file-decode")]
    pub(crate) fn transcribe_channel_streams(
        &self,
        data: bytes::Bytes,
        channels: usize,
        max_audio_secs: Option<f64>,
        triplet: &mut SessionTriplet,
        overrides: &TranscribeOverrides,
        hotwords: Option<&HotwordOverride>,
        ctl: DecodeControls,
    ) -> Result<TranscribeResult, GigasttError> {
        let request_biaser = hotwords.and_then(|hw| self.build_request_biaser(hw));
        let biaser = self.select_biaser(hotwords, &request_biaser);

        let spec = window_spec(self.ane_encoder, self.variant.is_ctc());
        let mut per_channel = Vec::with_capacity(channels);
        let mut completed_samples = 0u64;
        for k in 0..channels {
            let publish = |update: WordUpdate| {
                ctl.publish_update(WordUpdate {
                    channel: Some(k),
                    ..update
                })
            };
            let report = |n| ctl.report(completed_samples.saturating_add(n));
            let channel_ctl = DecodeControls {
                on_progress: ctl.on_progress.map(|_| &report as &dyn Fn(u64)),
                on_partial: ctl.on_partial.map(|_| &publish as &dyn Fn(WordUpdate)),
                ..ctl
            };
            let mut windows =
                audio::FileWindows::from_bytes_channel(data.clone(), spec, max_audio_secs, k)
                    .map_err(audio::decode_error)?;
            let words = self.decode_words_streaming(&mut windows, triplet, biaser, channel_ctl)?;
            channel_ctl.check_abort()?;
            completed_samples =
                completed_samples.saturating_add(windows.total_16k_samples() as u64);
            ctl.report(completed_samples);
            per_channel.push(TranscribeResult {
                confidence: aggregate_confidence(&words),
                text: String::new(),
                words,
                duration_s: windows.total_16k_samples() as f64 / 16000.0,
            });
        }

        let merged = merge_channel_results(per_channel);
        Ok(self.finish_transcribe_result(merged.words, merged.duration_s, overrides))
    }
}
