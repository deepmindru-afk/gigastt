# File transcription source contract

`Engine::transcribe_request` owns request routing. The caller owns the checked
out session for the entire call. Server blocking workers own their reservation
until they actually exit; returning a timeout response does not itself stop a
native call or return that session.

Only positive finite `max_audio_secs` values enable an operator duration limit.
Zero, negative, NaN and infinity follow the existing unlimited convention.
Limits are inclusive at the sample boundary; one extra sample is rejected with
`AudioTooLong`. For split inputs, duration means the longest individual channel,
not the sum of channel lengths. Progress measures cumulative recognition work,
so a two-channel file normally reports twice its elapsed-audio sample count.

| Source | Duration enforcement | PCM working storage | Recognition controls |
| --- | --- | --- | --- |
| `Path` | Source-rate decode budget | Windowed normally; whole buffer for offline diarization | Overrides, hotwords, abort, progress, partial, optional diarization |
| `Bytes` | Same as `Path` | Same PCM behavior; encoded bytes remain retained | Same as `Path` |
| `Samples` | Validate borrowed 16 kHz slice before inference | Caller already owns full mono PCM; recognition uses windows for long input | Same mono controls |
| `Channels` | Validate longest borrowed 16 kHz channel before any channel starts | Caller owns all channel buffers | Same controls, cumulative per-channel progress; channel index supplies speaker label, offline diarization ignored |
| `ChannelStreams` | Source-rate budget while decoding each selected channel | Windowed channel decode, retaining encoded bytes | Same split controls; channel index supplies speaker label, offline diarization ignored |
| Raw telephony adaptation | Codec decoder budget, then request budget | Existing bounded full decode and WAV adaptation | Normal shared server request controls after adaptation |

Encoded whole-buffer decoders have a separate 1,800-second allocation safety
ceiling; an operator limit can lower it. This protects allocation during
container/raw-codec expansion and is not imposed on PCM already supplied by
the caller. Offline diarization also has its own backend duration limit,
reported through `diarization_outcome`. That outcome is separate from the
operator input-duration error.

Mono file VAD can run through lazy speech windows. Split VAD uses decoded
channels. Ordinary channel scanning avoids full PCM materialization, but the
Opus scan uses the existing bounded whole-channel decoder. Source choice thus
matters for allocation ceilings even when recognition controls are equivalent.
Windowed PCM does not make total request memory constant: encoded uploads,
accumulated words and caller-requested owned snapshots still grow with data.

A cancellation flag already set at entry takes precedence over budget
validation and leaves existing partial/progress sinks untouched. During
recognition it is cooperative at decode boundaries; an active native runtime
call must return. Preprocessing, speaker loading and postprocessing have
additional synchronous boundaries: do not infer immediate worker termination
from a response timeout. Progress is updated after recognition windows, not
for every byte decoded, model load or postprocessing operation. Partial text
is provisional and remains readable after supported cancellation paths.

The server's common file request builder preserves controls when deciding
mono fallback versus split routing. Core callers choose the source explicitly;
`Channels` does not run dual-mono classification for them. This contract does
not change sample values, quantization, channel ordering or mixing rules.

`Samples` and `Channels` remain available without `file-decode`. Encoded
variants require that feature. Cancellation, duration validation and explicit
request controls remain available in lean builds. No new runtime dependency
is required.

Model-free regressions compare exact-limit and one-sample-over WAV path,
bytes, samples, channels and channel-stream requests; they also exercise
invalid-limit conventions and cancellation precedence. Existing routing and
channel-progress tests cover preserved overrides and cumulative work.
