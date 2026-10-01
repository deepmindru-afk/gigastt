# Raw telephony preparation

Raw REST uploads (`?codec=pcmu|pcma|g722&sample_rate=N`) now reject an operator
`max_audio_secs` limit before codec PCM allocation. Cancellation is checked before
and after the synchronous codec call, between resampling stages, and during
short-clip precision conversion. A running codec call or long-clip WAV encoding
is synchronous; cancellation is observed after it returns. The existing
30-minute whole-buffer safety ceiling still applies when the operator limit is
absent, disabled, or larger.

Clips with at most 480,000 output samples (30 seconds at 16 kHz) pass PCM directly
to the engine. Longer clips keep the previous PCM16 WAV representation, which
lets the existing file decoder consume bounded windows without retaining the
whole clip as floats. There is no new public setting or source type. Shared
request options retain VAD, hotwords, diarization and its outcome, overrides,
progress, partial transcripts, and cancellation. Raw audio is mono: requesting
channel splitting keeps the existing mono fallback and suppresses offline
diarization in favor of the channel-label policy.

## Precision and rate contract

Direct PCM reproduces `encode_wav_pcm16` followed by PCM16 decoding exactly:
finite values are clamped to [-1, 1], multiplied by 32767, rounded to `i16`, then
divided by 32768 as `f32`; non-finite values become zero. The CLI uses the same
helper for its existing precision step. Longer uploads encode the original
resampled floats once, avoiding a second quantization. This operation is distinct
from narrowband resampler rounding using a multiplier of 32768; if that separate
rounding is enabled, both stages remain necessary.

G.711 has one decoded source sample per byte at the declared rate. G.722 has two
samples per byte at 16 kHz, including when the caller supplies its historical
8 kHz RTP clock alias. Budget tests cover both G.722 aliases, exact limits,
one-byte excess, and the floating-point value immediately below an exact second.
An additional output-rate check covers resampling rounding.

## Why long clips keep WAV

The rejected always-PCM candidate saves the temporary WAV allocation but doubles
retained audio from two to four bytes per sample. For a 600-second clip this is
19.2 MB versus 38.4 MB. Peak preparation allocation is unchanged for G.711 because
source PCM, resampling state, and output coexist. The old path does **not** retain
an additional full decoded float buffer during non-VAD windowed inference.

The chosen threshold bounds direct PCM sample data to 1.92 MB per active request,
versus up to 0.96 MB plus a WAV header previously. Vector spare capacity and
allocator overhead are additional; the artifact reports retained capacity. This is a small-input storage
tradeoff, not a universal memory reduction. Beyond the threshold, retained WAV
payload is approximately `32,000 * duration_seconds + 44` bytes. Multiply audio
retention and temporary preparation memory by simultaneously active requests;
pool limits and `max_audio_secs` both matter. Upload bytes remain alive for
admission accounting. VAD and diarization may add their own whole-buffer storage;
these measurements exclude inference and those downstream allocations.

## Measurements

A release benchmark uses the repository's four-second telephony speech fixtures,
cycling their encoded bytes to 4, 60, and 600 seconds for preparation measurements.
Repeated G.722 bytes carry artificial decoder-state joins; this is an allocation
workload, not a speech-quality corpus. Each case runs in a fresh process, with
five uninstrumented timing passes and a separate counting-allocator pass.
Both preparations use the same current codec/resampler implementation; `legacy`
reconstructs the previous WAV materialization, so this comparison isolates the
representation change rather than measuring the small added budget checks.
Allocator counters assume no unrelated Rust allocations during that isolated
pass. RSS is the process high-water mark, not incremental retained memory.

Host: Linux x86_64, AMD Ryzen AI 9 HX 370, Rust 1.98.1, optimized release profile.
Other user inference workloads were active. A repeat pinned to logical CPU 0
reduced migration variability but does not remove contention or frequency changes.
Process CPU and wall times are reported separately in the artifact. These data
show **no reliable general preparation CPU speedup**. Model inference is excluded;
no end-to-end latency improvement is claimed.

Pinned CPU medians for the old WAV route and rejected always-PCM candidate:

| Codec | Seconds | WAV CPU ms | PCM CPU ms | Retained WAV / PCM MB |
|---|---:|---:|---:|---:|
| PCMU | 4 | 7.25 | 7.97 | 0.128 / 0.256 |
| PCMA | 4 | 8.04 | 7.23 | 0.128 / 0.256 |
| G.722 | 4 | 1.87 | 1.72 | 0.128 / 0.256 |
| PCMU | 60 | 50.49 | 55.63 | 1.92 / 3.84 |
| PCMA | 60 | 52.28 | 50.32 | 1.92 / 3.84 |
| G.722 | 60 | 29.09 | 27.67 | 1.92 / 3.84 |
| PCMU | 600 | 489.43 | 453.75 | 19.2 / 38.4 |
| PCMA | 600 | 437.40 | 475.73 | 19.2 / 38.4 |
| G.722 | 600 | 288.11 | 270.06 | 19.2 / 38.4 |

The adopted hybrid uses PCM only for the short rows and WAV for the longer rows.
A separate hybrid run measured 4-second CPU medians of 6.98/7.30/1.77 ms for
PCMU/PCMA/G.722; 600-second medians were 478.10/497.10/319.61 ms. Retention matched
the selected representation exactly. Long preparation peaks remained 59.2 MB for
G.711 and 76.8 MB for G.722; the change does not eliminate whole-codec buffering.
The baseline artifact includes cumulative allocation bytes, peak live allocation,
retained capacity, timing arrays, and Linux maximum RSS in KiB:
[raw telephony baseline](benchmarks/raw-telephony-baseline.json).

## Reproduction and parity scope

Build the opt-in harness:

```sh
cargo bench -p gigastt-core --bench raw_telephony --no-default-features \
  --features file-decode --no-run
```

Run the printed executable with `GIGASTT_RAW_BENCH_MODE=legacy|pcm|hybrid`,
`GIGASTT_RAW_BENCH_CODEC=pcmu|pcma|g722`, and
`GIGASTT_RAW_BENCH_SECONDS=4|60|600`. Run each case as a separate process for RSS.
`pcm` deliberately measures the rejected unbounded-retention alternative;
`hybrid` mirrors the selected preparation policy.

For native model parity use `GIGASTT_RAW_BENCH_MODE=parity` and
`GIGASTT_RAW_BENCH_MODEL_DIR` pointing to an installed `ml_ctc` model and
`vad/silero_vad.onnx`. The harness compares PCM bits, mel feature bits, and complete
serialized transcription results, with punctuation and ITN disabled and VAD both
off and on. It covers all three short codec fixtures and a 48-second repeated
PCMU fixture on the long WAV route. Native codec WAV fixtures are reported
separately because they do not include the raw route's additional precision step.
These matched-fixture checks establish parity for the tested inputs; they are not
an independent corpus WER evaluation.

Both clean-base and additional local narrowband-rounding validation passed all
eight raw/legacy result comparisons, including both selected storage routes.
Each also matched PCM and mel bits. The local overlay applies 32768-based rounding
after 8-to-16 kHz resampling, followed by the existing raw-route 32767-based snap;
it is not included in this change. Its exact patch hash and the clean base commit
are recorded in the artifact to distinguish these precision contexts. Native
codec WAV and raw fixtures all produced “шестьдесят тысяч тенге сколько будет
стоить” on the short clips in both contexts. The 48-second repeated fixture also
preserved the complete result within each context. Raw full word-alignment logs
remain local; the public artifact contains transcript text and canonical result
hashes alongside the harness's byte-equality outcome.
