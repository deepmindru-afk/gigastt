# Repeated live-window computation

This investigation keeps production defaults unchanged. The offline encoder
continues to recompute the retained window, with a fresh decoder state.
Test-only cadence overrides and a frontend-cache prototype are excluded from
production builds. A smaller stride is not automatically a faster first partial,
and a larger stride can change the committed transcript.

## Protocol and scope

The opt-in Linux harness drives the real engine with 100 ms chunks of 16 kHz
PCM on a real-time schedule. It uses the pinned rnnt INT8 bundle, punctuation
and ITN enabled, stable-prefix commits, four encoder threads per pool slot,
and one or two simultaneous sessions. The fixed thread setting makes these
controlled experiments distinct from the server's automatic thread selection.
No audio is dropped to catch up when inference falls behind its schedule.

The workload comprises the 15 checked-in Golos fixtures (71 normalized reference
words), plus three separately reported controls: a clip with two seconds of
leading and trailing silence, three seconds of silence, and direct concatenation
of the first three clips. This is a small exploratory workload, not a replacement
for corpus WER. Concurrent runs repeat the same inputs in both sessions.

Each configuration runs in a fresh process after model/sidecar warmup. Model
loading is outside the CPU/time interval. Stage timers measure elapsed time in
mel extraction, encoder invocation, decoder/joiner decoding and final text
postprocessing. Process CPU ticks cover all native threads across the entire
workload; they are not interchangeable with stage elapsed time. Sampled RSS is
a process-wide observation after chunks and may miss transient native peaks.
The model's mapped/shared pages also contribute to RSS.

First-partial latency starts at the first input chunk and ignores empty and
final-only events. Stop-flush time measures the call after all input is sent;
last-final time relative to audio EOF is recorded separately and can be negative
when an endpoint occurs during trailing silence. Partial stability counts words
retracted between successive nonempty partials, with final corrections reported
separately. WER uses the same normalizer as the benchmark suite and retains empty
hypotheses as errors. Silence hallucinations are reported separately because
silence has no reference words.

## Observed results

Measured on 2026-10-01, AMD Ryzen AI 9 HX 370 (24 logical CPUs), Linux,
Rust 1.98.1, debug profile, pinned ORT 2.0.0-rc.13. Source base and model hashes
are in [the raw artifacts](../benchmark/results/live-windows/provenance.json).
These are single passes on a shared host, in configuration order, without
randomization. Timing differences include host load/frequency effects and are
not causal estimates or release performance claims.

| Configuration | Sessions | Decode windows | Encoded seconds | Actual max window s | CPU s | First partial ms | Stop flush ms | Retracted words | RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1 | 112 | 261.5 | 4.1 | 33.38 | 772 | 91 | 19 | 452.7 |
| baseline | 2 | 224 | 522.9 | 4.1 | 78.22 | 758 | 110 | 38 | 815.5 |
| stride400 | 1 | 215 | 444.5 | 3.9 | 78.19 | 777 | 131 | 41 | 450.8 |
| stride400 | 2 | 430 | 889.0 | 3.9 | 148.61 | 834 | 109 | 82 | 816.7 |
| stride1200 | 1 | 76 | 203.0 | 4.8 | 40.92 | 1261 | 303 | 12 | 453.5 |
| stride1200 | 2 | 152 | 406.0 | 4.8 | 93.89 | 1351 | 369 | 24 | 808.6 |
| window5 | 1 | 112 | 287.4 | 5.6 | 67.77 | 917 | 296 | 20 | 457.2 |
| window5 | 2 | 224 | 574.8 | 5.6 | 135.68 | 969 | 331 | 40 | 808.5 |
| vad | 1 | 121 | 280.0 | 4.0 | 67.07 | 839 | 247 | 20 | 462.7 |
| vad | 2 | 242 | 560.1 | 4.0 | 136.99 | 835 | 246 | 40 | 828.7 |

Input totals 82.315 seconds per session. CPU, windows and encoded coverage sum
both sessions in concurrent runs; latency columns are medians over clips. The
2.5-second setting is a soft slide trigger, not a maximum retained window.
Baseline coverage is 3.18 times input duration, but this does not imply a 3.18x
available speedup. All measured window starts were hop aligned.

| Configuration | Fixture errors / 71 words | Concatenation errors / 15 words | Padded / pure silence errors |
|---|---:|---:|---:|
| baseline | 1 | 0 | 0 / 0 |
| stride400 | 1 | 1 | 0 / 0 |
| stride1200 | 1 | 1 | 0 / 0 |
| window5 | 0 | 0 | 0 / 0 |
| vad | 2 | 0 | 0 / 0 |

Both workers reproduce the same quality outcomes. The stride variants lose
“заказать” in the concatenated control. Short-clip WER alone conceals this
regression. The wider window corrects one word in a tiny sample; it increases
retained coverage and does not establish a general WER improvement. VAD changes
endpoint placement and adds windows here; current VAD is an endpointer, not an
encoder silence gate.

Baseline single-session stage elapsed seconds: mel **0.848**, encoder
**11.281**, decoder/joiner **0.454**, final postprocessing **0.012**. Mel is 6.7%
of these measured stages, while the encoder dominates. Other pipeline work,
VAD, scheduling and waits are outside these stage totals. Full per-configuration
stages, transcript events and final timing are retained in the JSON artifacts.

A separate single-session baseline samples `/proc/self/task/*/schedstat`
before and after each stage. Summed **process-thread CPU during stage** was
1.355 s for mel, 58.611 s for encoder, 1.305 s for decoder/joiner and 0.031 s
for postprocessing (61.50 s aggregate process CPU). This includes background
native workers and sampling overhead; thread snapshots are not atomic and do
not provide exclusive native-stack attribution. Concurrent sampling is rejected
because overlapping stages would double-count process CPU. The supplemental
run reproduced baseline transcripts/windows but had substantially different
elapsed times on the shared host (a build also ran), reinforcing that timing
ratios across these runs must not be interpreted as controlled speedups.

## Frontend reuse and ownership

Mel extraction has a 320-sample frame and 160-sample hop, without centered
padding or window-wide normalization. Complete frames can be reused only when
both windows refer to unchanged samples on the same absolute frame grid.
The research prototype invalidates on backward reanchoring, a changed grid
phase, or a sub-frame window. It is scoped to one immutable audio timeline;
a new stream requires a new cache.

Synthetic tests cover growth, overlap, backwards reanchoring, one-sample phase
changes and short-input transitions. Replaying the 15 baseline speech clips
produced bit-identical features, reusing 13,014 of 19,173 complete frames with no
observed phase invalidations. The debug replay took 530 ms for fresh extraction
versus 173 ms for the prototype, including its output construction. This is a
frontend-only result. It neither reuses context-dependent encoder activations
nor preserves LSTM state across overlapping windows.

A production cache would still need an explicit reset/ownership contract,
bounded retained storage, frame-grid invalidation and measurements after other
frontend optimizations. Its benefit is limited by the measured frontend share;
it cannot turn repeated coverage of audio into the same factor of total speedup.

## Bounded follow-up

Keep the 800 ms cadence and current window/VAD defaults. The 400 ms candidate
nearly doubles decode calls, adds revisions and does not improve the observed
median first partial. The 1200 ms candidate reduces coverage but delays partials
and loses a word in the continuous control. Rebenchmark candidates in randomized
release runs on an idle host before drawing CPU or latency conclusions.

The narrow safe implementation candidate is a per-stream mel-frame cache with
explicit absolute sample origin, invalidation on grid changes/reset, and storage
bounded by the retained window. Require bitwise frontend equivalence, complete
transcript/confidence/endpoint equivalence, bounded memory and a meaningful
release end-to-end improvement after existing frontend work lands. The current
prototype copies the assembled output and cache; it demonstrates reuse validity,
not an allocation-optimal implementation. Its measured frontend saving alone
would remove only a small fraction of baseline total stage time.

Silence gating needs onset/tail retention and endpoint tests before it can skip
encoder work. Reusing offline Conformer activations or decoder state across
windows has no equivalence guarantee: growing context can change old outputs.
A truly streaming model/backend is a separate quality and compatibility project.
Neither is implemented by this research change.

## Reproduction

Install both the rnnt INT8 bundle and the punctuation/VAD sidecars first. The
research harness never downloads models. Run from the repository root:

```sh
python3 benchmark/live_windows.py --output /absolute/local/results
python3 benchmark/live_windows.py --output /absolute/local/results --summarize-only
GIGASTT_LIVE_REPLAY=/absolute/local/results/baseline-c1.json \
  cargo test -p gigastt-core --lib benchmark_frontend_reuse_on_recorded_windows -- --ignored --nocapture
GIGASTT_LIVE_STAGE_CPU=1 python3 benchmark/live_windows.py \
  --output /absolute/local/cpu-results --cases baseline --concurrency 1
PYTHONPATH=benchmark python3 -m pytest benchmark/tests/test_live_windows.py -q
```

The default sweep takes roughly fifteen minutes because audio is paced in real
time. `--cases` and `--concurrency` select a smaller sweep. The ignored engine
experiment explicitly skips unless `GIGASTT_LIVE_PROBE` names an output JSON;
the normal model-coverage run therefore does not start the long sweep.
