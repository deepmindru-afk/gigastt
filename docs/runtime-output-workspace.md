# RNN-T runtime output workspace

`RuntimeSession::run_f32_into` lets a caller retain flat f32 output buffers.
The default implementation uses `run`, so existing runtimes remain compatible.
ORT validates all output types/counts, then copies borrowed native outputs into
those buffers while holding the session lock. No runtime-owned borrow escapes.
Decoder state is still committed only for a non-blank token, and cancellation
still happens between runtime calls. Encoder outputs continue to use `run`.

The workspace belongs to one greedy decode invocation, including its blank-run
cache. It is recreated for the next invocation; it is not a process-wide cache.
ORT still allocates input wrappers, shape vectors and native inference storage.
This change removes intermediate owned output tensors and one copy of each
f32 output. It does not make inference allocation-free.

## Reproduction

The small identity-model allocation regression runs in normal CI without model
files. Installed-model probes require both INT8 heads in `~/.gigastt/models`:

```sh
cargo test -p gigastt-core --test runtime_output_workspace
cargo test -p gigastt-core --test runtime_output_workspace -- --ignored --nocapture
cargo test -p gigastt-core --lib test_installed_runtime_workspace_equivalence -- --ignored --nocapture
```

The last test explicitly skips heads whose INT8 encoder is absent (main CI
may provision only rnnt). Install both heads for the full local comparison.
It compares full transcription results on `golos_00.wav` and
`golos_01.wav`, then verifies token IDs, frame indices, confidence bits, LSTM
state, endpoint decisions and cancellation after three checks. A legacy adapter
forces the old owned-output route using the same models and runtime settings.

## Local measurements

Measured on 2026-10-01, AMD Ryzen AI 9 HX 370, Linux x86_64, Rust 1.98.1,
ORT crate 2.0.0-rc.13, debug Rust build, CPU provider, four encoder threads and
one decoder/joiner thread. Allocation counts cover Rust's allocator on the
calling thread, **not ORT's native allocator**. The warmed stage probe uses the
four-second `golos_00` features, 5 encoder calls and 200 decoder/joiner calls.
Copy bytes count the explicit output materialization and workspace copies;
internal ORT copies and unchanged input/state copies are excluded.

| Stage | Rust allocations/call, old → new | Allocated bytes/call, old → new | Explicit output copy bytes/call, old → new |
|---|---:|---:|---:|
| Encoder, either head (control) | 14 → 14 | 308,532 → 308,532 | 307,204 → 307,204 |
| Decoder, either head | 20 → 13 | 5,712 → 1,576 | 7,680 → 3,840 |
| Joiner, rnnt | 10 → 7 | 1,240 → 848 | 272 → 136 |
| Joiner, e2e_rnnt | 10 → 7 | 5,204 → 848 | 8,200 → 4,100 |

On `golos_00`, rnnt emits 43 tokens with 44 decoder and 143 joiner calls;
e2e_rnnt emits 14 tokens with 15 decoder and 114 joiner calls. Combining these
observed call counts with the measured allocations above gives 53.72 → 36.58
Rust allocations per emitted rnnt token and 102.86 → 70.93 per e2e_rnnt token
for decoder/joiner runtime calls only (encoder calls and other application
allocations are excluded). Blank frames account for the extra joiner calls.

The full-file timing probe warms each engine once, then records six runs per
mode while alternating old/new order. The table gives medians; stage medians
are independent and need not sum to the median total. Throughput is audio
seconds per wall second for serial file transcription, without HTTP overhead.

| Head / fixture | Total ms, old → new | Encoder ms, old → new | Decoder ms, old → new | Joiner ms, old → new | Throughput, old → new |
|---|---:|---:|---:|---:|---:|
| rnnt / golos_00 | 214.92 → 209.56 | 193.70 → 187.88 | 2.62 → 2.49 | 4.06 → 4.03 | 18.61 → 19.09 |
| rnnt / golos_01 | 211.87 → 216.38 | 192.91 → 194.20 | 2.79 → 2.71 | 3.81 → 3.79 | 16.19 → 15.87 |
| e2e_rnnt / golos_00 | 261.15 → 261.53 | 230.26 → 233.29 | 1.93 → 2.38 | 6.36 → 5.56 | 15.32 → 15.30 |
| e2e_rnnt / golos_01 | 223.19 → 217.24 | 191.25 → 186.11 | 2.65 → 2.00 | 7.23 → 6.25 | 15.37 → 15.79 |

These short debug runs shared the host with other builds. Old/new ranges
overlap in every case (for example rnnt/golos_00: 196.72–223.84 ms versus
198.92–243.74 ms). They establish no end-to-end speedup or regression. The
supported conclusion is reduced Rust allocations and redundant output copying;
the much larger encoder cost is unchanged by this implementation.
