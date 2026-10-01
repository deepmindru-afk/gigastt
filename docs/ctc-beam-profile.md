# CTC hotword beam allocation profile

The eight-hypothesis beam remains unchanged. Prefix copying creates substantial
allocation traffic, but this measurement does not justify a persistent prefix
representation: decoding used at most about 2.3% of feature-plus-encoder time on the
captured speech. Even eliminating it entirely would save only that fraction in
this workload. The numbers below are a profiling baseline, not a speedup claim.

## Reproduce

Install the `ml_ctc` INT8 model and vocabulary, then run:

```sh
GIGASTT_CTC_BENCH_OUTPUT="$HOME/.cache/gigastt-ctc-profile" \
  cargo bench -p gigastt-core --no-default-features \
  --features __internals,file-decode --bench ctc_beam
```

`GIGASTT_CTC_BENCH_MODEL_DIR` overrides `~/.gigastt/models`. Without an output
directory the harness skips capture; it never downloads models. The harness
writes provenance, raw frame-major little-endian float logits, capture timing,
and decoder results including exact token IDs, frame indices and confidence
bits. The profiling interface is behind the private `__internals` feature.

The short input is `golos_00.wav` (4 seconds). The long input concatenates
`golos_00.wav` onward in numeric order and truncates at 384,000 mono 16 kHz
samples (24 seconds, one normal ORT file window). This is recorded speech,
but the joins are artificial; it is not a continuous long-form quality corpus.
The harness records SHA-256 for every source fixture, encoder and vocabulary.
Two glossaries use fixture words and unrelated Russian/English technical words;
1/8/32/64 phrases include single words and two-word continuations. Boost is 5,
beam width 8, acoustic top-k 6, vocabulary 71. Zero phrases exercises greedy
CTC. Uniform zero logits over 600 frames separately stress deterministic ties;
they are deliberately not representative speech.

Capture uses CPU ORT with four encoder threads, one warm-up and three measured
feature/encoder passes. Each decoder case has a separate frame-observer pass,
a counting-allocator pass and five timed production-path passes. All passes
must return identical token/frame/confidence bits. The allocator is disabled
for timing (its disabled flag check remains); counters cover Rust allocations,
not native ORT allocations, RSS or allocator metadata. The counting interval
is isolated decoder work with no other Rust allocating threads; these global
counters would include unrelated Rust work if such threads were added. Peak live bytes include
the returned token vector. Timings exclude model/session load, audio decoding,
glossary compilation and word formatting, and are not whole-request latency.

## Observed baseline

Measured on 2026-10-01, Linux x86_64, AMD Ryzen AI 9 HX 370, Rust 1.98.1,
release/LTO build. The host concurrently ran other inference/build work;
time ranges must not be treated as a controlled hardware comparison.
`ort` was 2.0.0-rc.13; native ORT identified itself as 1.28.0, commit da9b5e3.

- Encoder SHA-256: `e08e27ae5669b39f0c378fae101bbbb9a80505f74f9b66719c309bf5b894a480`.
- Vocabulary SHA-256: `4d130287892e1099fedfb3f93c4b4cf8a263151158801680b28977d1be4133f4`.
- Short logits SHA-256: `814903fe4ab2ec1b006308b20b5046943675cec9f56fc965327d00c9395e5f58`.
- Long logits SHA-256: `270c1c745ccee928e744fb061a1eb6c6184f0de3fa47bec4ba9d3275308bc65a`.

Short feature extraction took 0.30–0.70 ms, encoder 302–392 ms; long feature
extraction 1.87–4.73 ms, encoder 1,632–1,786 ms. The table reports median
production decoder time, cumulative allocated bytes and peak live bytes
(decimal MB), independently of these encoder timings.

| Input / glossary | Phrases | Decoder ms | Allocations | Allocated MB | Peak MB | Mean/max candidates | Max live prefixes |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4 s / fixture | 0 | 0.04 | 5 | 0.003 | 0.002 | — | — |
| 4 s / fixture | 1 | 2.10 | 10,885 | 5.06 | 0.116 | 7.02 / 9 | 68 |
| 4 s / fixture | 8 | 2.59 | 13,029 | 6.52 | 0.172 | 8.35 / 15 | 114 |
| 4 s / fixture | 32 | 3.76 | 14,525 | 6.91 | 0.220 | 9.30 / 19 | 148 |
| 4 s / fixture | 64 | 3.78 | 16,201 | 8.31 | 0.255 | 10.33 / 19 | 148 |
| 4 s / unrelated | 64 | 6.00 | 21,102 | 11.29 | 0.351 | 13.39 / 26 | 201 |
| 24 s / fixture | 0 | 0.23 | 7 | 0.012 | 0.006 | — | — |
| 24 s / fixture | 1 | 16.32 | 64,862 | 127.15 | 0.470 | 6.77 / 8 | 64 |
| 24 s / fixture | 8 | 20.35 | 75,068 | 146.20 | 0.726 | 7.85 / 14 | 109 |
| 24 s / fixture | 32 | 20.15 | 77,814 | 149.81 | 1.052 | 8.13 / 20 | 154 |
| 24 s / fixture | 64 | 21.38 | 80,483 | 154.13 | 1.039 | 8.41 / 20 | 152 |
| 24 s / unrelated | 64 | 37.12 | 96,180 | 184.52 | 1.333 | 10.02 / 26 | 206 |
| 600 uniform frames / fixture | 64 | 95.58 | 149,389 | 446.21 | 2.253 | 15.82 / 24 | 185 |
| 600 uniform frames / unrelated | 64 | 168.26 | 156,078 | 476.70 | 2.658 | 16.55 / 28 | 216 |

Phrase count alone does not determine cost: unrelated 32 and 64 phrase cases
have identical allocation/candidate counts on captured speech because their
additional continuations are not reached. A [compact baseline artifact](benchmarks/ctc-beam-baseline.json)
contains provenance, both capture timings, all 30 cases and individual decoder
timing samples supporting this table. Full text/alignments and raw logits are
local output only; reproduce them with the command above. Repeating capture on the same host produced
identical logits hashes, while timings varied with contention.

## Memory bound and decision

With B=8 retained hypotheses, V=71 vocabulary classes and T input frames,
each frame has at most V deduplicated candidates. Each old prefix contributes
at most one unchanged prefix plus V-1 extensions, so the unpruned next set
has at most B×V = 568 prefixes. Each prefix/alignment has at most T tokens.
Old beams, next beams and temporary extension buffers therefore require
O(B×V×T) live storage; cumulative copied bytes can grow quadratically in T.
On this 64-bit target, a label plus alignment token costs 32 bytes. At T=600,
the payload upper bound for 568 next prefixes alone is about 10.9 MB; old
beams, temporary replacements, vector bookkeeping and candidate scratch add
to that. This is a structural bound, not the observed peak or an RSS promise.
A single decode releases all this storage before the next file window. Whole
transcript/input memory and encoder/native allocator memory are separate.

The 24-second speech case reached 215 tokens per prefix, versus up to 429
in the uniform-tie cases. Observed maximum live Rust storage across the tested
cases was 2.66 MB despite hundreds of MB of cumulative allocation traffic.
The production beam, ordering, bias refunds, blank/repeat behavior and abort
checks are unchanged. Regression coverage verifies the observer preserves
exact alignments/confidence bits at every abort boundary, including ties;
existing hotword refund and repeat tests continue to cover decoder semantics.

Revisit allocation avoidance if a controlled target-device or concurrent-request
profile shows this decoder materially affects latency or allocator contention.
A small change that avoids allocating an extension before finding an existing
prefix should be measured before introducing shared persistent prefix nodes.
This baseline alone does not establish a meaningful end-to-end benefit.
