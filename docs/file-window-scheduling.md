# File-window thread launch and pool fairness

Keep the scoped worker threads and current scheduling policy. Measured launch
cost is tiny beside the native encoder. Fairness is a separate tradeoff:
opt-in file parallelism can retain every shared slot until the last full wave,
so an arriving live request can wait for most of a long file. The default
`--file-window-concurrency 1` avoids borrowing extra slots; a dedicated
`--batch-pool-size` separates file work from interactive pool admission.
Neither setting guarantees latency when the host CPU is overloaded.

## Reproduce

The opt-in `file_scheduling` benchmark has three modes. No mode runs unless
selected. `launch` and `mixed` need no model. For example, on Linux:

```sh
GIGASTT_SCHEDULING_BENCH_MODE=launch taskset -c 0-3 \
  cargo bench -p gigastt-core --no-default-features \
  --features __internals,file-decode --bench file_scheduling > launch.jsonl
GIGASTT_SCHEDULING_BENCH_MODE=mixed taskset -c 0-3 \
  cargo bench -p gigastt-core --no-default-features \
  --features __internals,file-decode --bench file_scheduling > mixed.jsonl
GIGASTT_SCHEDULING_BENCH_MODE=native \
GIGASTT_SCHEDULING_MODEL_DIR="$HOME/.gigastt/models" taskset -c 0-3 \
  cargo bench -p gigastt-core --no-default-features \
  --features __internals,file-decode --bench file_scheduling > native.jsonl
```

Choose CPU IDs available to your process. This baseline used `0`, `0-1` and
`0-3` for launch/mixed modes; native used `0-1` and `0-3`. Affinity restricts
where threads run; it does not reserve those CPUs from other processes.
The [compact baseline artifact](benchmarks/file-scheduling-baseline.json)
contains all launch trials, every mixed-case aggregate and native Run timing,
plus model/fixture hashes and settings. Individual live samples are emitted
locally as JSON lines. No raw speech/model data is added by this benchmark.

Measured 2026-10-01 on Linux x86_64, Ryzen AI 9 HX 370, Rust 1.98.1,
release/LTO. Other inference and builds ran concurrently. The figures describe
this contended run, not controlled throughput or a production SLA.

## Thread launch versus native model work

`launch` mirrors the production scoped spawn/join shape, including a worker
for the primary slot, input-order joins and temporary handle/output vectors.
Worker bodies are no-ops; the one-slot case runs inline, as production does.
It runs 21 trials for 1/4/16/64 waves with 1/2/4 available slots. This isolates
thread creation/join machinery; it does not measure features, model kernels,
stitching, PCM copying or actual inference result allocation.

The table divides each 64-wave trial by 64. Its p95 is a percentile of
**trial averages**, not a bound on individual thread startup latency.

| CPU affinity budget | Workers per wave | Median µs/wave | p95 µs/wave |
|---:|---:|---:|---:|
| 1 | 2 | 25.9 | 44.0 |
| 1 | 4 | 53.8 | 71.1 |
| 2 | 2 | 57.9 | 389.1 |
| 2 | 4 | 88.1 | 326.3 |
| 4 | 2 | 38.9 | 56.2 |
| 4 | 4 | 79.1 | 127.9 |

`native` runs the real INT8 multilingual CTC encoder on `golos_00.wav`
repeated to 24 seconds. It prepares features separately, loads two sessions,
warms each, then runs 2/8 windows with cap 1/2. Per-encoder intra-op threads
are `max(1, affinity_budget / 2)`. Native ORT identified itself as 1.28.0,
commit da9b5e3 (`ort` crate 2.0.0-rc.13). Session Run measurements include its
tensor handling; feature extraction, session loading and CTC token decoding
are outside those timings. This is an encoder scheduling comparison, not a
complete Engine/file/REST benchmark.

| CPU budget | Intra-op threads | Cap | Windows | Encoder-wave wall ms |
|---:|---:|---:|---:|---:|
| 2 | 1 | 1 | 2 | 5,709 |
| 2 | 1 | 2 | 2 | 3,427 |
| 2 | 1 | 1 | 8 | 27,421 |
| 2 | 1 | 2 | 8 | 21,935 |
| 4 | 2 | 1 | 2 | 7,190 |
| 4 | 2 | 2 | 2 | 4,126 |
| 4 | 2 | 1 | 8 | 18,294 |
| 4 | 2 | 2 | 8 | 12,338 |

Individual native Runs ranged from 1,767 to 8,827 ms. The substantial timing
variation and sequential case order prevent attributing exact speedups to
parallelism. The measured tens of microseconds for scoped launches do not
justify a permanent worker pool beside these second-scale Runs.

## Actual Engine mixed live/file scheduling

`mixed` uses the real Engine window loop, feature extraction, session pools
and `process_chunk` streaming path with a mock RNN-T runtime. The mock sleeps
20 ms per long-window encoder Run and 2 ms per live Run. These service times
are synthetic; they expose slot ownership independently of model weights.
Each case uses 2/8/16 file windows, eight arriving live callers, and five
repetitions. The first batch encoder is gated until live callers have queued
or entered inference. This deliberate arrival pattern tests saturation;
it is not a random-arrival traffic model. Gate setup and host scheduling can
contribute to observed times. The harness checks exact window/live Run counts
and that every pool slot is returned after the case.

The following four-CPU-affinity results aggregate 40 live requests per row.
Checkout p95 uses the nearest-rank percentile; completion includes one live
`process_chunk` call. Samples are too few to establish a production tail SLA.
“Early” means the slot was acquired before the final batch encoder completed,
not merely before the batch function returned. No HTTP/WebSocket networking,
async executor scheduling, timeout or reconnect behavior is measured.

| Total / dedicated batch slots | File cap | Windows | Median batch ms | Live checkout p50 / p95 / max ms | Live completion p95 ms | Early / 40 |
|---|---:|---:|---:|---|---:|---:|
| 2 / 0 | 1 | 16 | 415.1 | 12.7 / 28.5 / 33.6 | 30.9 | 40 |
| 2 / 0 | 2 | 2 | 30.8 | 28.1 / 46.3 / 50.4 | 50.1 | 0 |
| 2 / 0 | 2 | 8 | 117.5 | 118.9 / 134.4 / 137.0 | 139.4 | 0 |
| 2 / 0 | 2 | 16 | 248.0 | 243.9 / 278.1 / 283.2 | 283.0 | 0 |
| 4 / 0 | 2 | 16 | 256.5 | 3.4 / 12.9 / 16.7 | 15.2 | 40 |
| 4 / 0 | 4 | 2 | 26.3 | 2.6 / 8.0 / 10.9 | 11.6 | 40 |
| 4 / 0 | 4 | 16 | 120.6 | 113.5 / 137.4 / 138.1 | 142.7 | 0 |
| 4 / 3 | 3 | 16 | 165.6 | 7.6 / 18.5 / 20.6 | 21.3 | 40 |

A file that acquired all shared slots blocked every measured live caller
through its last full wave. Waiting grew with window count. This is bounded
by the finite file's remaining decode in these tests; it is not evidence of
indefinite starvation. Arbitrarily long files still have no independent
per-waiter latency bound, and configured checkout timeouts may expire first.

With four slots but only two file windows, idle extra slots were released
before the wave and all live callers entered early. Leaving shared headroom
or separating the batch pool also allowed early acquisition. Dedicated pools
isolate admission, while native CPU contention remains shared.

## Retained policy and alternatives

Extra file slots are acquired opportunistically once and retained across full
waves; unused final-wave extras are returned promptly. Keep that opt-in
throughput tradeoff and the default serial file cap. Scoped workers preserve
borrowed session ownership and finish all joins before returning an inference
error; this profiling change does not modify stitch order, panic conversion,
cancellation checks or slot release behavior.

A permanent worker pool would retain OS thread stacks and require job queues,
shutdown/join ownership and panic recovery machinery. It would not itself
solve the retained-slot fairness issue. Per-wave slot surrender would be a
separate admission policy, with throughput and reacquisition tradeoffs; it is
not justified here as an implicit behavior change. Concurrent PCM windows
already cost roughly `slots × 24 × 16000 × 4` bytes (1.536 MB per copied window)
before feature tensors, sessions, native arenas and thread stacks. Permanent
workers do not remove those buffers. Retain the simpler current implementation
unless a controlled target-device profile establishes meaningful launch cost
or a product requirement calls for a stronger shared-pool wait guarantee.
