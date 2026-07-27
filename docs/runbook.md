# Runbook

Operator-facing guidance for gigastt in production: graceful shutdown, session caps, pool exhaustion / backpressure, inference timeouts, model-download failures, and out-of-memory — with the knobs and escape hatches for each.

## At a glance

| Symptom | First check | Escape hatch |
|---|---|---|
| Clients lose `Final` on deploy | Drain window too short: check `shutdown_drain_secs` vs your orchestrator's grace period | Increase `GIGASTT_SHUTDOWN_DRAIN_SECS` OR disable WS tracking via `--shutdown-drain-secs 0` (clamped to 1 s) |
| Clients receive spurious `max_session_duration_exceeded` | Legitimate long sessions | Raise `GIGASTT_MAX_SESSION_SECS` (default 3600) or set `0` to disable |
| SIGTERM takes 30+ seconds to exit | In-flight spawn_blocking inferences can't be cancelled mid-chunk | Wait or lower `GIGASTT_SHUTDOWN_DRAIN_SECS`; process will still finish the current chunk |
| `Close(1008 Policy Violation)` unexpected | session-duration cap fired | Double check `max_session_secs` is set high enough for your use case |
| `Close(1001 Going Away)` seen by clients | Expected on SIGTERM — not a bug | None — clients should reconnect |
| REST `503` `timeout` / WS error `timeout` (`retry_after_ms`) | Pool saturated — every triplet busy | Raise `--pool-size`; isolate batch with `--batch-pool-size`; see [Pool exhaustion](#pool-exhaustion--backpressure) |
| `inference_timeout` (REST `504` / WS close) | A single run exceeded `--inference-timeout-secs` (default 600 s) | Default covers 10-min files; raise only for longer audio or investigate a wedged ONNX run |
| Server won't start, model errors | Missing / corrupt model files | See [Model download failures](#model-download-failures) |
| OOM / pod killed | Pool RSS exceeds the box | Lower `--pool-size`, use the INT8 encoder, `--pool-min-size` to boot degraded — see [Out-of-memory](#out-of-memory-oom) |

## Graceful drain (SIGTERM)

When the server receives `SIGTERM` (or the `run_with_shutdown` oneshot fires):

1. A process-wide `CancellationToken` is cancelled.
2. Every live `handle_ws_inner` session sees `cancel.cancelled()` in its `biased;` select loop, flushes its streaming state, emits a (possibly empty) `Final`, and closes with `Close(1001 Going Away)`.
3. SSE `/v1/transcribe/stream` tasks check the token between chunks and drop the channel sender, which terminates the SSE stream from the client's perspective.
4. After `axum::serve` returns, the main task waits up to `shutdown_drain_secs` seconds for the `TaskTracker` to report all tracked WS / SSE futures complete.
5. If the drain window expires with tracked tasks still running, a WARN is emitted (`Drain window expired with tracked tasks still running`) and the process exits anyway.

### Rollback: disable graceful drain

If v0.9.0 rollout breaks WS clients, the runtime supports a tiered rollback:

1. **Shrink the drain window to 1 s** (effectively disabling the wait):
   ```sh
   gigastt serve --shutdown-drain-secs 0
   # or: GIGASTT_SHUTDOWN_DRAIN_SECS=0 gigastt serve
   ```
   Note: `0` is internally clamped to `1` second. The cancel + Final path still fires, but the process won't wait longer than 1 s before exiting.

2. **Disable the session cap independently** (see the section below).

3. **Git revert** — v0.9.0's WS-lifecycle work lives in one PR and reverts cleanly. Only use if options 1-2 are insufficient; you'll need to re-cut the release.

## Max session duration

`idle_timeout` is reset on every frame, so a silence-streaming client could hold a `SessionTriplet` forever. `max_session_secs` is a *wall-clock* deadline that fires regardless of frame activity.

On cap expiry the server sends:
1. `ServerMessage::Error { message: "Maximum session duration exceeded", code: "max_session_duration_exceeded" }`
2. A best-effort `Final` frame (empty if no text accumulated).
3. `Close(1008 Policy Violation)`.

Overshoot ≤ 500 ms in the common case — a chunk that was already in flight when the deadline expired finishes first, then the loop hits the deadline branch on the next iteration.

### Rollback: disable the session cap

```sh
gigastt serve --max-session-secs 0
# or: GIGASTT_MAX_SESSION_SECS=0 gigastt serve
```

`0` parks the deadline at `u64::MAX / 2`, so `sleep_until` never fires. The session then runs as long as the idle timeout allows (default 300 s of silence).

### Config pitfalls

- If you set `--max-session-secs` *below* `--idle-timeout-secs`, the cap will always fire before the idle timer can apply. The server emits a `warn` at startup flagging this as a likely misconfiguration but does not refuse to start.
- Caps smaller than your typical transcription window will produce noisy `max_session_duration_exceeded` errors for legitimate clients.

## Pool exhaustion & backpressure

Each concurrent inference holds one `SessionTriplet` from a pool sized by
`--pool-size` (default 2). When all triplets are busy, callers wait up to
`--pool-checkout-timeout-secs` (default 30) for one to free up, then get
backpressure:

- **REST** `/v1/transcribe` and `/v1/transcribe/stream` → `503` with
  `Retry-After: <secs>` and `{"code":"timeout","retry_after_ms":…}`.
- **WebSocket** → `ServerMessage::Error { code: "timeout", retry_after_ms }`.

**Checkout timeout is the queue-vs-fail-fast knob.** A longer value keeps
callers waiting in line (absorb short saturation bursts); a shorter value
returns **503 / `timeout` + `retry_after_ms` sooner** so clients can back off or
retry another replica. See [Pool checkout timeout](#pool-checkout-timeout-queue-vs-fail-fast).

A single wedged inference run is bounded by `--inference-timeout-secs`
(default 600): the client gets `inference_timeout` (REST `504`, WS error +
close). The 600 s default already covers the advertised 10-minute (600 s audio)
file ceiling out of the box; raise it only for genuinely longer files.
Caveat — `spawn_blocking` can't be cancelled, so a truly hung run keeps
its triplet until it finishes (or restart); the timeout frees the *client*,
not the slot.

**Knobs**
- `--pool-size N` — total triplets (more concurrency, more RAM, and a small
  single-job RTF cost from thread split — see [Pool size tradeoffs](#pool-size-tradeoffs-ram-vs-concurrency-vs-rtf)).
- `--batch-pool-size N` — **split** N of those triplets for long REST file jobs
  so they can't starve interactive WebSocket / SSE (default 0 = shared pool).
  Not additive — total loaded sessions stay at `--pool-size` (see
  [batch_pool_size splits the pool](#batchpoolsize-splits-the-pool-not-additive)).
- `--pool-checkout-timeout-secs` — how long callers wait before backpressure
  (long = queue, short = fail-fast 503).
- `--inference-timeout-secs` — per-run ceiling; `0` disables.

**Metrics** (with `--metrics`)
- `gigastt_pool_available` / `gigastt_pool_waiters` — free triplets vs queued
  callers. Sustained `available == 0` with rising `waiters` = saturation.
- `gigastt_pool_timeouts_total` — checkout timeouts (backpressure events).
- `gigastt_inference_timeouts_total` — runs aborted by the inference timeout
  (a non-zero rate points at wedged runs or an over-tight timeout).
- When `--batch-pool-size` is set, the batch pool exports its own gauges
  `gigastt_batch_pool_available` and `gigastt_batch_pool_waiters`. Sample these
  to spot batch-pool saturation separately from the interactive pool.

**Triage**
1. If streaming is being starved by batch uploads, set `--batch-pool-size 1+`
   (remember it **splits** the existing pool). Monitor
   `gigastt_batch_pool_available` / `gigastt_batch_pool_waiters` to confirm the
   split is sized correctly.
2. If `gigastt_inference_timeouts_total` is climbing, capture a stuck run's
   input and check for an adversarial / huge file; raise the timeout only if the
   inputs are legitimately long.
3. If saturation is steady, scale `--pool-size` (watch RSS and single-job RTF)
   or add replicas.

## Model download failures

On first run `gigastt download` / `gigastt serve` fetches the model from
HuggingFace into `~/.gigastt/models/`, streaming each file to a `.partial`
path, verifying SHA-256, then atomically renaming. Concurrent processes
coordinate via an advisory `flock`; downloads use connect/read timeouts and a
bounded redirect policy.

**Symptoms & recovery**
- *SHA-256 mismatch* — a corrupt or tampered mirror. The `.partial` is deleted
  and nothing is promoted; just re-run. Persistent mismatches mean a bad mirror
  or a stale pinned checksum.
- *Hang / timeout mid-download* — network or HuggingFace issue; re-run (the
  `.partial` is re-fetched, not resumed).
- *"Model not found" after a crash* — a crash before rename leaves only a
  `.partial`; `model_files_exist()` ignores it, so re-running re-downloads.
- *Air-gapped / repeatable deploys* — bake the model into the image
  (`GIGASTT_BAKE_MODEL=1`, see `docs/deployment.md`) or pre-populate
  `~/.gigastt/models/` from a trusted copy.

To force a clean re-download, remove `~/.gigastt/models/` and re-run.

## Out-of-memory (OOM)

RSS scales with `--pool-size` (each triplet owns its ONNX sessions) plus ORT's
per-request scratch (a few minutes of 16 kHz audio allocates ~90+ MiB in the
encoder by itself). A default pool of 2 with the INT8 encoder sits around
~790 MiB resident (single session ~400 MiB); the FP32 encoder is much larger.

**Reduce footprint**
- Use the **INT8 encoder** (the default — auto-quantized on first run; don't
  pass `--skip-quantize`). It is ~4× smaller than FP32.
- Lower `--pool-size` (e.g. `1`–`2` on a 4 GB box). See also
  [Pool size tradeoffs](#pool-size-tradeoffs-ram-vs-concurrency-vs-rtf).
- `--pool-min-size 1` lets the server **boot on a degraded pool** if some
  triplets fail to load under memory pressure, instead of failing outright.
- On edge hosts, leave punctuation off (`--punctuation off`) if you do not need
  restored casing — the RuPunct model adds a small ready-RSS tax when present
  (see [Optional model ready tax](#optional-model-ready-tax)).
- The REST upload path is zero-copy (`bytes::Bytes` end-to-end), so concurrent
  large uploads no longer multiply the body in RAM — but the decoded PCM and
  encoder scratch still scale with audio length and `--pool-size`.

**Triage**
1. Check `terminationGracePeriodSeconds` isn't masking an OOM-kill as a slow
   shutdown.
2. Confirm the INT8 encoder is in use (`/v1/models` reports `"encoder":"int8"`).
3. Cap concurrency: `--pool-size` × (per-triplet RSS + peak scratch) must fit
   the box with headroom. Keep free RAM for admin reload if you use it
   ([Admin reload headroom](#admin-reload-headroom)).

## Resource & performance knobs

Operator notes for pool sizing, SKUs, VAD, and reload. Full flag list:
[`docs/cli.md`](cli.md).

### Pool size tradeoffs (RAM vs concurrency vs RTF)

- Each extra pool slot loads **another full encoder copy** — typically
  **hundreds of MiB** RSS (INT8 `rnnt` ≈ **+280…450 MiB** going from pool 1 → 2;
  default pool=2 ≈ **~790 MiB** ready, pool=1 ≈ **~400 MiB**).
- Pool > 1 also **splits encoder intra-op threads** across concurrent triplets.
  A **single** job on a busy multi-slot pool is therefore slower than the same
  job on pool=1 — typically about **+10–20% RTF** on a quiet serial workload
  (lab ≈ **+18%** at pool=2 vs pool=1). Raise pool for concurrent clients, not
  for single-stream latency.
- Edge / low-RAM: prefer **`--pool-size 1`**. Raise only when concurrent
  sessions need it and the host has free RAM after peak scratch.

### Pool checkout timeout (queue vs fail-fast)

`--pool-checkout-timeout-secs` (default **30**) is how long a handler waits
for a free triplet when the pool is full:

| Setting | Behaviour |
|---|---|
| **Longer** (e.g. 60–120) | Callers **queue** longer; fewer 503s under short bursts; higher tail latency and more in-flight waiters |
| **Shorter** (e.g. 5–15) | **Fail-fast**: REST `503` + `Retry-After` / WS `timeout` + `retry_after_ms` sooner so clients can back off or hit another replica |

Tune with `gigastt_pool_waiters` and `gigastt_pool_timeouts_total`. Details:
[Pool exhaustion & backpressure](#pool-exhaustion--backpressure).

### batch_pool_size splits the pool (not additive)

`--batch-pool-size N` **carves N triplets out of** `--pool-size` for long REST
file jobs. It does **not** allocate extra idle triplets.

Example: `--pool-size 4 --batch-pool-size 1` → **3** interactive (WS/SSE) +
**1** batch, total still **4** loaded sessions. `0` (default) = shared pool.
Clamped so at least one interactive triplet remains.

### Admin reload headroom

`POST /v1/admin/reload` builds a **second** engine, warms it, then swaps
atomically. Peak RSS during a successful reload is about **+0.5× ready** on top
of the live process — roughly **+536 MiB** at **pool=1** INT8 `rnnt` on the lab
host. Edge boxes with almost no free RAM can OOM mid-reload even when steady-state
`pool=1` fits; keep headroom or restart the process instead. See
[Admin reload](api.md#admin-reload).

### VAD for pause-rich long files

Enable **`--vad`** (or `GIGASTT_VAD=1`) for **meetings, podcasts, and other
pause-rich** long audio: Silero skips silence before decode and can finalize
streaming segments on trailing silence. On silence-rich material, wall time can
improve by up to about **×2.6** RTF vs running the full encoder over every
quiet stretch. Continuous speech gains little; VAD still downloads the Silero
model on first use (~few MB).

```sh
# Long meeting / podcast file
gigastt transcribe meeting.wav --vad
# Server-wide for REST + WS
gigastt serve --vad --pool-size 1
```

### Head SKU: ml_ctc is speed, not lean-RAM

`--model-variant ml_ctc` is a **throughput / RTF** choice (~**1.5×** faster RTF
than default `rnnt` in lab — e.g. RTF **~0.023** vs **~0.034**), **not** a
low-memory SKU. Ready RSS for `ml_ctc` is **about the same class as `rnnt`**
on multi-head installs (both ~225 MB INT8 encoder class). For less RAM use
**`--pool-size 1`**, not a head switch. Use `ml_ctc` / `ml_ctc_large` when you
need **ru/en/kk/ky/uz** or higher encode speed; keep `rnnt` for best Russian WER.

### Optional model ready tax

When the **punctuation** model is present and the pass is enabled (`auto`/`on`
for `rnnt`), ready RSS grows by about **+4…28 MiB** depending on host and
load path. Edge profiles that only need bare text can set
`--punctuation off` (and skip downloading `~/.gigastt/models/punct/`) to avoid
that tax. Accuracy of the acoustic model is unchanged; only casing/punctuation
restoration is skipped.

## Metrics

`gigastt_http_requests_total{path="/v1/ws",status="503"}` with code `shutting_down` in the body is the signal that upgrades are being rejected because shutdown was already in flight. Usually correlated with `terminationGracePeriodSeconds` being shorter than `shutdown_drain_secs`.

(Counter for cancelled-WS by reason is tracked separately — see `specs/prod-readiness-v1.0.md`.)

## On-call triage checklist

1. Pull a WS trace from the affected client. Confirm presence (or absence) of `Final` and the `Close` code.
2. Check server logs for `Shutdown signalled`, `Session cap reached`, or `Drain window expired`.
3. Confirm orchestrator `terminationGracePeriodSeconds` ≥ `shutdown_drain_secs + 5` (see `docs/deployment.md`).
4. If clients are seeing unexpected 503 `shutting_down`, the proxy LB may still be routing traffic after the pod started draining — add a `preStop` sleep to the k8s manifest so the LB deregisters the pod before the app sees `SIGTERM`.
5. If the cap is firing for legitimate long sessions, raise it — there's no correctness downside to `max_session_secs = 14400` (4 h), only a weaker guarantee against wedged sessions.
