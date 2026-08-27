# Deployment & ops

## Scenario

You are the admin who runs gigastt on a server, not a laptop. The path of
this chapter: **install → restrict → observe → upgrade** — one supervised
service (systemd or Docker), metrics flowing into Prometheus/Grafana, alerts
for the failure modes that matter, and an upgrade routine that does not cut
live transcription sessions.

Every recipe ends with a **Verify** step. Flags are checked against
`gigastt serve --help`; the full flag reference lives in
[docs/cli.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/cli.md)
and is not repeated here.

## Prerequisites

- gigastt installed (binary, package, or image) — see
  [Getting started](01-getting-started.md).
- A Linux host with **4+ GB RAM** is the usual production floor (OS + peaks).
  Process RAM: ~46 / ~66 MB resident at pool 1 / 2. Figures:
  [docs/benchmarks.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/benchmarks.md).
- For the systemd path: systemd 241 or newer (any modern distro, including
  Astra Linux, RED OS, ALT) and root access.
- For the Docker path: Docker 20.10+; the NVIDIA Container Toolkit only for
  the CUDA variant.
- The model either downloadable once (~225 MB lean INT8) or pre-installed from
  the offline bundle / model deb.

## Recipe

### Docker

Each tagged release publishes multi-arch images to GHCR — prefer pulling over
building:

```sh
TAG=$(gh api repos/ekhodzitsky/gigastt/releases/latest -q .tag_name)  # e.g. v2.18.0
VER=${TAG#v}
docker pull ghcr.io/ekhodzitsky/gigastt:${VER}        # CPU, linux/amd64 + linux/arm64
docker pull ghcr.io/ekhodzitsky/gigastt:${VER}-cuda   # CUDA, linux/amd64
```

Pin `$VER` for reproducible deploys; `:latest` / `:cuda` float.

Run with a named volume so the ~225 MB INT8 model (and any
encoder) survives container replacement:

```sh
docker run -d --name gigastt \
  -p 127.0.0.1:9876:9876 \
  -v gigastt-models:/home/gigastt/.gigastt/models \
  ghcr.io/ekhodzitsky/gigastt:${VER}
```

Notes:

- The image's default command is `serve --port 9876 --host 0.0.0.0
  --bind-all` (container networking needs it); `-p 127.0.0.1:9876:9876`
  keeps the host-side exposure on loopback. Put your TLS proxy in front
  exactly as with a bare install.
- The container runs as the unprivileged `gigastt` user; the model directory
  inside is `/home/gigastt/.gigastt/models` — that is the volume mount point.
- The image carries a `HEALTHCHECK` on `/health`, so `docker ps` shows
  `healthy` as soon as the port serves. During the first-run lean INT8
  download (~225 MB, if the volume is empty) `/health` answers `200` with
  `model:"loading"` while `/ready` answers
  `503 {"status":"not_ready","reason":"initializing"}` — gate traffic on
  `/ready`, not `/health`.
- **Baked image** (zero cold start, +~225 MB INT8): build locally with the model
  inside — `docker build --build-arg GIGASTT_BAKE_MODEL=1 -t gigastt:baked .`
- **CUDA**: `docker run --gpus all -p 127.0.0.1:9876:9876
  ghcr.io/ekhodzitsky/gigastt:${VER}-cuda` (requires the NVIDIA Container
  Toolkit; the binary falls back to CPU when no GPU is present).

**Verify:**

```sh
curl -s http://127.0.0.1:9876/ready
# {"status":"ready","pool_available":2,"pool_total":2}
curl -s http://127.0.0.1:9876/health
# {"status":"ok","model":"gigaam-v3-rnnt","variant":"rnnt","version":"...","punctuation":true,"itn":true}
```

### Air-gapped / offline installation

Core ASR for the default `rnnt` head is a **lean INT8-only** set (~225 MB):
`v3_rnnt_encoder_int8.onnx`, `v3_rnnt_decoder.onnx`, `v3_rnnt_joint.onnx`,
`v3_vocab.txt`. Prefer `gigastt download`. Full operator detail:
[deployment.md — Lean INT8-only install](https://github.com/ekhodzitsky/gigastt/blob/main/docs/deployment.md#lean-int8-only-install)
(canonical English).


For hosts with no internet access, each release ships a self-contained
tarball per Linux target — binary + pre-quantized INT8 `rnnt` model +
punctuation model + systemd unit + installer — and two Debian packages
(`gigastt_<ver>_<arch>.deb` + `gigastt-model-int8_<ver>_all.deb`). The full
bundle inventory lives in
[README-OFFLINE.md](https://github.com/ekhodzitsky/gigastt/blob/main/packaging/offline/README-OFFLINE.md)
and is not repeated here.

On a connected machine, download and **verify before** carrying the files
over (the why and the threat model:
[docs/verifying-releases.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/verifying-releases.md)):

```sh
TAG=$(gh api repos/ekhodzitsky/gigastt/releases/latest -q .tag_name)  # e.g. v2.18.0
VER=${TAG#v}
gh release download "$TAG" -R ekhodzitsky/gigastt \
    -p "gigastt-${VER}-offline-x86_64-unknown-linux-gnu.tar.gz" \
    -p "gigastt-${VER}-offline-x86_64-unknown-linux-gnu.tar.gz.sha256" \
    -p "gigastt-${VER}-offline-x86_64-unknown-linux-gnu.tar.gz.minisig"
sha256sum -c "gigastt-${VER}-offline-x86_64-unknown-linux-gnu.tar.gz.sha256"
minisign -Vm "gigastt-${VER}-offline-x86_64-unknown-linux-gnu.tar.gz" -p gigastt.pub
gh attestation verify "gigastt-${VER}-offline-x86_64-unknown-linux-gnu.tar.gz" \
    --repo ekhodzitsky/gigastt
```

On the target host:

```sh
mkdir gigastt-offline && tar xf "gigastt-${VER}-offline-x86_64-unknown-linux-gnu.tar.gz" -C gigastt-offline && cd gigastt-offline
sudo ./install.sh    # verifies SHA256SUMS.txt, then installs binary + models + unit
sudo systemctl enable --now gigastt
```

Debian-family alternative:

```sh
sudo dpkg -i "gigastt_${VER}_amd64.deb" "gigastt-model-int8_${VER}_all.deb"
sudo systemctl enable --now gigastt
```

The model is already INT8 — no download, no quantization step, no network.
The installed unit sets `GIGASTT_OFFLINE=1` via `/etc/gigastt/gigastt.env`,
so any code path that would fetch a model (enabling `--vad`, diarization, an
alternative recognition head) **fails fast with an error naming the file to
provide** instead of hanging on a connect timeout. To add optional models
later, run `gigastt download` on a connected machine and copy the files into
`/usr/share/gigastt/models/`.

Typical failures:

- `install.sh` aborts with `sha256sum: WARNING: 1 computed checksum did NOT
  match` — the tarball was corrupted on its way to the air-gapped host.
  Re-verify the outer `.sha256`, re-copy, re-run; nothing is installed
  partially.
- An offline-mode error naming a missing file (e.g. the VAD model after
  enabling `--vad`) — that model is not in the bundle; fetch it on a
  connected machine and copy it over.

**Verify:**

```sh
systemctl is-active gigastt
# active
curl -s http://127.0.0.1:9876/health
# {"status":"ok",...} — served immediately, the model is pre-installed
```

### systemd service

The hardened unit ships in
[packaging/systemd/](https://github.com/ekhodzitsky/gigastt/tree/main/packaging/systemd)
and is installed by both the deb and the offline bundle. Key properties (the
unit itself is short and commented — read it for the full list):

- Runs as the unprivileged `gigastt` user; models under
  `/usr/share/gigastt/models` are read-only to it.
- Loopback bind only (`127.0.0.1:9876`); expose the API through a reverse
  proxy.
- `Restart=on-failure`, `RestartSec=5` — a crash is restarted, a clean
  `systemctl stop` is not.
- Hardening set compatible with systemd 241 (`ProtectSystem=strict`,
  `NoNewPrivileges`, `PrivateTmp`, …), so the unit works unmodified on Astra
  Linux, RED OS and ALT.
- Overrides live in `/etc/gigastt/gigastt.env` (`GIGASTT_*` variables,
  `RUST_LOG`), loaded via `EnvironmentFile`.

Logs go to the journal:

```sh
journalctl -u gigastt -f          # follow
journalctl -u gigastt -n 100      # recent
```

Change log verbosity by editing `/etc/gigastt/gigastt.env`
(`RUST_LOG=gigastt=debug`), then `sudo systemctl restart gigastt`.

Change flags with a drop-in — never edit the shipped unit (package upgrades
replace it). `ExecStart` must be cleared before being re-set:

```ini
# sudo systemctl edit gigastt
[Service]
ExecStart=
ExecStart=/usr/bin/gigastt serve --model-dir /usr/share/gigastt/models --punct-model-dir /usr/share/gigastt/models/punct --metrics
```

`systemctl restart gigastt` sends `SIGTERM`; the server drains live
WebSocket/SSE sessions — each client receives a `Final` frame +
`Close(1001 Going Away)` — for up to `--shutdown-drain-secs` (default 10 s),
well inside systemd's default 90 s stop timeout. How to use this for version
bumps: [Upgrades and rollback](#upgrades-and-rollback) below.

**Verify:**

```sh
systemctl status gigastt --no-pager
curl -s http://127.0.0.1:9876/health
```

### Observability

Metrics are opt-in and served on a **separate listener** — never on the API
port, so they sit outside the CORS allowlist and the per-IP rate limiter:

```sh
gigastt serve --metrics                                  # http://127.0.0.1:9090/metrics
gigastt serve --metrics --metrics-listen 127.0.0.1:9100  # custom port
```

Keep the listener on loopback unless your Prometheus runs on another host —
and even then bind it to a trusted interface, never a public one.

Minimal Prometheus wiring (`prometheus.yml`):

```yaml
scrape_configs:
  - job_name: gigastt
    static_configs:
      - targets: ["127.0.0.1:9090"]

rule_files:
  - /etc/prometheus/rules/gigastt-alerts.yml   # copy of docs/observability/alerts.yml
```

Watch **pool saturation** (`gigastt_pool_available == 0`), **5xx**, and
node-level RAM. Import rules and the Grafana dashboard from
[`docs/observability/`](https://github.com/ekhodzitsky/gigastt/tree/main/docs/observability)
— do not copy the metric catalog here. Logs: `RUST_LOG=gigastt=info` (debug
for triage); no transcript text
([docs/privacy.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/privacy.md)).

**Verify:**

```sh
curl -s http://127.0.0.1:9876/ready > /dev/null   # samples the pool gauges once
curl -s http://127.0.0.1:9090/metrics | grep '^gigastt_pool_available'
# gigastt_pool_available 2
```

### Secure by default

Do not weaken the defaults: loopback bind, origin allowlist, body/frame
caps. Proxy + rate-limit snippets:
[docs/deployment.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/deployment.md).
Verify a release before install:
[docs/verifying-releases.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/verifying-releases.md).
Minimum routine:

```sh
TAG=$(gh api repos/ekhodzitsky/gigastt/releases/latest -q .tag_name)  # e.g. v2.18.0
VER=${TAG#v}
minisign -Vm "gigastt-${VER}-x86_64-unknown-linux-gnu.tar.gz" -p gigastt.pub
gh attestation verify "gigastt-${VER}-x86_64-unknown-linux-gnu.tar.gz" \
    --repo ekhodzitsky/gigastt
```

- **Privacy.** No telemetry, no outbound calls after the one-time model
  download, transcripts never logged
  ([docs/privacy.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/privacy.md)).

**Verify:**

```sh
ss -ltnp | grep 9876
# tcp LISTEN 0 ... 127.0.0.1:9876 ...   (loopback only)
curl -s -o /dev/null -w '%{http_code}\n' \
    -H 'Origin: https://attacker.example' http://127.0.0.1:9876/v1/models
# 403
```

### Hot-reload the model without restart

When you replace model files on disk (new INT8 encoder, switched head files,
refreshed punct model) you can rebuild the engine **in place** without
stopping `serve`:

```sh
# Must be called from loopback — non-loopback peers get 403 loopback_only
# even with --bind-all.
curl -s -X POST http://127.0.0.1:9876/v1/admin/reload
# {"reloaded":true,"variant":"rnnt","encoder":"int8"}
```

The server rebuilds from the **boot recipe** (model dir, pool sizes, punct /
ITN / VAD / hotwords), warms the new engine, then atomically swaps it in.
In-flight requests finish on the old engine. A build failure leaves the
previous model serving (`503 reload_failed`). Concurrent reloads return
`409 reload_in_progress`. Full contract:
[docs/api.md — Admin reload](https://github.com/ekhodzitsky/gigastt/blob/main/docs/api.md#admin-reload).

**Verify:**

```sh
curl -s -X POST http://127.0.0.1:9876/v1/admin/reload | tee /tmp/reload.json
python3 -c "import json; d=json.load(open('/tmp/reload.json')); assert d['reloaded'] is True"
# From a non-loopback bind (only if you deliberately opened one): expect 403.
```

### Upgrades and rollback

Pin what you deploy (image tag, deb version) so an upgrade is a deliberate,
reversible step. The model directory is state: it persists across upgrades,
the engine auto-detects the installed recognition head, and **no silent
re-download happens** when you bump the binary. Prefer resolving the latest
release when scripting installs:

```sh
TAG=$(gh api repos/ekhodzitsky/gigastt/releases/latest -q .tag_name)   # e.g. v2.18.0
VER=${TAG#v}
```

Docker (upgrade to `$VER` from the block above):

```sh
docker pull ghcr.io/ekhodzitsky/gigastt:${VER}
docker stop --time 15 gigastt && docker rm gigastt
docker run -d --name gigastt \
  -p 127.0.0.1:9876:9876 \
  -v gigastt-models:/home/gigastt/.gigastt/models \
  ghcr.io/ekhodzitsky/gigastt:${VER}
```

`docker stop` sends `SIGTERM`; `--time 15` gives the drain window
(`--shutdown-drain-secs`, default 10 s) room to finish before `SIGKILL` —
Docker's default of 10 s races the drain. Clients receive `Final` +
`Close(1001)` and reconnect; short REST uploads in flight may need a retry.

systemd / deb:

```sh
sudo dpkg -i "gigastt_${VER}_amd64.deb"
sudo systemctl restart gigastt
journalctl -u gigastt -f    # expect a clean drain, no "Drain window expired"
```

On Kubernetes the same rule applies from the orchestrator side:
`terminationGracePeriodSeconds` ≥ `shutdown_drain_secs + 5` (full manifest:
[docs/deployment.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/deployment.md#graceful-shutdown--session-caps)).

**Rollback.** Re-deploy the **previous** tag or package — the on-disk model set
is unchanged, so the old binary starts against the same files:

```sh
docker run -d --name gigastt \
  -p 127.0.0.1:9876:9876 \
  -v gigastt-models:/home/gigastt/.gigastt/models \
  ghcr.io/ekhodzitsky/gigastt:${PREV_TAG#v}   # e.g. previous release tag
# or: sudo dpkg -i gigastt_${PREV_TAG#v}_amd64.deb && sudo systemctl restart gigastt
```

If a drain-related regression breaks your WebSocket clients after an upgrade,
the escape hatch is `--shutdown-drain-secs 0` (clamped to 1 s) — see
[docs/runbook.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/runbook.md)
for the full symptom table.

**Verify** (after every upgrade or rollback):

```sh
curl -s http://127.0.0.1:9876/health
# "version" is the release you deployed; "model"/"variant" are unchanged
curl -s http://127.0.0.1:9876/ready
```

## Verifying the result

End-to-end smoke after any recipe above:

```sh
systemctl is-active gigastt || docker ps --filter name=gigastt --format '{{.Status}}'
curl -s http://127.0.0.1:9876/health     # status ok, expected version
curl -s http://127.0.0.1:9876/ready      # ready, pool_available >= 1
curl -s http://127.0.0.1:9090/metrics | grep '^gigastt_pool_available'
```

Then transcribe one short file through the API you actually expose (REST
recipes: [CLI and batch processing](02-cli-batch.md); CLI check:
[Getting started](01-getting-started.md)).

## Common pitfalls

- **OOM — container or service killed.** Budget resident (~46 / ~66 MB at
  pool 1 / 2). `--pool-min-size 1` boots degraded. Figures and OOMKilled:
  [docs/benchmarks.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/benchmarks.md),
  [docs/runbook.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/runbook.md).
- **503 `timeout` under load.** Every triplet is busy and a caller waited out
  `--pool-checkout-timeout-secs` (30 s): REST gets `503` + `Retry-After`,
  WebSocket gets an error with `retry_after_ms`. This is backpressure, not a
  bug — raise `--pool-size`, split batch work off with `--batch-pool-size`,
  and watch `gigastt_pool_waiters` / `gigastt_pool_timeouts_total`.
- **`/metrics` unreachable from the Prometheus host.** By design: the
  listener defaults to `127.0.0.1:9090`. Point the scraper at the gigastt
  host itself, or deliberately re-bind with `--metrics-listen` on a trusted
  interface — never a public one. A scraper still aimed at `:9876/metrics`
  gets 404: metrics moved off the API port.
- **Readiness probes flapping on first start.** The first run may download
  lean INT8 (~225 MB) if the model dir is empty. `/health` returns `200` with
  `model:"loading"` the whole time, but `/ready` returns `503 initializing`.
  If your load balancer routes on `/health`, early clients get 503s — probe
  `/ready`, or pre-install / bake the model so the window disappears.
- **Rate limiter punishing everyone behind a proxy.** Without
  `--trust-proxy` — and a proxy that overwrites rather than appends
  `X-Forwarded-For` — all clients share one bucket keyed on the proxy's
  address. Symptoms and the exact proxy configuration:
  [docs/deployment.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/deployment.md#rate-limiter--x-forwarded-for).

## Links

- [Getting started](01-getting-started.md) — install and first transcription
- [CLI and batch processing](02-cli-batch.md) — REST / SSE / jobs recipes
- [Streaming over WebSocket](04-streaming-ws.md) — WebSocket protocol patterns
- [Appendix A — Error codes](appendix-error-codes.md) — HTTP/WS/close jump table
- [Appendix B — Offline checklist](appendix-offline-checklist.md) — air-gapped operator list
- [docs/deployment.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/deployment.md) — reverse proxy, TLS, Kubernetes manifests
- [docs/runbook.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/runbook.md) — symptom → cause → escape hatch
- [docs/cli.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/cli.md) — full `serve` flag reference
- [docs/observability/alerts.yml](https://github.com/ekhodzitsky/gigastt/blob/main/docs/observability/alerts.yml) — Prometheus alerting rules
- [docs/observability/dashboard.json](https://github.com/ekhodzitsky/gigastt/blob/main/docs/observability/dashboard.json) — Grafana dashboard
- [docs/verifying-releases.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/verifying-releases.md) — minisign, SBOM, SLSA provenance
- [docs/privacy.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/privacy.md) — what data moves where
- [packaging/systemd/](https://github.com/ekhodzitsky/gigastt/tree/main/packaging/systemd) — unit + env file
- [packaging/offline/README-OFFLINE.md](https://github.com/ekhodzitsky/gigastt/blob/main/packaging/offline/README-OFFLINE.md) — offline bundle contents
