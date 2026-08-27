# Privacy

gigastt is designed to keep all speech recognition entirely on the device that
runs it. This document describes precisely what data moves where and what is
retained.

## Audio and transcript data

- All audio processing runs locally via ONNX Runtime. Audio frames never leave
  the machine.
- Transcripts are returned to the caller of the local API. The inference
  server does **not** persist them unless you enable `/v1/jobs` (in-memory,
  TTL `--jobs-ttl-secs`) or use CLI `transcribe-batch` / `watch` (you chose
  the output path). Logs never include transcript text.
- Tracing logs (controlled by `RUST_LOG`) record request metadata such as
  duration and word count. They do not contain transcript text. PII sanitization
  of log output shipped in v0.9.6.

## Telemetry and analytics

- gigastt contains no telemetry, analytics, or "phone-home" code.
- No usage data, error reports, or performance metrics are transmitted to any
  external service.

## Network traffic

### Runtime

The only outbound network call a running gigastt process makes is the one-time
model download (ASR heads, and optionally punctuation / VAD / speaker models):

- Default `rnnt` / `e2e_rnnt` INT8 files come from **GitHub Releases**.
  `ml_ctc` / `ml_ctc_large` INT8 encoders come from HuggingFace
  (`istupakov/gigaam-multilingual-ctc-onnx` and
  `istupakov/gigaam-multilingual-large-ctc-onnx`). Optional punctuation,
  Silero VAD, and WeSpeaker diarization weights are also fetched when those
  features are first enabled.
- Each file is SHA-256 verified before use and written atomically to disk.
- After the initial download, gigastt operates fully offline.
- Audited: the HTTP client (`reqwest`) is referenced from exactly one runtime
  module — the model downloader (`gigastt-core/src/model/`) — and every fetch
  in that module funnels through a single download function. (The server
  crate's e2e tests also use `reqwest`, but as a dev-dependency that is never
  compiled into the shipped binary.) No other runtime
  code path opens outbound connections. `GIGASTT_OFFLINE=1` (or `--offline`)
  turns even that path into a fast, instructive error for air-gapped hosts.

### Build time (not runtime)

Building with the default `ort` features also downloads a prebuilt onnxruntime
native library at **compile** time (verified by an embedded checksum, outside
`Cargo.lock`). That is a developer/CI concern, not a runtime phone-home. Air-gapped
builds use `ort` with `default-features = false` + `load-dynamic` (or a vendored
onnxruntime); see [architecture.md](architecture.md).

## Server binding

- By default the server listens on `127.0.0.1` (loopback only). Traffic is
  therefore not reachable from other hosts on the network.
- Binding to a non-loopback address requires an explicit opt-in:
  `--bind-all` flag or `GIGASTT_ALLOW_BIND_ANY=1` environment variable.
- Cross-origin requests are denied by default; the origin allowlist is empty
  unless `--allow-origin` or `--cors-allow-any` is passed. Loopback origins
  (`localhost`, `127.0.0.1`, `::1`) are always allowed regardless.

## Prometheus metrics

When `--metrics` is enabled, the `/metrics` endpoint exposes request counts and
latency histograms. These metrics contain no audio content, transcript text, or
user-identifying information — only aggregate HTTP counters and durations.

## Summary

| Data type | Leaves the device? | Stored on disk? | Logged? |
|-----------|-------------------|-----------------|---------|
| Audio frames | No | No | No |
| Transcript text | No | CLI batch/watch write the files you asked for; `/v1/jobs` keeps results in memory for `--jobs-ttl-secs` | No |
| Request metadata (duration, word count) | No | Only if you redirect logs | Yes (word count only) |
| Model weights | No (downloaded once, then local) | Yes (`~/.gigastt/models/`) | No |
