# GigaSTT Workbook

Scenario-driven recipes for [gigastt](https://github.com/ekhodzitsky/gigastt),
the local Russian speech-to-text server powered by GigaAM v3. Each chapter
follows the same shape: **scenario → prerequisites → recipe → verifying the
result → common pitfalls → links**.

This book is a **cookbook, not a reference**. The canonical references stay in
[`docs/`](../../../) — the workbook links to them instead of duplicating them.

**Documented against gigastt 2.18.x.** Prefer resolving the latest release tag
in install scripts (`gh api …/releases/latest`) rather than hard-coding older
minors.

## I want to…

| Goal | Time | Chapter |
|---|---|---|
| First transcript on this machine | ~5–15 min | [Getting started](01-getting-started.md) |
| Batch a folder / watch a drop box / async jobs | ~15–30 min | [CLI and batch processing](02-cli-batch.md) |
| Transcribe PBX / Opus / raw telephony; stereo speakers | ~20 min | [Telephony & VoIP](03-telephony-voip.md) |
| Live captions or a voice bot over WebSocket | ~30 min | [Streaming over WebSocket](04-streaming-ws.md) |
| Ship a macOS / Electron / mobile app | ~30–60 min | [Desktop & embedded](05-desktop-embedded.md) |
| Run production with metrics, upgrades, model hot-reload | ~45 min | [Deployment & ops](06-deployment-ops.md) |
| Pick head / INT8 / GPU / pool size / punct / hotwords | ~20 min | [Models and backends](07-models-and-backends.md) |
| Label speakers on mono meetings | (in ch.3) | [Telephony — diarization](03-telephony-voip.md#mono-meeting-recording--speakers-via-diarization) |
| Decode an HTTP/WS error code | ~2 min | [Appendix A — Error codes](appendix-error-codes.md) |
| Ship air-gapped / offline | ~30 min | [Appendix B — Offline checklist](appendix-offline-checklist.md) |
| Install on Windows | ~10 min | [Getting started — Windows](01-getting-started.md#recipe-windows-prebuilt-binary) |

## Chapters

1. [Getting started](01-getting-started.md) — install (macOS/Linux/Windows/Docker/air-gap),
   first transcription. **Beginner · ~5–15 min**
2. [CLI and batch processing](02-cli-batch.md) — CLI, batch, and watch-mode
   recipes for audio files. **Beginner · ~15–30 min**
3. [Telephony & VoIP](03-telephony-voip.md) — G.711/G.722/Opus, PBX
   recordings, stereo split, and diarization. **Intermediate · ~20 min**
4. [Streaming over WebSocket](04-streaming-ws.md) — live partials over
   WebSocket (buffered RNN-T; VAD endpointing, session caps). **Intermediate · ~30 min**
5. [Desktop & embedded](05-desktop-embedded.md) — Swift/SPM, sidecar,
   Electron, UniFFI. **Intermediate · ~30–60 min**
6. [Deployment & ops](06-deployment-ops.md) — production deployment,
   monitoring, upgrades, admin reload. **Ops · ~45 min**
7. [Models and backends](07-models-and-backends.md) — model variants,
   quantization, execution providers, punctuation/ITN, hotwords. **Intermediate · ~20 min**

### Appendices

- [A — Error codes](appendix-error-codes.md) — REST/WS/close code jump table
- [B — Offline checklist](appendix-offline-checklist.md) — air-gapped operator list

The [Russian version](../../ru/src/README.md) mirrors this book chapter by
chapter.

## Which API?

| You have | Use | Do not use |
|---|---|---|
| A file on disk, WER matters | REST `/v1/transcribe` or CLI `transcribe` — [01](01-getting-started.md), [02](02-cli-batch.md) | Live WebSocket (WER is ~11–15 pp worse) |
| A folder / drop box | `transcribe-batch` / `watch` — [02](02-cli-batch.md) | One-shot `transcribe` in a loop |
| A long file you cannot wait on | `/v1/jobs` (`--enable-jobs`) — [02](02-cli-batch.md) | A single blocking REST call without a timeout plan |
| A microphone / call leg, partials while speaking | WebSocket `/v1/ws` — [04](04-streaming-ws.md) | REST; do not quote the 1000-row WER table for this path |
| An OpenAI-compatible client | `/v1/audio/transcriptions` — [docs/api.md](../../../api.md) | Custom WS if the client only speaks multipart |
| An in-process app (no server) | Bindings — [05](05-desktop-embedded.md) | Spawning `serve` unless you need crash isolation |

## More documentation

The full map of references (API, CLI, benchmarks, runbook, backends) lives in
[docs/README.md](../../../README.md). This book links out; it does not copy
those pages.

## Rules for contributors

- The workbook holds **recipes**; `docs/api.md`, `docs/cli.md`, and the
  AsyncAPI/OpenAPI schemas remain the canonical references. Link to them —
  do not copy their content.
- Every command and example in a chapter must be verified before merge.
- Inside the book (chapter ↔ chapter, chapter ↔ intro) use **relative `.md`
  links** — they work both on GitHub and in the rendered book. Links from the
  book to repository files (`docs/`, `crates/`, …) must be **absolute GitHub
  URLs** — relative ones 404 on the published site. No mdBook-specific
  templating.
- New chapters follow the [`_template.md`](_template.md) structure.
- **English is canonical.** The Russian book (`docs/workbook/ru/`) mirrors this
  one with identical file names; both versions are updated in the same PR.
- When a feature changes the documented surface (CLI flags, error codes, audio
  formats), update the chapter, the book `SUMMARY.md`, and the canonical
  references in the same PR — and keep the docs-drift gate green:
  `python3 scripts/check-docs-drift.py` (advisory in CI; it compares CLI
  flags, WS error codes, audio formats, mdBook TOCs, EN/RU heading-count
  parity, relative links, OpenAPI/SECURITY/crate pins, and workbook version
  currency + required recipe tokens against the code). Translation freshness
  is a review duty — the gate only counts `^#{1,6} ` lines (markdown headings
  *and* start-of-line `# ` comments in fences). Do not expand that gate.
  Before merge, for every changed chapter, read the other language and check:
  - same headings, Verify blocks, flags, env vars, paths, and error codes
  - same measured numbers (RAM, RTF, sizes) — do not invent figures
  - same start-of-line `# ` comment count (otherwise the parity gate fails)
  - no previous-minor version pins (resolve latest via `TAG`/`VER`, or keep
    `vX.Y.0` only as an example in a comment)
