# Zero-Friction Trial: Hosted Demo vs In-Browser WASM

**Status:** DRAFT for human review
**Date:** 2026-06-13
**Scope:** Design only. Nothing here changes README, README_RU, or any code. This document evaluates two paths for closing the "no zero-friction trial" gap and recommends one to pursue first.

---

## Problem statement

Today every way to try gigastt requires downloading the ~850 MB model (~225 MB INT8 encoder + decoder + joiner) and building or running the server locally. There is no hosted demo and no in-browser path. An evaluator who wants to answer one question — *"how good is this at Russian speech?"* — cannot, without a multi-step local install. For a pre-traction, solo-maintained project, that friction is the difference between a casual GitHub visitor becoming a user and bouncing.

The audit's recommendation: stand up a **hosted mic → transcript demo** so anyone can speak Russian into their browser and watch the transcript appear, with no install. WASM (run the model entirely client-side) is raised as an alternative worth assessing honestly.

This doc designs both and picks one.

---

## Option 1 — Hosted mic → transcript demo

A static web page captures microphone audio in the browser, downsamples it to 16 kHz PCM16, and streams it over the **existing** gigastt WebSocket endpoint (`/v1/ws`) to a hosted instance of the server. Partial and final transcripts stream back and render live. The server is unchanged — this is purely a deployment + thin frontend exercise.

### Why this is low-effort

The server already does the hard part. From `crates/gigastt-core/src/protocol/mod.rs` and `crates/gigastt/src/server/ws.rs`:

- **WebSocket streaming exists** at `/v1/ws`. The client sends raw PCM16 binary frames; the server emits `Ready`, `Partial`, and `Final` JSON messages (`ServerMessage` enum). `PROTOCOL_VERSION = "1.0"`.
- **Sample-rate negotiation exists.** A `Configure { sample_rate }` message (must precede the first audio frame) accepts 8/16/24/44.1/48 kHz; the server resamples to 16 kHz internally via rubato. The browser can capture at 48 kHz and let the server resample, or downsample client-side to 16 kHz to cut bandwidth ~3×.
- **Abuse controls exist.** Per-IP token-bucket rate limiting (`--rate-limit-per-minute`, `--rate-limit-burst`), a wall-clock session cap (`--max-session-secs`, default 3600 — for a demo, set this *low*, e.g. 30–60 s), idle timeout (`--idle-timeout-secs`), pool-saturation backpressure (WS error carries `retry_after_ms`), and an origin allowlist (`--allow-origin`).
- **Metrics exist** (`--metrics` → Prometheus `/metrics`) for watching abuse/load.

So Option 1 is roughly: **(a)** a ~150-line static HTML/JS page, **(b)** a container deployment of the binary that already exists (the project already ships `Dockerfile`), **(c)** correct flags.

### Architecture

```
Browser (static page, any CDN/host)
  navigator.mediaDevices.getUserMedia({ audio })
    -> AudioContext / AudioWorklet
       -> Float32 @ 48 kHz  --(downsample)-->  Int16 PCM @ 16 kHz
  WebSocket  wss://demo.<host>/v1/ws
    1. send  {"type":"configure","sample_rate":16000}
    2. send  <binary PCM16 frames, ~100-250 ms each>
    <- {"type":"ready", ...}
    <- {"type":"partial", "text": "..."}   (interim, may change)
    <- {"type":"final",   "text": "..."}   (utterance complete)
  Render partial in grey, promote to black on final.

Hosted gigastt server (the existing binary, in a container)
  gigastt serve --bind-all \
    --allow-origin https://demo.<host> \
    --rate-limit-per-minute 20 --rate-limit-burst 5 \
    --max-session-secs 45 \
    --metrics
```

**Frontend specifics**
- Use `getUserMedia` + an `AudioWorkletNode` (preferred over the deprecated `ScriptProcessorNode`) to pull Float32 PCM. Linear-resample 48 kHz → 16 kHz (or request a 16 kHz context where supported) and convert to little-endian Int16 (`sample * 0x7FFF`, clamped). Send each ~100–250 ms chunk as a binary WS frame.
- The server carries trailing odd PCM16 bytes across frames (V1-25), so chunk sizes don't need to be even — but keeping them even avoids the edge case entirely.
- TLS is required: browsers only grant mic access on secure contexts (`https://` / `wss://`), and most hosts terminate TLS for you.

**Backend specifics — required flags**
- `--bind-all` (or `GIGASTT_ALLOW_BIND_ANY=1`): the server binds loopback-only by default; a hosted instance must listen on `0.0.0.0`. (The provided `Dockerfile` already passes `--bind-all`.)
- `--allow-origin https://demo.<host>`: lock cross-origin WS upgrades to the demo page's origin. Do **not** use `--cors-allow-any` in production-facing demo.
- `--rate-limit-per-minute` + `--rate-limit-burst`: cap connections per IP. `/health` is exempt.
- `--max-session-secs 30..60`: critical. The default 3600 is wrong for a public demo — a single silent socket could pin a pool slot for an hour. A short cap closes idle/abusive sockets fast.
- `--metrics`: enable so you can watch `gigastt_http_requests_total` and spot abuse.
- Consider lowering `--pool-size` to match the host's CPU; each concurrent stream needs a pooled session triplet.

### Hosting recommendation

Constraints: solo dev, pre-traction, wants cheap/free, model is 225 MB INT8 (fits typical free-tier disk), needs a long-lived process that holds a WebSocket (rules out pure serverless/edge-function platforms that don't support stateful WS well), CPU-only inference is fine for a demo.

| Option | Free/cheap tier | WS support | Disk for 225 MB model | Cold start | Notes |
|---|---|---|---|---|---|
| **Hugging Face Spaces (Docker SDK)** | Free CPU tier | Yes (full HTTP server) | Yes | Sleeps when idle; wakes on hit (seconds) | **Recommended.** Audience-fit: HF is where Russian-ASR evaluators already are; the GigaAM model itself lives on HF. Docker Spaces run an arbitrary container — drop the existing `Dockerfile`. Free, no credit card. Idle-sleep is fine for a demo (label "first request wakes the box"). |
| **Fly.io** | Small always-on/scale-to-zero VM, low cost | Yes | Yes (volume) | scale-to-zero wake ~seconds | Strong #2. Real VM, full control, global anycast. Not strictly free but cheap; `fly.toml` + the existing Dockerfile. Good if HF Spaces' sleep behavior or CPU is too constrained. |
| **Render** | Free web service (sleeps after inactivity) | Yes | Yes | Wake from sleep can be slow (tens of seconds) on free tier | Workable; free tier sleep/wake is the roughest of the three. |
| **Small VPS** (Hetzner/Fly machine/etc.) | ~€4/mo | Yes | Yes | None (always on) | Most control, no cold start, but you own ops/TLS/patching. Overkill pre-traction. |

**Recommendation: Hugging Face Spaces (Docker SDK) first.** Rationale: zero cost, no card, reuses the existing `Dockerfile` almost verbatim, and — uniquely — it places the demo in front of exactly the audience evaluating Russian ASR models (HF model hub). The model can be baked into the image (the repo already supports `GIGASTT_BAKE_MODEL=1`, ~1.1 GB image) to avoid a cold download, or pulled on boot. Fly.io is the fallback if Spaces' free CPU/idle-sleep proves too limiting under real traffic.

**Cold-start note.** On INT8 + CoreML the encoder is fast, but hosted demos run **CPU-only** (no Neural Engine, no CUDA on free tiers). First inference also pays the one-time INT8 quantization unless the image ships a pre-quantized encoder — bake it (`gigastt quantize` at build time, or `GIGASTT_BAKE_MODEL=1`) so the container starts warm. The server already exposes `Engine::warmup()` and calls it before serving, so steady-state latency is fine; the concern is only the first cold boot after idle-sleep. Label it.

### Privacy framing (non-negotiable)

gigastt's entire pitch is **on-device, no cloud, no API keys, full privacy.** A hosted demo *inverts* that — audio leaves the user's machine and is transcribed on a server you operate. This must be stated plainly on the page and in any link to it, or it actively undermines the project's core message. Suggested copy:

> ⚠️ **This hosted demo is the one exception to gigastt's on-device promise.** Your microphone audio is streamed to a public server to show transcription quality without an install. It is processed in memory and not stored. For real use, run gigastt locally — your audio never leaves your machine.

Operationally back that up: the server holds audio only in memory (no transcript/audio persistence exists in the codebase — keep it that way), and the short `--max-session-secs` cap limits exposure.

### Abuse controls (summary)

Already in the binary, just enable them: per-IP rate limit, short max-session cap, idle timeout, pool backpressure (503 + `Retry-After` on REST, `retry_after_ms` on WS), origin allowlist. Additionally: put the host's own edge/CDN rate limiting in front if available, and keep `--metrics` on to watch for abuse. A demo doesn't need auth, but a per-IP daily cap (host-level) is worth adding if it gets popular.

### Concrete next steps (Option 1)

1. Write the static page (`getUserMedia` + AudioWorklet → 16 kHz PCM16 → WS, render partial/final). ~1 day.
2. Add a Spaces-friendly container entrypoint with the demo flags above (the existing `Dockerfile` is 90% there; main change is the flag set and a short `--max-session-secs`). ~0.5 day.
3. Bake the pre-quantized model into the image to kill cold-start (`GIGASTT_BAKE_MODEL=1`). ~0.5 day.
4. Deploy to HF Spaces (Docker SDK), set the page origin in `--allow-origin`, smoke-test mic → transcript end-to-end on Chrome + Safari. ~0.5 day.
5. Add the privacy banner and a prominent "this is a cloud demo; run locally for privacy" note. ~trivial.

**Total: ~2–3 focused days.** No server code changes required.

---

## Option 2 — In-browser WASM (no server)

Run the whole pipeline (mel features → Conformer encoder → RNN-T decode loop) client-side via WebAssembly, so there is no server at all and the on-device/no-cloud promise is preserved even in the demo.

Two sub-paths exist, and they are very different in feasibility:

### Path A — compile the `ort` Rust crate to WASM and reuse `gigastt-core`

**Verdict: not viable today as a drop-in.** Findings:

- `ort` (pyke) is a **wrapper around Microsoft's ONNX Runtime C/C++ library** ([pykeio/ort](https://github.com/pykeio/ort)). gigastt pins `ort` `2.0.0-rc.12`, which wraps ONNX Runtime 1.26 ([crates.io/ort](https://crates.io/crates/ort), [docs.rs/ort](https://docs.rs/ort)). To run in a browser, ONNX Runtime itself must be compiled to WASM (via Emscripten) — you cannot link the native desktop library.
- The project's stated air-gapped/offline build path uses `ort`'s `load-dynamic` feature. **`load-dynamic` cannot work in WASM**: it `dlopen`s the shared library at runtime, and the browser sandbox has no `dlopen`/`LoadLibrary` equivalent ([pykeio/ort DeepWiki — Installation & Setup](https://deepwiki.com/pykeio/ort/2.1-installation-and-setup)).
- `ort`'s actual WASM story is its **`alternative-backend`** feature + `ort::set_api()`, which disables linking the native library and wires in a backend exposing `OrtGetApiBase` ([pykeio/ort DeepWiki](https://deepwiki.com/pykeio/ort/2.1-installation-and-setup)). This is an advanced, sparsely-documented path. `wasm32-unknown-unknown` "does not (easily) support interop with C/C++ code," so the realistic target is `wasm32-unknown-emscripten` ([rustc book — wasm32-unknown-emscripten](https://doc.rust-lang.org/beta/rustc/platform-support/wasm32-unknown-emscripten.html)). The only WASM usage visible in `ort`'s ecosystem is one third-party OCR project (retto) — not the streaming-ASR shape gigastt needs, and not a documented, supported recipe.
- Even if it linked, `gigastt-core`'s WASM build would inherit its whole native dependency tree (symphonia, rubato, threading model, file I/O in `model/mod.rs`), much of which assumes a native environment. This is a port, not a recompile.

**Effort/risk for Path A:** high effort, high risk, poorly-trodden. Not appropriate pre-traction.

### Path B — ONNX Runtime Web (JS/WASM) + reimplement the decode loop in JS

This is the *realistic* WASM path, and it is the one whisper.cpp's demo and transformers.js actually use — neither runs a Rust crate; both run ONNX Runtime compiled to WASM and drive it from JS.

- **ONNX Runtime Web** compiles the native CPU engine to WASM via Emscripten, with SIMD + multi-threaded backends and a WebGPU EP (since ORT 1.7) for GPU acceleration ([onnxruntime.ai — Build for web](https://onnxruntime.ai/docs/build/web.html), [Using WebGPU](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)). Non-SIMD / non-threaded builds are being dropped since v1.19 — **SIMD + threads are the assumed baseline** ([onnxruntime-web npm](https://www.npmjs.com/package/onnxruntime-web)).
- **transformers.js** runs Whisper entirely in-browser on top of ONNX Runtime Web, with a WebGPU path and a WASM fallback, streaming from the mic via `MediaRecorder` + chunking ([HF — Transformers.js](https://huggingface.co/docs/transformers.js/index), [xenova/whisper-web](https://github.com/xenova/whisper-web), [blog.rasc.ch](https://blog.rasc.ch/2025/01/transformers-js-speech.html)). This proves the *pattern* — but transformers.js ships a pipeline for Whisper's architecture; it would **not** decode GigaAM's RNN-T out of the box.
- **whisper.cpp WASM** ([ggml.ai/whisper.cpp](https://ggml.ai/whisper.cpp/), [whisper.cpp/examples/whisper.wasm](https://github.com/ggml-org/whisper.cpp/tree/master/examples/whisper.wasm)) documents the real-world constraints, which apply equally to gigastt-in-WASM:

What Path B would require gigastt to build:
1. **A JS/TS reimplementation of `gigastt-core`'s non-ONNX logic** — the 64-bin/FFT=320/hop=160 HTK mel front-end (`features.rs`), the BPE tokenizer (`tokenizer.rs`, 1025 tokens), and the **RNN-T greedy decode loop** (`decode.rs`) that threads decoder+joiner state. ORT Web only runs the three ONNX graphs; the loop that calls encoder → decoder → joiner and manages blank/state is gigastt's own code and must be rewritten in JS. This is the bulk of the work and the main risk.
2. **The three ONNX models served to the browser.** Encoder INT8 is ~225 MB. That is a large client download and a large memory footprint.

#### The hard constraints (from the WASM ASR ecosystem)

- **Model size / memory.** whisper.cpp's WASM demo caps out at the `small` model "inclusive… beyond that memory and performance are unsatisfactory," and notes a base model "~140 MB compressed, 400 MB+ in memory during inference" ([whisper.cpp discussion #533](https://github.com/ggml-org/whisper.cpp/discussions/533)). gigastt's 225 MB INT8 encoder is in the danger zone: a ~225 MB download plus expanded runtime memory, on every visitor, every cold load. **Firefox cannot load files >256 MB — use Chrome** ([ggml.ai/whisper.cpp](https://ggml.ai/whisper.cpp/)). Caching (Cache API / IndexedDB) helps repeat visits but not the first.
- **Cross-origin isolation is mandatory for threads.** WASM multi-threading needs `SharedArrayBuffer`, which every browser gates behind **COOP `same-origin` + COEP `require-corp`** headers ([TestMu — WASM threads](https://www.testmuai.com/learning-hub/wasm-threads-browser-support/), [whisper.cpp examples](https://github.com/ggml-org/whisper.cpp/tree/master/examples/whisper.wasm)). Setting COEP breaks loading of any non-opted-in cross-origin asset — a real deployment headache. (HF Spaces can set these headers; a plain static host may not.)
- **SIMD required, performance modest.** The WASM CPU path needs SIMD; even so, whisper.cpp reports only "x2–x3 real-time for tiny/base on a modern CPU" ([ggml.ai/whisper.cpp](https://ggml.ai/whisper.cpp/)). WebGPU is dramatically faster (one background-removal benchmark: ~550× vs single-thread no-SIMD on an M3 Max — [IMG.LY](https://img.ly/blog/browser-background-removal-using-onnx-runtime-webgpu/)), but WebGPU is still experimental, browser-gated (Chrome/Edge solid, Firefox default mid-2024, Safari catching up — [onnxruntime.ai WebGPU](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)), and the GigaAM graphs may hit unsupported ops, forcing partial CPU fallback.

#### Path B verdict

**Partially realistic, but not pre-traction.** It is technically achievable — the ORT-Web + JS-decode-loop pattern is proven by transformers.js and whisper.cpp — but for gigastt specifically it means **porting the mel front-end, tokenizer, and RNN-T decode loop to JavaScript**, shipping a **225 MB model to every browser**, and managing **COOP/COEP + SIMD/threads + WebGPU-op-coverage** caveats. That is a multi-week effort with real "does GigaAM's RNN-T even decode acceptably in JS at usable speed?" risk, and the privacy win (no server) is the *only* thing it buys over Option 1 — at ~10× the effort.

**Effort/risk:** Path A — not viable (high risk). Path B — feasible but multi-week, medium-high risk, large per-visitor download.

---

## Recommendation

**Pursue Option 1 (hosted mic → transcript demo) first. Defer WASM.**

Reasoning, biased toward the lowest-effort credible win for a pre-traction solo project:

1. **Effort asymmetry is enormous.** Option 1 is ~2–3 days with **zero server code changes** — the WS streaming, rate limiting, session cap, origin allowlist, and metrics already exist; it's a thin frontend + a container deploy of the existing binary. Option 2 (the only viable variant, Path B) is multi-week and requires reimplementing the mel front-end, tokenizer, and RNN-T decode loop in JS.
2. **It directly closes the audited gap.** "Anyone can try Russian transcription in-browser with no install" is satisfied the moment the hosted demo is live.
3. **Audience fit.** Hosting on HF Spaces puts the demo in front of the exact people evaluating Russian ASR, next to where the GigaAM model already lives — free, no card, reuses the existing `Dockerfile`.
4. **The one real cost is the privacy-message tension, and it's manageable** with explicit labeling ("this hosted demo is the one cloud exception; run locally for privacy") plus the already-short-able `--max-session-secs` cap and in-memory-only processing.
5. **WASM's only advantage over the hosted demo is preserving the no-cloud promise in the demo itself** — genuinely on-brand, but not worth ~10× the effort and a 225 MB per-visitor download before the project has traction. Revisit Path B (ORT-Web + JS decode loop) *after* traction, ideally targeting WebGPU and possibly a smaller/distilled encoder to get the download under the practical browser ceiling.

**Suggested sequencing:** ship Option 1 now; file a tracked "WASM in-browser demo (ORT-Web + JS RNN-T decode loop)" item as a post-traction stretch, with the open question recorded: *can GigaAM v3 RNN-T decode acceptably in-browser, and can the 225 MB encoder be shrunk to fit the practical browser download/memory ceiling?*

---

## Sources

WASM / ort / in-browser ASR claims above are drawn from:

- pyke `ort` crate & WASM constraints: [github.com/pykeio/ort](https://github.com/pykeio/ort), [crates.io/crates/ort](https://crates.io/crates/ort), [docs.rs/ort](https://docs.rs/ort), [pykeio/ort DeepWiki — Installation & Setup](https://deepwiki.com/pykeio/ort/2.1-installation-and-setup)
- Rust WASM targets: [rustc book — wasm32-unknown-unknown](https://doc.rust-lang.org/beta/rustc/platform-support/wasm32-unknown-unknown.html), [rustc book — wasm32-unknown-emscripten](https://doc.rust-lang.org/beta/rustc/platform-support/wasm32-unknown-emscripten.html)
- ONNX Runtime Web (SIMD/threads/WebGPU): [onnxruntime.ai — Build for web](https://onnxruntime.ai/docs/build/web.html), [onnxruntime.ai — WebGPU EP](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html), [onnxruntime-web on npm](https://www.npmjs.com/package/onnxruntime-web), [IMG.LY — WebGPU background removal benchmark](https://img.ly/blog/browser-background-removal-using-onnx-runtime-webgpu/)
- transformers.js / Whisper in-browser: [HF — Transformers.js](https://huggingface.co/docs/transformers.js/index), [xenova/whisper-web](https://github.com/xenova/whisper-web), [blog.rasc.ch — Speech recognition with Transformers.js](https://blog.rasc.ch/2025/01/transformers-js-speech.html)
- whisper.cpp WASM (constraints, model size, COOP/COEP, SIMD, perf): [ggml.ai/whisper.cpp](https://ggml.ai/whisper.cpp/), [whisper.cpp/examples/whisper.wasm](https://github.com/ggml-org/whisper.cpp/tree/master/examples/whisper.wasm), [whisper.cpp discussion #533](https://github.com/ggml-org/whisper.cpp/discussions/533)
- WASM threads / SharedArrayBuffer / cross-origin isolation: [TestMu — WASM threads, Atomics, COOP/COEP](https://www.testmuai.com/learning-hub/wasm-threads-browser-support/)
