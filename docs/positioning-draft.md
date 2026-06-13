# Positioning — DRAFT (for review; NOT yet in README)

**Status:** draft · **Date:** 2026-06-13 · Strategic positioning (audit findings #29/#31/#42).

Task 02 already removed the *false/unmeasured* claims from the README. This doc is the
next step the audit flagged but no task closed: **forward positioning** — what gigastt
should actively claim, who it's for, and how it stands against the specific competitors
a knowledgeable Russian-ASR engineer reaches for. Pre-traction (≈10★, ≈500 downloads),
so every skeptical reader matters; the goal is a niche that is *true today* and survives
a competitor winning any single WER column.

---

## 1. The problem with current positioning

- **Too many personas.** The README addresses ~5 broad audiences (voice assistants,
  transcription apps, call-center, accessibility, embedded). At a narrow FFI/embedded
  niche with one model, that breadth reads as unfocused and invites comparison on axes
  gigastt loses (clean-read WER vs Vosk, multilingual vs Whisper/Parakeet).
- **No stance vs the closest competitors.** The comparison table lists
  whisper.cpp / faster-whisper / Vosk / sherpa / Cloud — but **not** the three projects
  that actually overlap gigastt's pitch: `onnx-asr`, `transcribe-rs`, and **T-one**.
  Silence there looks like avoidance.

---

## 2. The niche (the honest one-liner)

> **Embeddable on-device Russian STT for Rust and mobile apps** — one static binary or
> one crate, no Python, no cloud, on the SOTA open Russian model (GigaAM v3), with a
> hardened streaming server and a C-ABI FFI.

Everything in the README should ladder up to *that*. Drop the broad-assistant framing
as the headline; keep concrete use-cases as examples *under* the niche.

**Lead axes (true today, see `benchmark_narrative_draft.md` Thesis B):**
single static binary (~225 MB) · true incremental WebSocket streaming · C-ABI FFI for
Android/mobile · hardened server (rate-limit / origin allowlist / graceful drain /
metrics) · MIT-clean **code *and* weights** · accuracy leadership on far-field /
telephony / spontaneous (3 of 4 domains, pending re-run vs Vosk 0.54 + T-one).

**Axes to stop competing on as headline:** clean-read WER (Vosk wins), multilingual
(Whisper/Parakeet win), "only Rust-native" (false — see transcribe-rs), raw latency
numbers (TTFP is ~0.7 s, not sub-200 ms).

---

## 3. Head-to-head positioning (concede honestly, differentiate sharply)

### vs `onnx-asr` (istupakov, ~331★) — the strongest adjacent competitor
- **What it is:** a pip-installable Python *library*; GigaAM v2/v3 (CTC+RNNT), many EPs
  (CUDA/TensorRT/CoreML/DirectML/ROCm/WebGPU), Win/Linux/macOS x86+Arm, no protoc.
- **Concede:** for a Python pipeline or notebook, `onnx-asr` is the easier, broader,
  more-backends choice. Say so.
- **Differentiate:** gigastt is a **deployable hardened server + an embeddable native
  binary/FFI**, not a library you wire up. No Python runtime; one artifact; production
  ops (backpressure, drain, rate-limit, metrics) built in. *gigastt is what you ship;
  onnx-asr is what you prototype with.*
- **Action:** add `onnx-asr` to the comparison and to the acknowledgments framing —
  ceding Python pipelines explicitly is more credible than omitting it.

### vs `transcribe-rs` (MIT, ~208★) — kills the "only Rust-native" claim
- **What it is:** Rust-native GigaAM v3 with CoreML/CUDA/int8 *today*.
- **Concede:** gigastt is **not** the only Rust-native GigaAM v3 option. Remove that
  claim everywhere (task 02 dropped "only Rust-native"; keep it gone).
- **Differentiate:** transcribe-rs is a transcription *library/CLI*; gigastt is a
  **server + FFI + mobile** story (WebSocket streaming, Android cdylib, hardened HTTP).
  Compete on the *server/embedded* surface, not on "Rust-native" as a bare fact.

### vs **T-one** (T-Bank, Apache-2.0) — the telephony specialist
- **What it is:** purpose-built streaming CTC-Conformer (300 ms chunks), Apache on code
  *and* weights, production-hardened in a bank; strong on call-center / named entities.
- **Concede:** for pure Russian telephony streaming, T-one is purpose-built and may beat
  gigastt on calls (pending the head-to-head run; published T-one numbers are vs
  GigaAM-RNNT-**v2**, not v3 — so this is genuinely open).
- **Differentiate:** gigastt is **general-domain + embeddable** (far-field, video, calls
  on one model) with a single-binary/FFI deployment; T-one is telephony-focused. Position
  as *the embeddable generalist*, and let the re-run decide the telephony column honestly.

### vs Whisper (whisper.cpp / faster-whisper)
- Already in the table. Keep: gigastt is far faster (real-time CPU vs >1.0 RTF for
  faster-whisper) and ~13× smaller, and leads RU WER on every measured domain. Whisper's
  edge is multilingual — out of gigastt's niche.

### vs Vosk
- Concede clean-read accuracy (4.82 vs 8.60) and raw RTF. Win on far-field/telephony/
  YouTube accuracy + ~6× smaller footprint + streaming partials. (pending Vosk 0.54.)

---

## 4. Persona narrowing (from ~5 broad → 2 sharp)

**Primary:** Rust / mobile developers embedding offline Russian STT — one crate or one
`cdylib`, no Python, no cloud (privacy/compliance), small footprint.

**Secondary:** self-hosters who want a private, hardened Russian STT *server* (WebSocket
+ REST + SSE) they can run on a loopback or a VPS without sending audio to a cloud API.

Everything else (general voice assistants, accessibility, etc.) becomes an *example* of
those two, not a separate headline persona.

---

## 5. Concrete next edits (after review, separate task)

1. README headline → the niche one-liner (§2); demote the broad-assistant framing.
2. Add `onnx-asr`, `transcribe-rs`, `T-one` to the competitor table / prose with the
   honest concede/differentiate lines above.
3. Collapse the persona list to the two in §4.
4. Keep the latency framing honest (~0.7 s TTFP; no sub-200 ms).
5. Cross-check against `benchmark_narrative_draft.md` so positioning and numbers agree.

> This is messaging strategy, not measured fact — it must be reconciled with the full
> re-run numbers (Vosk 0.54 + T-one) before any of it lands in the README.
