# Benchmark Narrative — DRAFT (for review; NOT yet in README)

**Status:** draft · **Date:** 2026-06-13 · Part D of the benchmark-expansion design.

This is the review stop-point before any README/README_RU change. Numbers for the
**existing four engines** come from the committed 1000-sample renormalized matrix
(`benchmark/results_full/renorm_summary.md` for WER + 95% CI;
`benchmark/results_full/detailed_analysis.md` for per-domain RTF). Numbers marked
**`pending`** require a full run of the newly-added runners (Vosk 0.54,
faster-whisper-turbo, T-one) and of the new axes (punctuation, hallucinations,
footprint) — see "What's still pending" at the end.

All measurements were taken on an Apple **M1**, CPU execution provider, INT8 where
applicable. WER is computed with failures counted as 100% WER and ITN
renormalization applied; 95% bootstrap CIs (1000 iterations).

---

## 1. Accuracy by domain (WER %, 95% CI)

Domains: **Clean read** = `golos_crowd_1k` · **Far-field** = `golos_farfield` ·
**Phone** = `openstt_calls` · **YouTube** = `openstt_youtube`.

| Engine | Model | Clean read | Far-field | Phone calls | YouTube |
|---|---|---|---|---|---|
| **gigastt** | GigaAM v3 (INT8) | 8.60 (7.51–9.66) | **5.90 (5.09–6.83)** | **19.28 (17.88–20.67)** | **11.35 (10.32–12.31)** |
| Vosk 0.42 | vosk-model-ru-0.42 | **4.82 (4.03–5.60)** | 13.93 (12.49–15.47) | 38.57 (36.72–40.64) | 20.65 (19.38–21.98) |
| whisper.cpp | Large v3 | 15.26 (13.74–16.71) | 17.91 (16.29–19.57) | 32.73 (30.69–34.91) | 22.61 (20.97–24.20) |
| faster-whisper | Large v3 (INT8) | 15.53 (13.94–17.10) | 17.34 (15.62–19.07) | 24.93 (23.32–26.57) | 15.45 (14.15–16.62) |
| Vosk 0.54 | vosk-model-ru-0.54 | **2.97 (2.41–3.56)** | 6.29 (5.43–7.28) | 22.74 (21.28–24.17) | 17.24 (16.0–18.43) |
| faster-whisper-turbo | Large v3 turbo | `pending` | `pending` | `pending` | `pending` |
| T-one | t-tech/T-one | `pending` | `pending` | `pending` | `pending` |

**Reading (modern Vosk 0.54 now measured — 1000 samples/domain, RTF ~0.04, 0 fail):**
Against the *outdated* Vosk 0.42, gigastt led 3 of 4 domains. Against **modern Vosk
0.54** the honest picture is narrower:
- **Clean read:** Vosk 0.54 **2.97%** — best of all, beats 0.42 (4.82) and gigastt (8.60).
- **Far-field:** now a **statistical tie** — gigastt 5.90 (5.09–6.83) vs Vosk 0.54 6.29
  (5.43–7.28), CIs overlap. gigastt's old far-field "lead" was against 0.42's 13.93; it
  does **not** hold against 0.54.
- **Phone calls:** **gigastt wins** — 19.28 (17.88–20.67) vs 22.74 (21.28–24.17), CIs
  separated.
- **YouTube:** **gigastt wins** — 11.35 (10.32–12.31) vs 17.24 (16.0–18.43), separated.

So gigastt's defensible accuracy claim shrinks to **2 clear wins (phone + YouTube),
1 tie (far-field), 1 loss (clean read)** — spontaneous/telephony speech, not "hard
speech in general". Vosk 0.54 also beats gigastt on RTF (~0.04 vs ~0.16) but stays
~6× larger and offers no embeddable single-binary / FFI / streaming story.

> Still open: **T-one** (`t-tech/T-one`; runner fixed, blocked on an HF download) and
> **faster-whisper-turbo**, to be added once their weights finish downloading (need an
> `HF_TOKEN`). T-one is purpose-built for telephony and is the most likely to contest
> the phone-call win, so the "phone win" claim stays provisional until T-one is measured.

---

## 2. Speed (RTF, lower = faster; M1 CPU)

RTF = processing time / audio duration, per domain.

| Engine | Clean read | Far-field | Phone calls | YouTube |
|---|---|---|---|---|
| Vosk 0.42 | **0.035** | **0.029** | **0.029** | **0.029** |
| gigastt | 0.157 | 0.164 | 0.212 | 0.158 |
| whisper.cpp | 0.357 | 0.556 | 0.624 | 0.765 |
| faster-whisper | 1.187 | 1.604 | 2.312 | 1.879 |
| Vosk 0.54 | **0.043** | **0.042** | **0.042** | **0.042** |
| turbo / T-one | `pending` | `pending` | `pending` | `pending` |

**Reading:** both Vosk versions are the fastest (~0.03–0.04). Modern **Vosk 0.54 is
fast *and* accurate** (~0.04 RTF, and it wins/ties gigastt on 2 of 4 domains), so the
old "Vosk trades accuracy for speed" line no longer holds against 0.54. gigastt is
still comfortably real-time (~0.16–0.21) and far faster than faster-whisper (>1.0 RTF,
**slower than real-time** on CPU) — but gigastt's speed edge is over the Whisper
engines, not over Vosk.

---

## 3. Streaming latency (true incremental streaming)

Only gigastt exposes genuine token-by-token WebSocket streaming. Measured on
`golos_00.wav` (4.0 s), audio fed in real time, timer from connection.

| Engine | TTFP (time-to-first-partial) | Finalization lag | Streaming? |
|---|---|---|---|
| gigastt (CPU) | ~782 ms | ~3449 ms | yes (`/v1/ws`) |
| gigastt (CoreML) | ~693 ms | ~4041 ms | yes |
| whisper.cpp | — | — | no true streaming (chunked offline) |
| faster-whisper | — | — | no true streaming |
| Vosk (server) | `pending` | `pending` | yes (vosk-server) |

**Honest framing (do NOT revert to "sub-200ms"):** gigastt's TTFP is **~0.7–0.8 s**,
not sub-200ms. The decode is a sliding-window re-decode over an offline RNN-T, so
"streaming" means *buffered/chunked over an offline model*, not a natively streaming
acoustic model. The architectural win to claim is *"true incremental partials from a
single embedded binary"* vs Whisper's no-streaming — not a latency number gigastt
doesn't hit. Vosk-server and T-one (300 ms chunks) are genuine streaming designs and
must be measured before any latency-leadership claim.

---

## 4. Footprint

| Engine | Model on disk | Peak RSS (cold) | Cold-start |
|---|---|---|---|
| gigastt | **225 MB** (INT8; 851 MB FP32) | `pending` | `pending` |
| Vosk 0.42 | ~1.3 GB | `pending` | `pending` |
| faster-whisper | Large v3 ≈1.5 GB (INT8) | `pending` | `pending` |
| whisper.cpp | Large v3 ≈3 GB | `pending` | `pending` |
| T-one | `pending` | `pending` | `pending` |

`benchmark_footprint.py` fills peak RSS + cold-start. The on-disk number is gigastt's
strongest single stat: **~6× smaller than Vosk, ~13× smaller than whisper.cpp**, which
is what makes it embeddable.

---

## 5. New axes (structure only — `pending` numbers)

- **Punctuation / capitalization F1** (`benchmark_punctuation.py`, Common Voice ru):
  measures whether an engine restores `.,!?` and capitalization. ⚠️ Caveat to verify:
  GigaAM v3 e2e_rnnt likely emits **lowercase, unpunctuated** text — so gigastt may
  score *low* here, same as Vosk. Report honestly; this is a column where the Whisper
  engines probably win. Do not spin it.
- **Hallucinations on non-speech** (`benchmark_hallucinations.py`, MUSAN noise/music):
  inserted words per minute on audio with no speech. Whisper models are known to
  hallucinate on silence; this is a plausible gigastt/Vosk advantage. `pending`.

---

## 6. Two positioning theses (pick after the full re-run)

**Thesis A — "Best local accuracy on *spontaneous/telephony* Russian speech" (narrowed by the Vosk 0.54 data).**
Against **modern Vosk 0.54**, gigastt's accuracy wins shrink to **phone calls and
YouTube** (CI-separated); far-field is a **tie** and clean read is a loss. So the honest
claim is *the smallest engine that wins on noisy phone calls and spontaneous video* —
NOT "hard speech in general", and NOT far-field anymore. Caveat: **T-one** (telephony
specialist) is the most likely to take the phone-call column, so even this narrowed
claim is provisional until T-one is measured.

**Thesis B — "Streaming + footprint + Rust-native embeddability" (robust regardless of WER).**
Frame on the axes that don't depend on winning WER: true incremental streaming (vs
Whisper's none), ~225 MB single static binary, C-ABI FFI for Android/mobile, hardened
server (rate-limit / origin allowlist / graceful drain), MIT-clean code *and* weights.
This is the defensible niche from the competitive analysis and survives even if a
competitor wins a WER column.

**Recommendation:** lead with **B** (it's true today and unbreakable), and add the
specific A claims *per domain* once the re-run confirms them against Vosk 0.54 + T-one.

---

## What's still pending (blocks publishing to README)

1. ✅ **Vosk 0.54 done** (all 4 domains, 1000 samples — numbers above, via sherpa-onnx).
   Still pending: **faster-whisper-turbo** + **T-one** (`t-tech/T-one`, runner fixed) —
   both blocked on stalled unauthenticated HuggingFace downloads; need an `HF_TOKEN`
   (or a manual model fetch) to finish.
2. `benchmark_footprint.py` + `benchmark_punctuation.py` + `benchmark_hallucinations.py`
   actual numbers (smoke first, ≤50 samples).
3. Vosk-server streaming latency for a fair streaming comparison.
4. Only then: update the README cross-domain table + write the final, single
   positioning paragraph.
