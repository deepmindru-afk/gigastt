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
| Vosk 0.54 | vosk-model-ru-0.54 | `pending` | `pending` | `pending` | `pending` |
| faster-whisper-turbo | Large v3 turbo | `pending` | `pending` | `pending` | `pending` |
| T-one | voicekit-team/T-one | `pending` | `pending` | `pending` | `pending` |

**Reading of the existing four:** gigastt leads on **3 of 4** domains — far-field,
phone, and YouTube — by a wide margin (far-field 5.90 vs Vosk's 13.93; phone 19.28
vs 38.57; YouTube 11.35 vs 20.65). **Vosk 0.42 wins clean read** (4.82 vs 8.60) and
this should be stated plainly. The Whisper engines trail on every domain here.

> Open question the new runners answer: **Vosk 0.54** (Zipformer2) is the real
> modern Vosk and is reported ≈6% on Common Voice ru — it may narrow or change the
> clean-read story. **T-one** is purpose-built for telephony and is the engine most
> likely to contest gigastt's phone-call lead. Both are why this table must be
> re-run before any "we lead" claim.

---

## 2. Speed (RTF, lower = faster; M1 CPU)

RTF = processing time / audio duration, per domain.

| Engine | Clean read | Far-field | Phone calls | YouTube |
|---|---|---|---|---|
| Vosk 0.42 | **0.035** | **0.029** | **0.029** | **0.029** |
| gigastt | 0.157 | 0.164 | 0.212 | 0.158 |
| whisper.cpp | 0.357 | 0.556 | 0.624 | 0.765 |
| faster-whisper | 1.187 | 1.604 | 2.312 | 1.879 |
| Vosk 0.54 / turbo / T-one | `pending` | `pending` | `pending` | `pending` |

**Reading:** Vosk is the fastest by far (~0.03) but pays in accuracy on hard audio.
gigastt is the fastest *accurate* engine (~0.16–0.21, comfortably real-time on CPU).
faster-whisper is **slower than real-time** on CPU (>1.0 RTF) across all domains.

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

**Thesis A — "Best local accuracy on *hard* Russian speech" (data currently supports this).**
On the committed matrix gigastt already leads far-field, phone, and YouTube at ~6×
smaller footprint and real-time CPU speed. Frame: *the smallest engine that wins on
the speech that actually matters in production — far-field commands, noisy phone
calls, spontaneous video — conceding only clean studio read-speech to Vosk.* Risk:
T-one may take telephony; Vosk 0.54 may shift the picture. Only claim after the re-run.

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

1. Full run of **Vosk 0.54**, **faster-whisper-turbo**, **T-one** on all 4 domains
   (WER + RTF). Requires: confirm `vosk-model-ru-0.54` id on alphacephei; `pip install
   torch transformers` + confirm T-one's exact HF/processor API; ~GBs of downloads;
   a several-hour run → **needs an agreed time window**.
2. `benchmark_footprint.py` + `benchmark_punctuation.py` + `benchmark_hallucinations.py`
   actual numbers (smoke first, ≤50 samples).
3. Vosk-server streaming latency for a fair streaming comparison.
4. Only then: update the README cross-domain table + write the final, single
   positioning paragraph.
