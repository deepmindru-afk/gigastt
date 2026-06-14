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
| faster-whisper-turbo ¹ | Large v3 turbo | 14.45 (11.52–18.0) | 18.30 (16.69–20.03) | 29.22 (26.02–32.76) | 15.16 (12.82–17.45) |
| **T-one (beam+LM)** | t-tech/T-one | 6.61 (5.43–7.91) | 14.62 (12.48–16.97) | 21.73 (19.96–23.68) | 23.23 (21.46–25.11) |
| T-one (greedy, no LM) | t-tech/T-one | 7.85 (6.67–9.24) | 17.22 (15.02–19.58) | 22.37 (20.58–24.22) | 26.54 (24.66–28.48) |

¹ turbo is a **300-sample** directional slice (CPU-slow; wider CIs); all others are 1000.
T-one's **beam+LM** is its production config (the 5.5 GB KenLM, fetched manually);
**greedy** is the no-LM fallback. Both shown for honesty.

**Reading (modern Vosk 0.54 now measured — 1000 samples/domain, RTF ~0.04, 0 fail):**
Against the *outdated* Vosk 0.42, gigastt led 3 of 4 domains. Against **modern Vosk
0.54** the honest picture is narrower:
- **Clean read:** Vosk 0.54 **2.97%** — best of all, beats 0.42 (4.82) and gigastt (8.60).
- **Far-field:** now a **statistical tie** — gigastt 5.90 (5.09–6.83) vs Vosk 0.54 6.29
  (5.43–7.28), CIs overlap. gigastt's old far-field "lead" was against 0.42's 13.93; it
  does **not** hold against 0.54.
- **Phone calls:** **gigastt holds** — 19.28 (17.88–20.67) beats Vosk 0.54 22.74
  (21.28–24.17, separated) and ties/leads even **T-one's production beam+LM** 21.73
  (19.96–23.68; CIs barely overlap, gigastt point estimate ahead). ⚠️ Caveat:
  `openstt_calls` very likely overlaps GigaAM v3's training (contamination flatters
  gigastt), and T-one's *published* telephony strength is on its own call-center set,
  not this slice.
- **YouTube:** **gigastt wins clearly** — 11.35 (10.32–12.31) vs Vosk 17.24 and T-one
  beam 23.23, all separated.

**T-one (telephony specialist):** its production **beam+LM** (5.5 GB KenLM, fetched
manually since the HF download hangs) lifts it over greedy on every domain — but it
still loses to gigastt on far-field (14.62 vs 5.90) and YouTube (23.23 vs 11.35), ties
on phone, and loses clean read to Vosk 0.54. It is *not* general-domain; the honest
read is that **full beam+LM T-one did not take gigastt's phone column** (only tied it).

So even against **modern Vosk 0.54 *and* full beam+LM T-one**, gigastt's defensible
accuracy claim is: **1 clear win (YouTube), phone held (tie/ahead, with the
contamination caveat), far-field tied (Vosk 0.54), clean read lost (Vosk 0.54)**. The
durable story is **Thesis B** — embeddable single-binary / FFI / streaming / ~225 MB —
not a WER crown: Vosk 0.54 beats gigastt on RTF (~0.04 vs ~0.16) and T-one matches it,
but both are larger and neither ships an embeddable Rust/FFI streaming server.

> ✅ Resolved: **T-one** (greedy *and* production beam+LM — the latter via a
> manually-curl'd 5.5 GB KenLM, since the HF download hangs) and **faster-whisper-turbo**
> (300-sample slice) are now measured. T-one did **not** take the phone column (tie).
> All 7 engines have WER numbers; only the non-WER axes (footprint, punctuation,
> hallucinations) remain.

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
| T-one (beam+LM) | 0.056 | 0.060 | 0.065 | 0.065 |
| faster-whisper-turbo | 1.811 | 1.974 | 2.047 | 1.815 |

**Reading:** the CTC/transducer engines are all fast — Vosk (~0.03–0.04), **T-one
beam+LM ~0.06**, gigastt ~0.16–0.21 — and modern **Vosk 0.54 is fast *and* accurate**,
so the old "Vosk trades accuracy for speed" line no longer holds. gigastt is comfortably
real-time but **not the fastest** (Vosk and T-one are quicker); its speed edge is only
over the Whisper engines, which run **slower than real-time** on CPU (faster-whisper
>1.0 RTF, turbo ~1.8–2.0).

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

**Thesis A — "Best local accuracy on *spontaneous/telephony* Russian speech" (now fully measured vs Vosk 0.54 + T-one).**
gigastt's only **CI-separated** accuracy win is **YouTube** (11.35 vs all). **Phone is
held** — gigastt 19.28 ties/leads even T-one's production beam+LM (21.73) and beats Vosk
0.54 (22.74) — but with a **contamination caveat** (`openstt_calls` likely in GigaAM v3's
training). Far-field is a **tie** (Vosk 0.54), clean read a **loss** (Vosk 0.54). So the
honest accuracy claim is narrow: *the smallest engine that wins YouTube and holds noisy
phone calls* — not "best/hard speech in general", not far-field, not clean read.

**Thesis B — "Streaming + footprint + Rust-native embeddability" (robust regardless of WER).**
Frame on the axes that don't depend on winning WER: true incremental streaming (vs
Whisper's none), ~225 MB single static binary, C-ABI FFI for Android/mobile, hardened
server (rate-limit / origin allowlist / graceful drain), MIT-clean code *and* weights.
This is the defensible niche from the competitive analysis and survives even if a
competitor wins a WER column.

**Recommendation:** lead with **B** (unbreakable). The re-run is done, so the only
A-claim worth making is *per domain*: "wins YouTube; holds noisy phone calls (with the
data caveat)" — and concede clean read (Vosk 0.54) and tie far-field plainly. Do **not**
claim general accuracy leadership.

---

## What's still pending (blocks publishing to README)

1. ✅ **All 7 engines measured.** Vosk 0.54 (sherpa-onnx), faster-whisper-turbo
   (300-sample slice) and T-one (greedy *and* production beam+LM, the latter via a
   manually-curl'd 5.5 GB KenLM) are done — full WER + RTF tables above.
2. `benchmark_footprint.py` + `benchmark_punctuation.py` + `benchmark_hallucinations.py`
   actual numbers (smoke first, ≤50 samples).
3. Vosk-server streaming latency for a fair streaming comparison.
4. Only then: update the README cross-domain table + write the final, single
   positioning paragraph.
