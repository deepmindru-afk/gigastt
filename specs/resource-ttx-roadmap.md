# Resource TTX — completed program

**Status:** TTX-01…TTX-20 all **done**. Kept as the historical record of the
lean-INT8 / pool / cache work. There is no standing local task queue;
the shipped state is `CHANGELOG.md`.
**Lab evidence:** `specs/research/` (CATALOG / RESULTS / experiments).
**Goal (then):** same accuracy class, less disk and RAM, not slower.

Only items with **confirmed** impact (or a **confirmed gap**) appear as tasks below.  
Research spikes without a ship decision stay in § Optional later / out of backlog.

---

## Backlog (do these)

Status: `todo` | `blocked` | `done`.  
Type: `code` | `docs` | `ops-tooling`.

### P0 — ship first (highest impact)

| ID | Task | Type | Theory | Impact (confirmed) | Status |
|----|------|------|--------|--------------------|--------|
| **TTX-01** | **`ensure` / model presence accepts INT8-only (prequantized) set** — serve/transcribe/download must not require FP32 encoder when INT8+dec+joint+vocab are complete | code | T-026 | Avoids **~844 MB** FP32 download / enables lean install | **done** |
| **TTX-02** | **Document lean install** = `v3_rnnt_encoder_int8.onnx` + decoder + joint + vocab (~220 MB class); offline errors name those files | docs | T-001 | Disk SKU; unblocked by TTX-01 | **done** |
| **TTX-03** | **Edge / low-RAM: default or profile `pool-size 1`** (docs + optional serve profile / auto later) | code+docs | T-009 | **−280…450 MiB** RSS vs pool=2 | **done** (docs+CLI help; default stays 2) |
| **TTX-04** | **Encoder threads: never recommend/default `1` on multi-core**; help text + edge guide (threads=1 only explicit debug) | docs (+ guard if any path forces 1) | T-044 | Avoids RTF **×3…3.65** regression | **done** |
| **TTX-05** | Prefer **`download --prequantized`** (or default lean path) once ensure accepts INT8-only | code+docs | T-001/T-026 | Lean download UX | **done** (default lean; `--fp32` opt-in) |

### P1 — next (disk / RAM / speed productization)

| ID | Task | Type | Theory | Impact (confirmed) | Status |
|----|------|------|--------|--------------------|--------|
| **TTX-06** | **Model-dir GC:** drop non-active `optimized_cache` graphs; keep only current head INT8 optimized | code and/or CLI | T-028/T-037 | Reclaim **~1.3 GiB** on polluted installs | **done** (`gigastt cache-gc`) |
| **TTX-07** | **Content-hash dedupe** (or hardlink) for exact duplicate files under model dir | code/ops-tooling | T-091 | Reclaim **~679 MB** exact dups | **done** (`gigastt cache-gc --dedupe`) |
| **TTX-08** | **Lazy-load speaker encoder** only when diarization requested | code | T-027 | **−~40 MiB** ready when speaker file present but unused | **done** |
| **TTX-09** | **Docs + edge profile: `--vad` for pause-rich** long files (meetings/podcasts) | docs (+ optional profile) | T-048 | RTF up to **×2.6** on silence-rich | **done** (docs; optional `--profile edge` remains TTX-16) |
| **TTX-10** | **Long-form speech-region path** (Silero segments + word merge; fallback fixed chunks) — productize or document client-side stitch | code or docs | T-016/T-045 | Peak **−100…−190 MiB** @64–128 s; J≥0.94 | **done** (`--vad` + empty-region fallback + docs) |
| **TTX-11** | **SKU docs: `ml_ctc` = speed (~1.5× RTF), not lean-RAM** | docs | T-120 | RTF **0.023** vs rnnt **0.034**; ready RSS ≈ rnnt | **done** |
| **TTX-12** | **Docs: pool>1 costs RAM and ~+10–20% single-job RTF** (thread split) | docs | T-009/T-117 | Operators pick concurrency knowingly | **done** |
| **TTX-13** | **Docs: admin reload needs ~+0.5× ready free RAM** (pool=1 ≈ **+536 MiB** peak) | docs | T-054 | Prevent edge OOM on reload | **done** |
| **TTX-14** | **Docs: pool checkout timeout** = queue vs fail-fast **503** + `retry_after_ms` | docs | T-121 | Ops tuning | **done** |
| **TTX-15** | **Docs: `batch_pool_size` splits pool** (no extra idle triplets) | docs | T-085 | Avoid config myth | **done** |
| **TTX-16** | Optional **`--profile edge`** bundling pool=1, sane threads, vad-on-long, optional ml_ctc note | code+docs | T-011 + above | One switch for weak hosts | **done** (`--profile edge` → pool=1 + vad when unset) |

### P2 — worth doing later (confirmed gaps / smaller ROI)

| ID | Task | Type | Theory | Impact | Status |
|----|------|------|--------|--------|--------|
| **TTX-17** | **cgroup `memory.max` pool clamp** (Docker/k8s), not only host RAM | code | T-041 | Avoid OOM under container limits | **done** |
| **TTX-18** | Soft reload without double-resident peak (drop-old-first / file soft-swap) | code | T-054 follow | Zero-headroom reload on edge | **done** (swap-before-warm + `?soft=true` drain) |
| **TTX-19** | Weight-shared pool / PrepackedWeights spike → re-measure pool Δ | code spike | T-002/T-021 | Potential large RAM at pool≥2 | **done** (CPU shared PrepackedWeights; remeasure Δ) |
| **TTX-20** | Docs: punct model ready tax (~+4…28 MiB); edge may leave punct off | docs | T-049 | Small RAM | **done** |

---

## Explicitly not in backlog (confirmed low/no impact)

Do **not** open product work for these without new evidence:

| Finding | Why skip |
|---------|----------|
| INT4 / residual float MatMul quant (T-006) | No const weights; residual ~2 MiB |
| Metrics / rate-limit / OpenAI multipart / hotwords-200 RTF tax | Negligible |
| 48 kHz vs 16 kHz file REST as major RTF lever | +~4% only |
| ORT Sequential vs Parallel on Mac CPU | Parity |
| Input tensor reuse / IOBinding as RTF win | Ratio ~1.0 |
| ONNX metadata strip | ~0 bytes |
| zstd of INT8 as transport (×1.15) | Weak |
| `optimized_cache` as **warm RTF** accelerator | T-116: no effect |
| CLI vs serve for lower single-job HWM | Peak parity |

---

## Suggested implementation order

```text
TTX-01 ensure prequantized     →  TTX-02 lean docs  →  TTX-05 prequantized default
TTX-03 pool=1 edge             →  TTX-04 threads floor docs  →  TTX-12 pool>1 docs
TTX-06 + TTX-07 model dir GC
TTX-08 lazy speaker
TTX-09 vad docs  →  TTX-10 long-form path
TTX-11 ml_ctc SKU docs
TTX-13 reload docs  →  TTX-14/15 ops docs
TTX-16 --profile edge (bundles above)
TTX-17 cgroup (Linux) when packaging containers
```

---

## Evidence snapshot (one-liners)

| Theory | Confirmed result |
|--------|------------------|
| T-026 | INT8-only dir still triggers FP32 ensure/download |
| T-009 | pool 1→2 **+280–450 MiB** |
| T-028 | optimized_cache reclaim **~1273 MiB** |
| T-091 | SHA256 dups reclaim **~679 MB** |
| T-044 | threads=1 RTF **×3–3.65** vs auto |
| T-048 | --vad silence-rich RTF **×~2.6** |
| T-045/T-016 | Silero stitch peak **−115…−192 MiB**, J≥0.94 |
| T-027 | speaker file **+~39 MiB** ready |
| T-054 | reload peak **+536 MiB** |
| T-120 | ml_ctc RTF **0.023** vs rnnt **0.034**; ready ≈ rnnt |
| T-117 | pool=2 serial RTF **~+18%** |

Full tables: `specs/research/RESULTS.md`. Method: `specs/research/METHOD.md`.

---

## Links

| Doc | Role |
|-----|------|
| `specs/research/CATALOG.md` | theory index |
| `specs/research/RESULTS.md` | experiment verdicts |
| `specs/todo.md` | pointer to this backlog |
| `docs/README.md` | user-facing docs index |

## Changelog

| Date | Note |
|------|------|
| 2026-07-27 | Lazy speaker: probe at boot; ONNX on first diarization request |
| 2026-07-27 | Operator docs: VAD, pool/RTF tradeoffs, reload headroom, ml_ctc speed SKU, checkout timeout, batch split, punct tax |
| 2026-07-27 | Lean install docs: minimum INT8+dec+joint+vocab file set (~220 MB); offline errors point at lean paths |
| 2026-07-27 | Download / empty-dir ensure default to lean prequantized INT8; `--fp32` for HuggingFace FP32 + quantize |
| 2026-07-27 | Long-form: document speech-region vs fixed-window paths; empty VAD regions fall back to full/chunked decode |
| 2026-07-27 | Pool RAM clamp reads Linux cgroup `memory.max` / v1 limit (min with host RAM) |
| 2026-07-27 | Soft reload: swap before warm; `?soft=true` waits for old engine drain |
| 2026-07-27 | Weight-share spike: CPU production factory attaches shared ORT PrepackedWeights |
| 2026-07-27 | `--profile edge` / `GIGASTT_PROFILE=edge`: default pool-size 1 + VAD when those flags are unset |
| 2026-07-27 | ensure accepts INT8-only (prequantized) install: `is_usable_present` = FP32 download set OR prequantized INT8 set; serve/transcribe no longer re-fetch FP32 when lean tree is complete |
| 2026-07-27 | Model-dir hygiene: `gigastt cache-gc` prunes non-active `optimized_cache/*_optimized.onnx`; `--dedupe` hardlinks content-identical files |
| 2026-07-27 | Edge pool=1 + encoder threads: keep `--pool-size` default 2; CLI help + docs recommend `--pool-size 1` for edge; document pool>1 RAM + ~10–20% single-job RTF; warn `--encoder-intra-threads 1` is ~3× slower (explicit `1` still passes through) |
| 2026-07-27 | Reworked as **actionable backlog** (TTX-01…TTX-20): only confirmed-impact work to do; evidence demoted to snapshot |
| 2026-07-26…27 | Lab R0–R19 filled confirmations (see research RESULTS) |
