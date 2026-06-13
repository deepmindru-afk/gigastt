# Draft: gigastt Performance Narrative (fix/benchmark-methodology)

> This draft is written before the full 4×4 cross-domain benchmark finishes.
> It will be finalized once `benchmark/run_full_suite.sh` completes and the
> actual WER/RTF numbers are inserted into README.md / README_RU.md.

## Headline options

**Option A — if data confirm gigastt leads on spontaneous/far-field:**
> On-device Russian speech recognition with the best local WER on
> spontaneous and far-field speech.

**Option B — if data show mixed leadership:**
> Rust-native, real-time Russian speech recognition with sub-200 ms latency,
> hardware acceleration, and a 6× smaller model footprint than Whisper/Vosk.

## Key claims to support with numbers

1. **No single engine wins every domain.** Vosk’s 1.3 GB Kaldi model with a
   strong LM dominates clean read speech (Golos crowd). gigastt’s 230 MB
   GigaAM v3 is competitive there and leads on far-field commands.
2. **gigastt is the strongest local option for streaming/far-field use-cases**
   because it is the only Rust-native engine with real-time WebSocket,
   CoreML/CUDA/NNAPI acceleration, and INT8 quantization.
3. **Whisper-based engines catch up on spontaneous/noisy domains** but are
   3–4× slower and 12× larger than gigastt.

## Proposed README Performance section

```markdown
## Performance

| Metric | Value |
|---|---|
| **WER (Russian, clean read)** | TBD% (1 000 Golos crowd samples, TBD words, 95% CI) |
| **WER (Russian, far-field)** | TBD% (1 000 Golos farfield samples, TBD words, 95% CI) |
| **WER (Russian, phone calls)** | TBD% (1 000 OpenSTT calls samples, TBD words, 95% CI) |
| **WER (Russian, YouTube)** | TBD% (1 000 OpenSTT YouTube samples, TBD words, 95% CI) |
| **INT8 vs FP32** | 0% WER degradation (verified on Golos crowd) |
| **Latency (16s audio, M1)** | ~700 ms |
| **Memory (RSS)** | ~560 MB |
| **Model size** | 851 MB (FP32) / 222 MB (INT8) |

### Cross-ASR by domain (1 000-sample slices, CPU)

| Engine | Clean read | Far-field | Phone calls | YouTube | RTF | Size |
|---|---|---|---|---|---|---|
| Vosk | TBD% | TBD% | TBD% | TBD% | TBDx | 1.3 GB |
| gigastt | TBD% | TBD% | TBD% | TBD% | TBDx | 230 MB |
| whisper.cpp | TBD% | TBD% | TBD% | TBD% | TBDx | ~3 GB |
| faster-whisper | TBD% | TBD% | TBD% | TBD% | TBDx | ~3 GB |

> **Take-away:** Vosk remains the accuracy king on clean studio speech, but
> gigastt is the best balance of accuracy, speed, size, and streaming
> capability for real-world Russian ASR.
```

## Methodology summary to keep

- Pre-warmed engines for RTF.
- Failures counted as 100% WER.
- Bootstrap 95% CI.
- Decode params documented.
- Dataset licenses: Sber Public License (Golos), CC BY-NC 4.0 (OpenSTT).

## Next step

After `run_full_suite.sh` finishes, replace all `TBD` placeholders above,
update README.md / README_RU.md, and choose headline Option A or B based on
which claim the data actually support.
