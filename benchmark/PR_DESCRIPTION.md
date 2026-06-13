# fix(benchmark): apply symmetric text normalization before WER computation

## Summary

This PR fixes a methodology bug in the Python cross-ASR benchmark: WER was previously computed on raw reference and hypothesis strings, which unfairly penalized engines for benign differences such as Arabic digits vs. Russian number words, Latin brand names vs. Cyrillic transliterations, and punctuation/whitespace choices.

The benchmark now applies a single, symmetric normalization pipeline to **both** the reference and the hypothesis for every engine before computing WER. The pipeline includes:

- Lowercasing and `ё` → `е` normalization.
- Tokenizing letters, digit sequences, and symbols as separate tokens.
- Converting Russian number-word sequences (including ordinals, compounds, and scale words) into Arabic digits.
- Merging adjacent short digit groups (≤ 3 digits each) so phone numbers and chunked digit strings align.
- Dropping currency/symbol artifacts and wake words (`плюс`, `минус`, `номер`, `процент`, `$`, `%`, etc.).
- Mapping common anglicisms (e.g. `youtube` → `ютуб`).

Empty/whitespace-only references are now filtered at manifest load time and reported as `skipped_empty_refs`.

## Key results (before → after)

| Dataset | Engine | Old WER | New WER |
|---|---|---|---|
| `golos_crowd_1k` | gigastt | **10.77%** | **8.60%** |
| `golos_crowd_1k` | faster-whisper | 15.54% | 15.53% |
| `golos_crowd_1k` | whisper.cpp | 15.80% | 15.26% |
| `golos_crowd_1k` | vosk | 4.57% | 4.82% |

The gigastt crowd improvement is the largest: **–2.17 pp**, driven mainly by numbers and brand/artist names now being compared on equal footing. Other datasets are essentially unchanged because their references already contain little digit or foreign-name variance.

## Test command and result

```bash
cd benchmark
.venv/bin/python -m pytest tests/test_common.py -v
```

```
23 passed in 0.04s
```

## Residual errors

After renormalization, the remaining WER on `golos_crowd_1k` for gigastt is **8.60%** (95% CI 7.51%–9.66%). The dominant residual error source is no longer numbers but **foreign brand/artist names in Latin spelling vs. Russian transliteration**, e.g.:

- Ref: `включи гуд лайф джи изи и кехлани`  
  Hyp: `включи good life g eazy i kehlani`
- Ref: `киношка окко смарт бокс на окко`  
  Hyp: `киношка okko смартбокс на okko`

A full top-10 residual error sample list is available in `benchmark/results_full/residual_errors_gigastt_crowd.md`.

## Files changed

- **Benchmark harness**: `benchmark/benchmark.py`, `benchmark/common.py`, `benchmark/recompute_wer.py`, `benchmark/monitor.sh`, `benchmark/run_full_suite.sh`
- **Runners**: `benchmark/runners/gigastt.py`, `benchmark/runners/whisper_cpp.py`
- **Tests**: `benchmark/tests/test_common.py` (23 new normalization unit tests)
- **Manifests**: `benchmark/manifests/*.json` (4 datasets)
- **Results**: `benchmark/results_full/*.json` + `*_renorm.json` (full 4×4 matrix) and `benchmark/results_full/renorm_summary.md`
- **Docs**: `benchmark/README.md`, `docs/benchmark_narrative_draft.md`, `README.md`, `README_RU.md`
- **CI**: `.github/workflows/benchmark.yml`
- **Dataset prep scripts**: `scripts/extract_golos.py`, `scripts/extract_golos_farfield.py`, `scripts/prepare_common_voice_ru.py`, `scripts/prepare_openstt_calls.py`, `scripts/prepare_openstt_youtube.py`

> No commits have been pushed to origin; this PR description is generated from the local `fix/benchmark-methodology` branch.
