# gigastt Cross-ASR Benchmark

Reproducible benchmark comparing **gigastt** against popular open-source ASR engines on Russian speech.

## Supported Engines

| Engine | Backend | Language | Installation |
|--------|---------|----------|--------------|
| gigastt | ONNX Runtime / Rust | Russian | Built from source or `cargo install` |
| whisper.cpp | GGML / C++ | Multilingual | Auto-downloaded on first run |
| faster-whisper | CTranslate2 / Python | Multilingual | `pip install faster-whisper` |
| Vosk | Kaldi / C++ | Russian | `pip install vosk` (model auto-downloaded) |

## Metrics

- **WER** (Word Error Rate) — lower is better. Computed with the same text-normalization pipeline across all engines (lowercase, ё→е, digit-to-words, anglicisms, punctuation stripped).
- **RTF** (Real-Time Factor) — `processing_time / audio_duration`. Lower is better; < 1.0 means faster than real-time.

## Methodology

### Timing

RTF is measured against a **pre-warmed engine** so that model-load time is not unfairly charged to any runner:

- **gigastt** is measured via HTTP calls to a `gigastt serve` process that stays up for the whole benchmark.
- **faster-whisper** and **Vosk** load their models once in `is_available()` and reuse them for every sample.
- **whisper.cpp** runs in **server mode** (`whisper-server`). The model is loaded once when the server starts; each sample is sent as an HTTP POST to `/inference` and the wall-clock request latency is used as `processing_time`. This replaces the previous per-sample `whisper-cli` invocation that re-loaded the ~3 GB model on every file and produced an artificially high RTF.

WER is unchanged by this switch: whisper.cpp still uses the same `large-v3` Russian model and the same text normalization pipeline as the other engines.

### Decode parameters

The following decode parameters are used so readers can reproduce the comparison exactly:

| Engine | Parameter | Value | Notes |
|---|---|---|---|
| gigastt | greedy beam search | beam width 1 | RNN-T greedy decode via ONNX Runtime |
| whisper.cpp | default CLI/server defaults | — | temperature 0, prompt none, language `ru` |
| faster-whisper | `beam_size` | 5 | CTranslate2, `language="ru"`, `compute_type="int8"` |
| Vosk | default Kaldi graph | — | `SetWords(False)`, 16 kHz mono 16-bit input |

### Failure handling

If a runner crashes or fails on a sample, that sample is counted as a 100% WER deletion of the reference (all reference words marked as errors). The per-runner `failures` counter and the top-level `total_failures` field in `results.json` make these cases visible instead of silently dropping them from the denominator.

### Confidence intervals

WER is reported with a bootstrap 95% confidence interval computed by resampling per-sample `(ref_words, errors)` pairs with replacement 1 000 times and taking the 2.5th and 97.5th percentiles. This mirrors the Rust CI implementation in `crates/gigastt/tests/benchmark.rs`.

## Quick Start

```bash
cd benchmark
pip install -r requirements.lock.txt

# Run on 100 samples (default)
python benchmark.py

# Run on full Golos crowd dataset (slow!)
python benchmark.py --max-samples 0 --output results_full.json

# Run only specific engines
python benchmark.py --runners gigastt,whisper_cpp

# Use environment variable for limit
GIGASTT_BENCHMARK_MAX_SAMPLES=50 python benchmark.py
```

### Lockfile

`requirements.lock.txt` pins the full transitive dependency tree used by CI.
Regenerate it from `requirements.txt` with [uv](https://docs.astral.sh/uv/):

```bash
uv pip compile requirements.txt \
  --python-version 3.12 \
  --python-platform x86_64-manylinux_2_31 \
  --output-file requirements.lock.txt
```

## Docker (fully isolated)

If you prefer not to install Python dependencies locally, use the provided Dockerfile:

```bash
# Build image
docker build -f benchmark/Dockerfile -t gigastt-benchmark .

# Run benchmark with mounted model caches
docker run -v ~/.gigastt/models:/root/.gigastt/models:ro \
           -v ~/.gigastt/benchmarks:/root/.gigastt/benchmarks:ro \
           -v $(pwd)/benchmark/results:/workspace/benchmark/results \
           gigastt-benchmark \
           --max-samples 100 --runners all
```

Or use Docker Compose:

```bash
cd benchmark
GIGASTT_BENCHMARK_MAX_SAMPLES=100 docker-compose up
```

> **Note:** On macOS, Docker Desktop must be running. On Linux with NVIDIA GPUs, add `runtime: nvidia` to `docker-compose.yml` and use `--gpus all` with `docker run`.

## Datasets

The benchmark supports multiple Russian speech datasets. Use `--dataset <name>` to select one (default: `golos_crowd`).

### Golos crowd

The default **Golos crowd** test set (9 994 samples of Russian speech).

- **Source:** SberDevices
- **Repository:** https://github.com/sberdevices/golos
- **Paper:** Karpov et al., *Golos: Russian Dataset for Speech Research*, arXiv:2106.10161 (2021)
- **License:** Sber Public License (attribution/non-commercial/share-alike) — https://github.com/sberdevices/golos/blob/master/license/en_us.pdf

```bash
# Download and extract (one-time)
python ../scripts/extract_golos.py
```

### Golos crowd 1k

A deterministic 1 000-sample slice (`random.seed(42)`) of the Golos crowd test
set. Use this for cross-dataset comparisons so all domains have the same sample
size and comparable confidence intervals.

```bash
python benchmark.py --dataset golos_crowd_1k --max-samples 0
```

### Golos farfield

The **Golos farfield** test set (1 916 samples) recorded at a distance from the microphone.

- **Source:** SberDevices
- **Repository:** https://github.com/sberdevices/golos
- **Paper:** Karpov et al., *Golos: Russian Dataset for Speech Research*, arXiv:2106.10161 (2021)
- **License:** Sber Public License (attribution/non-commercial/share-alike) — https://github.com/sberdevices/golos/blob/master/license/en_us.pdf

```bash
# Download and extract (one-time), then create the committed 1 000-sample manifest
python ../scripts/extract_golos_farfield.py
```

Run the benchmark on the farfield slice:

```bash
python benchmark.py --dataset golos_farfield --max-samples 0
```

### Common Voice Russian

An alternative benchmark slice can be prepared from **Mozilla Common Voice** Russian (`ru`) test split.

- **Source:** Mozilla Common Voice contributors
- **Dataset:** https://huggingface.co/datasets/mozilla-foundation/common_voice_16_1
- **Project page:** https://commonvoice.mozilla.org/ru
- **License:** CC0-1.0

```bash
# Prepare a deterministic 1000-sample slice (one-time)
python ../scripts/prepare_common_voice_ru.py
```

Run the benchmark on the Common Voice slice:

```bash
python benchmark.py --dataset common_voice_ru --max-samples 0
```

### OpenSTT phone calls

An **OpenSTT** `asr_calls_2_val` validation slice (1 000 manually-annotated phone-call samples).

- **Source:** snakers4 / OpenSTT
- **Repository:** https://github.com/snakers4/open_stt
- **License:** CC BY-NC 4.0 — https://creativecommons.org/licenses/by-nc/4.0/

```bash
# Prepare a deterministic 1000-sample slice (one-time).
# The full archive is ~0.8 GB; use --use-unpacked-source to fetch only the
# selected 1000 wav+txt pairs instead.
python ../scripts/prepare_openstt_calls.py --use-unpacked-source
```

Run the benchmark on the OpenSTT phone-calls slice:

```bash
python benchmark.py --dataset openstt_calls --max-samples 0
```

### OpenSTT YouTube

An **OpenSTT** `public_youtube700_val` validation slice (1 000 manually-annotated YouTube samples).

- **Source:** snakers4 / OpenSTT
- **Repository:** https://github.com/snakers4/open_stt
- **License:** CC BY-NC 4.0 — https://creativecommons.org/licenses/by-nc/4.0/

```bash
python ../scripts/prepare_openstt_youtube.py --use-unpacked-source
```

Run the benchmark on the OpenSTT YouTube slice:

```bash
python benchmark.py --dataset openstt_youtube --max-samples 0
```

### Common Voice Russian

An alternative benchmark slice can be prepared from **Mozilla Common Voice** Russian (`ru`) test split.

- **Source:** Mozilla Common Voice contributors
- **Dataset:** https://huggingface.co/datasets/mozilla-foundation/common_voice_16_1
- **Project page:** https://commonvoice.mozilla.org/ru
- **License:** CC0-1.0

```bash
# Prepare a deterministic 1000-sample slice (one-time).
# Hugging Face may require accepting the dataset terms or setting HF_TOKEN.
python ../scripts/prepare_common_voice_ru.py
```

Run the benchmark on the Common Voice slice:

```bash
python benchmark.py --dataset common_voice_ru --max-samples 0
```

If the external dataset is missing, the benchmark falls back to the bundled fixtures (15 samples) from `crates/gigastt/tests/fixtures/`.

## Output Format

`results.json` contains run metadata, per-engine summaries with failures and 95% CI, and per-sample details:

```json
{
  "manifest_samples": 100,
  "total_failures": 0,
  "runners": [
    {
      "name": "gigastt",
      "samples": 100,
      "failures": 0,
      "wer": 11.40,
      "ci_low": 10.9,
      "ci_high": 11.9,
      "rtf": 0.045,
      "total_errors": 57,
      "total_ref_words": 500,
      "details": [...]
    }
  ],
  "metadata": {
    "collected_at": "2026-06-12T14:32:00+00:00",
    "host": { "cpu": "...", "ram_bytes": ..., "os": "...", "python_version": "..." },
    "dataset": { "name": "golos", "source": "...", "license": "...", "manifest_path": "..." },
    "engines": [ { "name": "gigastt", "version": "...", "model_sha256": "..." }, ... ]
  }
}
```

## CI / Automation

A GitHub Action runs the benchmark weekly (Sunday at 04:00 UTC) on `ubuntu-latest` and commits `results.json` to the `benchmark-results-local` branch. See `.github/workflows/benchmark.yml`.

### Badges

Add to your README:

```markdown
![WER](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fekhodzitsky%2Fgigastt%2Fbenchmark-results-local%2Fresults.json&query=%24.runners%5B0%5D.wer&suffix=%25&label=WER&color=blue)
![RTF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fekhodzitsky%2Fgigastt%2Fbenchmark-results-local%2Fresults.json&query=%24.runners%5B0%5D.rtf&suffix=x&label=RTF&color=green)
```
