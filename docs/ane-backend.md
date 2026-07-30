# ANE (CoreML) backend — user guide

> Optional `--features ane`: run the GigaAM v3 **encoder** on the Apple **Neural
> Engine** via native Core ML `.mlpackage`s. Additive and opt-in; the default `ort`
> path is unchanged. The original pre-implementation plan is archived in
> [`docs/archive/ane-backend-plan.md`](archive/ane-backend-plan.md).

gigastt's default inference backend is ONNX Runtime (`ort`). An **optional**
native Apple **Neural Engine** backend is available behind the `ane` Cargo
feature: it runs the GigaAM v3 **encoder** on the ANE via per-bucket fixed-shape
Core ML `.mlpackage`s, while the decoder/joiner (and any encoder window outside
the bucket fill floor) stay on the `ort` CPU path.

It is **additive and opt-in** — the default build is unchanged and still uses `ort`.

## Status

- **macOS ARM64 (Apple Silicon) only.** The backend links Apple's Core ML
  framework; on every other target the `ane` feature degrades to the `ort` path.
- Targets the default **`rnnt`** head (char vocab). An `e2e_rnnt` model
  transparently falls back to the `ort` encoder (the ANE backend is rnnt-only,
  mirroring `candle`).
- **File-mode** backend: the encoder window is padded up to the nearest fixed
  bucket and run on the ANE. **Streaming and short windows below the fill floor
  fall back to the CPU/`ort` encoder** — they work without crashing but get no
  ANE speedup (this is intentional; see [Behavior](#behavior)).
- `ane` is **mutually exclusive** with `coreml` (the ort CoreML EP), `cuda`,
  `nnapi`, and `candle` (a `compile_error!` fires if combined). Auxiliary models
  (VAD, punctuation) continue to run on the CPU `ort` path.

## 1. Build with the feature

```sh
# server binary (macOS ARM64 only)
cargo build --release --features ane

# or just the core library
cargo build -p gigastt-core --release --features ane
```

Do **not** combine with `--features coreml`, `--features cuda`,
`--features nnapi`, or `--features candle` (mutually exclusive). The default
`ort` build remains `cargo build --release` (unchanged).

## 2. Obtain the bucket packages

The ANE backend reads per-bucket `.mlpackage`s from
`~/.gigastt/models/ane/` (a sibling of the ONNX encoder), one fixed-shape
package per bucket in the ladder `[512, 768, 1536, 3000]` mel frames (≈5 s / 8 s /
15 s / 30 s windows):

```
~/.gigastt/models/ane/gigaam_v3_encoder_512.mlpackage
~/.gigastt/models/ane/gigaam_v3_encoder_768.mlpackage
~/.gigastt/models/ane/gigaam_v3_encoder_1536.mlpackage
~/.gigastt/models/ane/gigaam_v3_encoder_3000.mlpackage
```

**Pre-built packages (default):** the per-bucket packages are published on the
[`ane-v3-2026-06-24` GitHub release](https://github.com/ekhodzitsky/gigastt/releases/tag/ane-v3-2026-06-24);
`gigastt download --ane` fetches them (SHA-256-verified) into the same
directory:

```sh
gigastt download --ane
```

**Convert locally instead:** to rebuild the packages from the PyTorch model
(e.g. with a different bucket ladder), run on macOS ARM64:

```sh
uv run --python 3.13 \
    --with torch --with coremltools --with gigaam --with soundfile --with scikit-learn \
    python scripts/convert_gigaam_ane.py
```

This writes the per-bucket `.mlpackage`s into `~/.gigastt/models/ane/`.

The bucket-768 package alone is enough to engage the ANE path for short files;
the engine logs which buckets it found and compiles each present package once
(shared across the session pool).

## 3. Run

The server and CLI use the ANE encoder automatically when built with
`--features ane` and a `rnnt` model is loaded — `production_factory` routes the
encoder through the composite ANE factory. Usage is otherwise identical to the
default build:

```sh
gigastt serve                      # ANE encoder + ort decoder/joiner
gigastt transcribe audio.wav       # file-mode transcription on the ANE
```

If no bucket package is present, the encoder load fails with a clear message
pointing at the conversion / `gigastt download --ane` step.

## Behavior

- **Pad-up to fixed buckets.** Each file-mode encoder call pads its mel window up
  to the smallest bucket `N ≥ frames`, runs it on the ANE (Float16), then trims
  the output back to the real frame count. The frame count emitted matches the
  `ort` encoder exactly.
- **Fill floor (`FILL_FLOOR = 0.5`).** A window must fill **≥ 50%** of its bucket
  for the ANE path to be trusted. Below that, the mask-free zero-pad output
  diverges enough that a borderline token could flip, so the window falls back to
  the variable-length `ort` encoder. The smallest bucket (512) therefore covers
  real frame counts in `[256, 512]` — the typical 3–5 s clip range — at higher
  fill (less pad-up waste / lower divergence) than routing those clips up to 768;
  768 now covers `(512, 768]`. All buckets clear the ~288-mel ANE-residency floor,
  so each stays resident on the Neural Engine.
- **Streaming falls back to CPU.** The streaming window is capped at 2.5 s
  (≤ 250 mel frames), which is below the 256-frame floor of the smallest (512)
  bucket, so **every streaming window takes the `ort` fallback**. Streaming works exactly as
  on the default build — no crash, no ANE benefit. ANE is a file-mode
  accelerator.
- **Over-max windows.** Files longer than the largest bucket use gigastt's
  existing 24 s windowed chunking; any window outside the bucket range falls back
  to `ort` (no silent truncation).

## Performance & accuracy (honest numbers)

Measured on an Apple M1 Pro at the v2.5.0 ship gate (`rnnt` head, Golos clips) —
these are the numbers carried in the v2.5.0 CHANGELOG entry and `specs/todo.md`.
An earlier revision of this section quoted a pre-ship measurement round
(≈ 3.7× e2e, ~230× encoder); it is superseded by the shipped figures below.

- **Warm end-to-end ≈ 10× over the `ort` CPU build** (≈ 112 RTFx warm),
  **decode-bound**: the ANE cuts the encoder to ≈ 23.6 ms from ≈ 369 ms per
  window (≈ 15.6×, 99.8% ANE residency), but the CPU RNN-T greedy decode loop
  and feature extraction dominate the full pipeline once the encoder is
  offloaded, so the e2e win is far smaller than the raw encoder win.
- **WER vs `ort` ≈ 1.11%** on the 15-clip Golos set: transcripts are
  byte-identical except for a single borderline FP16-pad-up token flip on one
  clip. The ANE encoder is FP16 (not byte-exact), so parity is "near-lossless",
  not bit-exact (unlike the `candle` backend, which is byte-for-byte identical).
- **Cold-start:** a compiled-model disk cache cuts the ~20 s first load to
  ~86 ms on later starts.

## Confirming the ANE path is engaged

- **Startup log:** `gigastt serve --features ane` on a `rnnt` model logs
  `ANE encoder backend active (Core ML / Apple Neural Engine, macOS ARM64): …`.
  On an `e2e_rnnt` model it instead logs that the head is not `rnnt` and the ort
  encoder is used.
- **Per-window debug log:** at `--log-level debug` the encoder logs
  `ANE encoder path (bucketed pad-up)` (with the chosen bucket) for file-mode
  windows, and `ANE encoder path (ort fallback: no bucket within fill-floor)`
  for streaming / sub-floor windows.
- **RTFx:** a file transcription completing well above real time (with the
  decode loop, not the encoder, as the bottleneck) confirms the ANE path.
