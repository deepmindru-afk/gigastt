#!/usr/bin/env python3
"""Prepare a fixed 1000-sample Common Voice Russian test slice for benchmarking.

Dataset provenance:
  - Name: Mozilla Common Voice (Russian / ru)
  - Locale: ru
  - Split: test
  - Source: https://huggingface.co/datasets/mozilla-foundation/common_voice_16_1
  - Project page: https://commonvoice.mozilla.org/ru
  - License: CC0-1.0 (public domain dedication)
  - Attribution: Mozilla Common Voice contributors

This script streams the Common Voice 16.1 Russian ``test`` split, writes the
selected samples as 16 kHz mono PCM16 WAV files, and emits a manifest in the
standard format used by ``benchmark/benchmark.py``.

The deterministic slice is built with ``random.seed(42)`` so the same 1000
samples are chosen on every run.
"""

import argparse
import json
import random
import sys
import wave
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Common Voice Russian benchmark slice"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("~/.gigastt/benchmarks/common_voice_ru").expanduser(),
        help="Directory to write WAV files to",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path(__file__).parent.parent
        / "benchmark/manifests/common_voice_ru.json",
        help="Path to write the manifest JSON file",
    )
    parser.add_argument(
        "--slice-size",
        type=int,
        default=1000,
        help="Number of samples to include in the manifest",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sample selection",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="mozilla-foundation/common_voice_16_1",
        help="Hugging Face dataset id to load",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="ru",
        help="Dataset config/locale to load",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=True,
        help="Stream the dataset (default, low memory footprint)",
    )
    parser.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="Download the full split before sampling",
    )
    return parser.parse_args()


def write_wav(path: Path, samples: np.ndarray, sample_rate: int = 16000) -> None:
    """Write a 16 kHz mono PCM16 WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    # datasets decodes to float32 in [-1.0, 1.0]; scale to int16.
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def load_common_voice_ru_test(dataset_id: str = "mozilla-foundation/common_voice_16_1",
                              config: str = "ru",
                              streaming: bool = True):
    """Return the requested Common Voice Russian test split.

    The default follows the task spec (Common Voice 16.1 ``ru``). Pass a
    different ``dataset_id`` if a newer release is required.
    """
    from datasets import load_dataset, Audio

    ds = load_dataset(
        dataset_id,
        config,
        split="test",
        streaming=streaming,
    )
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    return ds


def main() -> int:
    args = parse_args()

    print(f"Loading {args.dataset_id} ({args.config}) test split ...")
    ds = load_common_voice_ru_test(
        dataset_id=args.dataset_id,
        config=args.config,
        streaming=args.streaming,
    )

    # First pass: collect metadata for samples with non-empty references.
    print("Collecting metadata ...")
    candidates = []
    for idx, row in enumerate(ds):
        ref = (row.get("sentence") or "").strip()
        if not ref:
            continue
        candidates.append((idx, ref))
        if (idx + 1) % 500 == 0:
            print(f"  scanned {idx + 1} rows, {len(candidates)} candidates")

    total_available = len(candidates)
    print(f"Total candidates with non-empty references: {total_available}")

    if total_available == 0:
        print("No usable samples found.", file=sys.stderr)
        return 1

    slice_size = min(args.slice_size, total_available)
    if slice_size < args.slice_size:
        print(
            f"Warning: requested {args.slice_size} samples but only "
            f"{total_available} available; using {slice_size}.",
            file=sys.stderr,
        )

    rng = random.Random(args.seed)
    selected = rng.sample(candidates, slice_size)
    selected_indices = {idx for idx, _ in selected}
    selected_refs = {idx: ref for idx, ref in selected}

    # Second pass: materialize the selected samples.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    processed = 0

    print(f"Extracting {slice_size} selected samples ...")
    for idx, row in enumerate(ds):
        if idx not in selected_indices:
            continue

        audio = row["audio"]
        array = np.asarray(audio["array"])
        sample_rate = int(audio["sampling_rate"])
        duration = len(array) / sample_rate if sample_rate > 0 else 0.0

        wav_name = f"{idx:05d}.wav"
        wav_path = args.output_dir / wav_name
        write_wav(wav_path, array, sample_rate=sample_rate)

        samples.append(
            {
                "filename": wav_name,
                "reference": selected_refs[idx],
                "duration": round(duration, 3),
            }
        )
        processed += 1
        if processed % 100 == 0 or processed == slice_size:
            print(f"  wrote {processed}/{slice_size} ({wav_name})")

    manifest = {
        "dataset": "common_voice_ru",
        "audio_root": "~/.gigastt/benchmarks/common_voice_ru",
        "slice_seed": args.seed,
        "slice_size": slice_size,
        "total_available": total_available,
        "license": "CC0-1.0",
        "source": "https://huggingface.co/datasets/mozilla-foundation/common_voice_16_1",
        "attribution": "Mozilla Common Voice contributors",
        "samples": samples,
    }

    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Wrote {processed} WAV files to {args.output_dir}")
    print(f"Manifest: {args.manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
