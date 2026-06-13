#!/usr/bin/env python3
"""Extract WAV files and manifest from the Golos farfield test parquet.

Dataset provenance:
  - Name: Golos (Russian speech corpus)
  - Authors: Alexander Denisenko, Angelina Kovalenko, Fedor Minkin, Nikolay Karpov
    (SberDevices)
  - Paper: Karpov et al., "Golos: Russian Dataset for Speech Research",
    arXiv:2106.10161 (2021)
  - Repository: https://github.com/sberdevices/golos
  - License: Sber Public License (attribution/non-commercial/share-alike)
    https://github.com/sberdevices/golos/blob/master/license/en_us.pdf

This script expects the farfield-domain parquet file to be placed in
~/.gigastt/benchmarks/golos/farfield/ (e.g. downloaded from the HuggingFace
mirror at bond005/sberdevices_golos_10h_farfield).

It extracts all 1 916 WAV files to ~/.gigastt/benchmarks/golos_farfield_wav/
and writes two manifests:
  - ~/.gigastt/benchmarks/golos_farfield_wav/manifest.json (full set)
  - benchmark/manifests/golos_farfield.json (deterministic 1 000-sample slice,
    seed 42, relative filenames)
"""

import io
import json
import os
import random
import sys
import wave
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf


SRC_DIR = Path("~/.gigastt/benchmarks/golos/farfield").expanduser()
DST_DIR = Path("~/.gigastt/benchmarks/golos_farfield_wav").expanduser()
SLICE_SIZE = 1000
SLICE_SEED = 42
SAMPLE_RATE = 16000
REPO_ROOT = Path(__file__).parent.parent
SLICE_MANIFEST_PATH = REPO_ROOT / "benchmark" / "manifests" / "golos_farfield.json"


def _write_pcm16_wav(path: Path, audio_bytes: bytes) -> float:
    """Decode any WAV/PCM source and write a 16 kHz mono PCM16 WAV.

    The Golos farfield parquet stores float32 WAVs, which some runners (e.g.
    Vosk) cannot consume. We normalize to PCM16 to keep the benchmark uniform.
    """
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)

    pcm = (np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())
    return len(pcm) / SAMPLE_RATE


def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)
    SLICE_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(SRC_DIR.glob("*.parquet"))
    if not parquet_files:
        print(f"No parquet files found in {SRC_DIR}", file=sys.stderr)
        sys.exit(1)

    full_samples = []
    total = 0

    for pf_path in parquet_files:
        print(f"Processing {pf_path.name} ...")
        table = pq.read_table(str(pf_path))
        n_rows = len(table)
        audio_col = table["audio"]
        text_col = table["text"]
        id_col = table["id"]

        for i in range(n_rows):
            sample_id = id_col[i].as_py()
            text = text_col[i].as_py()
            audio_bytes = audio_col[i]["bytes"].as_py()

            wav_path = DST_DIR / f"{sample_id}.wav"
            duration = _write_pcm16_wav(wav_path, audio_bytes)

            full_samples.append({
                "filename": str(wav_path),
                "reference": text,
                "duration": duration,
            })
            total += 1

    full_manifest_path = DST_DIR / "manifest.json"
    with open(full_manifest_path, "w", encoding="utf-8") as f:
        json.dump(full_samples, f, ensure_ascii=False, indent=2)

    print(f"Extracted {total} samples to {DST_DIR}")
    print(f"Full manifest: {full_manifest_path}")

    # Deterministic slice for the committed benchmark manifest.
    rng = random.Random(SLICE_SEED)
    slice_samples = rng.sample(full_samples, min(SLICE_SIZE, len(full_samples)))

    # Convert absolute paths to filenames relative to the audio root.
    audio_root = "~/.gigastt/benchmarks/golos_farfield_wav"
    committed_samples = []
    for s in slice_samples:
        rel_name = Path(s["filename"]).name
        committed_samples.append({
            "filename": rel_name,
            "reference": s["reference"],
            "duration": s["duration"],
        })

    committed_manifest = {
        "dataset": "golos_farfield",
        "audio_root": audio_root,
        "slice_seed": SLICE_SEED,
        "slice_size": len(committed_samples),
        "total_available": total,
        "license": "Sber Public License (attribution/non-commercial/share-alike)",
        "source": "https://github.com/sberdevices/golos",
        "samples": committed_samples,
    }

    with open(SLICE_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(committed_manifest, f, ensure_ascii=False, indent=2)

    print(f"Committed slice manifest: {SLICE_MANIFEST_PATH}")


if __name__ == "__main__":
    main()
