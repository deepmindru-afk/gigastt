#!/usr/bin/env python3
"""Extract WAV files and manifest from the Golos crowd test parquet.

Dataset provenance:
  - Name: Golos (Russian speech corpus)
  - Authors: Alexander Denisenko, Angelina Kovalenko, Fedor Minkin, Nikolay Karpov
    (SberDevices)
  - Paper: Karpov et al., "Golos: Russian Dataset for Speech Research",
    arXiv:2106.10161 (2021)
  - Repository: https://github.com/sberdevices/golos
  - License: Sber Public License (attribution/non-commercial/share-alike)
    https://github.com/sberdevices/golos/blob/master/license/en_us.pdf

This script expects the crowd-domain parquet files to be placed in
~/.gigastt/benchmarks/golos/crowd/ (e.g. downloaded from the HuggingFace
mirror at bond005/sberdevices_golos_10h_crowd).
"""

import json
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq


def main():
    src_dir = Path("~/.gigastt/benchmarks/golos/crowd").expanduser()
    dst_dir = Path("~/.gigastt/benchmarks/golos_wav").expanduser()
    dst_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(src_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"No parquet files found in {src_dir}", file=sys.stderr)
        sys.exit(1)

    manifest = []
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

            wav_path = dst_dir / f"{sample_id}.wav"
            with open(wav_path, "wb") as f:
                f.write(audio_bytes)

            manifest.append({"filename": str(wav_path), "reference": text})
            total += 1

    manifest_path = dst_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Extracted {total} samples to {dst_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
