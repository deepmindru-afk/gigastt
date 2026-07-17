#!/usr/bin/env python3
"""Prepare a fixed 1000-sample LibriSpeech test-clean slice for English ASR benchmarking.

LibriSpeech ``test-clean`` is read English audiobook speech — the standard
clean-English ASR benchmark. Public, no gating, no HF account required. Used to
measure the GigaAM Multilingual (``ml_ctc`` / ``ml_ctc_large``) heads on English.

Dataset provenance:
  - Name: LibriSpeech ASR corpus (test-clean)
  - Source: https://www.openslr.org/12/  (test-clean.tar.gz)
  - Paper: Panayotov et al., "Librispeech: an ASR corpus based on public domain
    audio books", ICASSP 2015
  - License: CC BY 4.0

Downloads + extracts test-clean, then emits a manifest pointing at the FLAC files
directly — gigastt decodes FLAC via symphonia, so no audio conversion is needed.
The deterministic slice is built with ``random.seed(seed)``.
"""

import argparse
import json
import random
import sys
import tarfile
import urllib.request
from pathlib import Path

URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
REPO_ROOT = Path(__file__).parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare a LibriSpeech test-clean benchmark slice")
    p.add_argument(
        "--root",
        type=Path,
        default=Path("~/.gigastt/benchmarks/librispeech").expanduser(),
        help="Directory to download + extract into",
    )
    p.add_argument(
        "--manifest-path",
        type=Path,
        default=REPO_ROOT / "benchmark/manifests/librispeech_test_clean.json",
        help="Manifest JSON path",
    )
    p.add_argument("--slice-size", type=int, default=1000, help="Number of samples in the manifest")
    p.add_argument("--seed", type=int, default=42, help="Random seed for deterministic selection")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    tar_path = args.root / "test-clean.tar.gz"
    extract_root = args.root / "LibriSpeech" / "test-clean"

    if not extract_root.exists():
        if not tar_path.exists():
            print(f"Downloading {URL} ...")
            urllib.request.urlretrieve(URL, tar_path)
        print("Extracting ...")
        with tarfile.open(tar_path) as t:
            t.extractall(args.root, filter="data")  # creates <root>/LibriSpeech/test-clean/...

    # utt_id -> reference (LibriSpeech refs are UPPERCASE, no punctuation; lowercased here).
    refs: dict[str, str] = {}
    for trans in extract_root.rglob("*.trans.txt"):
        for line in trans.read_text(encoding="utf-8").splitlines():
            uid, _, text = line.partition(" ")
            text = text.strip()
            if uid and text:
                refs[uid] = text.lower()

    # utt_id -> flac path relative to the manifest audio_root.
    flac_rel: dict[str, str] = {}
    for flac in extract_root.rglob("*.flac"):
        flac_rel[flac.stem] = str(flac.relative_to(extract_root))

    ids = sorted(uid for uid in refs if uid in flac_rel)
    total_available = len(ids)
    print(f"Total utterances with audio + reference: {total_available}")
    if total_available == 0:
        print("No usable samples found.", file=sys.stderr)
        return 1

    slice_size = min(args.slice_size, total_available)
    rng = random.Random(args.seed)
    chosen = sorted(rng.sample(ids, slice_size))
    samples = [{"filename": flac_rel[uid], "reference": refs[uid]} for uid in chosen]

    manifest = {
        "dataset": "librispeech_test_clean",
        "audio_root": "~/.gigastt/benchmarks/librispeech/LibriSpeech/test-clean",
        "slice_seed": args.seed,
        "slice_size": slice_size,
        "total_available": total_available,
        "language": "en",
        "license": "CC BY 4.0",
        "source": "https://www.openslr.org/12/",
        "attribution": "LibriSpeech (Panayotov et al., ICASSP 2015)",
        "samples": samples,
    }
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Selected {slice_size} samples; manifest: {args.manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
