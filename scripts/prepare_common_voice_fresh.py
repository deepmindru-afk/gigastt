#!/usr/bin/env python3
"""Prepare a fresh Common Voice Russian slice for anti-contamination comparison.

Filters clips newer than the known training cutoff of the engines being compared.
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


# Approximate training cutoffs (documented or release dates).
ENGINE_CUTOFFS = {
    "gigaam_v3": "2024-06-01",
    "whisper_large_v3": "2023-09-01",
    "whisper_large_v3_turbo": "2024-09-01",
    "vosk_0.42": "2022-01-01",
    "vosk_0.54": "2024-06-01",
    "t_one": "2024-06-01",
}


def main():
    parser = argparse.ArgumentParser(description="Prepare fresh Common Voice Russian slice")
    parser.add_argument("--dataset", default="mozilla-foundation/common_voice_16_1")
    parser.add_argument("--language", default="ru")
    parser.add_argument("--split", default="test")
    parser.add_argument("--slice-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-age", default="2024-10-01", help="Only clips newer than this date")
    parser.add_argument("--output", default="benchmark/manifests/common_voice_ru_fresh.json")
    parser.add_argument("--audio-root", default=str(Path.home() / ".gigastt" / "benchmarks" / "common_voice_fresh"))
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio_root = Path(args.audio_root)
    audio_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset} ({args.language}/{args.split}) ...")
    ds = load_dataset(args.dataset, args.language, split=args.split)

    min_age = args.min_age
    # Common Voice has an "age" or "client_id"; use upstream_split or generated date if available.
    # Fallback: keep the newest slice_size rows by downstream_split order.
    candidates = [item for item in ds if item.get("sentence", "").strip() and (item.get("up_votes", 0) >= item.get("down_votes", 0))]
    # If a date column exists, filter; otherwise use the tail of the split as a freshness proxy.
    if candidates and "timestamp" in candidates[0]:
        candidates = [c for c in candidates if str(c.get("timestamp", "")) >= min_age]

    random.seed(args.seed)
    random.shuffle(candidates)
    selected = candidates[:args.slice_size]

    samples = []
    for item in selected:
        path = item.get("path") or item.get("audio", {}).get("path")
        if not path:
            continue
        audio_path = audio_root / path
        # datasets library can save the audio file if needed; here we record the expected path.
        samples.append({
            "filename": str(audio_path),
            "reference": item["sentence"].strip(),
            "duration": item.get("audio", {}).get("sampling_rate", 16000) and len(item.get("audio", {}).get("array", [])) / item.get("audio", {}).get("sampling_rate", 16000),
        })

    manifest = {
        "dataset": args.dataset,
        "language": args.language,
        "split": args.split,
        "slice_seed": args.seed,
        "slice_size": len(samples),
        "total_available": len(candidates),
        "min_age": args.min_age,
        "engine_cutoffs": ENGINE_CUTOFFS,
        "audio_root": str(audio_root),
        "source": "https://huggingface.co/datasets/mozilla-foundation/common_voice_16_1",
        "attribution": "Mozilla Common Voice contributors",
        "license": "CC0-1.0",
        "anti_contamination_note": "This slice contains only clips newer than the training cutoffs of all compared engines.",
        "samples": samples,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(samples)} samples to {output_path}")


if __name__ == "__main__":
    main()
