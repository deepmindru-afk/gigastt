#!/usr/bin/env python3
"""Measure punctuation and capitalization F1 on a dataset with original text."""

import argparse
import importlib
import json
import os
from pathlib import Path

import common


PUNCTUATION_MARKS = set(".,!?;:\"'()[]{}«»—…")


def extract_punctuation(text: str) -> list[tuple[int, str]]:
    """Return (position, mark) for each punctuation char."""
    return [(i, ch) for i, ch in enumerate(text) if ch in PUNCTUATION_MARKS]


def extract_capitalization(text: str) -> list[tuple[int, str]]:
    """Return (position, char) for each uppercase Cyrillic/Latin letter."""
    return [(i, ch) for i, ch in enumerate(text) if ch.isupper() and ch.isalpha()]


def f1_score(ref_items: list, hyp_items: list) -> dict:
    ref_set = set(ref_items)
    hyp_set = set(hyp_items)
    tp = len(ref_set & hyp_set)
    fp = len(hyp_set - ref_set)
    fn = len(ref_set - hyp_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def aggregate(items: list[dict]) -> dict:
    tp = sum(i["tp"] for i in items)
    fp = sum(i["fp"] for i in items)
    fn = sum(i["fn"] for i in items)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def evaluate_runner(runner, manifest: list[dict]) -> dict:
    punct_results = []
    cap_results = []
    for sample in manifest:
        ref = sample["reference"]
        try:
            hyp, _ = runner.transcribe(sample["filename"])
        except Exception as e:
            print(f"[{runner.name}] error on {sample['filename']}: {e}")
            hyp = ""
        punct_results.append(f1_score(extract_punctuation(ref), extract_punctuation(hyp)))
        cap_results.append(f1_score(extract_capitalization(ref), extract_capitalization(hyp)))

    return {
        "name": runner.name,
        "samples": len(manifest),
        "punctuation": aggregate(punct_results),
        "capitalization": aggregate(cap_results),
    }


def _load_runner_classes() -> list:
    """Load runner classes dynamically, skipping any not yet exported."""
    candidates = [
        "GigasttRunner",
        "FasterWhisperRunner",
        "FasterWhisperTurboRunner",
        "WhisperCppRunner",
        "VoskRunner",
        "Vosk054Runner",
        "TOneRunner",
    ]
    classes = []
    for name in candidates:
        try:
            cls = getattr(importlib.import_module("runners"), name)
            classes.append(cls)
        except Exception as e:
            print(f"[benchmark_punctuation] Skipping {name}: {e}")
    return classes


def main():
    parser = argparse.ArgumentParser(description="Punctuation and capitalization benchmark")
    default_dataset = os.environ.get("GIGASTT_BENCHMARK_DATASET", "golos_crowd")
    parser.add_argument("--dataset", default=default_dataset, help="Dataset manifest name")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=int(os.environ.get("GIGASTT_BENCHMARK_MAX_SAMPLES", "50")),
        help="Maximum samples (smoke tests default 50)",
    )
    parser.add_argument("--output", default="results_punctuation.json")
    parser.add_argument("--runners", default="all", help="Comma-separated runner names")
    args = parser.parse_args()

    runner_classes = _load_runner_classes()
    manifest_data = common.load_manifest(
        max_samples=args.max_samples if args.max_samples > 0 else None,
        dataset=args.dataset,
    )
    manifest = manifest_data["samples"]
    print(f"Loaded {len(manifest)} samples for punctuation benchmark")

    all_runners = [cls() for cls in runner_classes]

    requested = set(args.runners.split(",")) if args.runners != "all" else {"all"}
    active = [r for r in all_runners if "all" in requested or r.name in requested]

    results = []
    for runner in active:
        if not runner.is_available():
            print(f"Skipping {runner.name} (not available)")
            continue
        print(f"\n=== {runner.name} ===")
        cm = getattr(runner, "__enter__", lambda: runner)
        cm_exit = getattr(runner, "__exit__", None)
        if cm_exit:
            with runner:
                results.append(evaluate_runner(runner, manifest))
        else:
            results.append(evaluate_runner(runner, manifest))

    output = {"dataset": args.dataset, "runners": results}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
