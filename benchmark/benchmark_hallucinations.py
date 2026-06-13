#!/usr/bin/env python3
"""Measure hallucination rate on non-speech audio (MUSAN)."""

import argparse
import json
import os
import subprocess
import time
import urllib.request
import wave
import zipfile
from pathlib import Path

import common


MUSAN_URL = "http://www.openslr.org/resources/17/musan.tar.gz"


def download_musan(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "musan.tar.gz"
    if not archive.exists():
        print(f"Downloading MUSAN from {MUSAN_URL} ...")
        urllib.request.urlretrieve(MUSAN_URL, archive)
    extracted = cache_dir / "musan"
    if not extracted.exists():
        print("Extracting MUSAN ...")
        subprocess.run(["tar", "-xzf", str(archive), "-C", str(cache_dir)], check=True)
    return extracted


def collect_non_speech_clips(musan_dir: Path, max_clips: int = 100) -> list[Path]:
    clips = []
    for sub in ("noise", "music"):
        path = musan_dir / sub
        if path.exists():
            clips.extend(sorted(path.rglob("*.wav")))
    return clips[:max_clips]


def count_words(text: str) -> int:
    return len(text.split())


def audio_duration(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as w:
        return w.getnframes() / w.getframerate()


def evaluate_runner(runner, clips: list[Path]) -> dict:
    total_words = 0
    total_minutes = 0.0
    for clip in clips:
        try:
            hyp, _ = runner.transcribe(str(clip))
        except Exception as e:
            print(f"[{runner.name}] error on {clip}: {e}")
            hyp = ""
        total_words += count_words(hyp)
        total_minutes += audio_duration(clip) / 60.0
    wpm = total_words / total_minutes if total_minutes > 0 else 0.0
    return {
        "name": runner.name,
        "clips": len(clips),
        "words_inserted": total_words,
        "audio_minutes": round(total_minutes, 2),
        "words_per_minute": round(wpm, 2),
    }


def _load_runner_classes() -> list:
    """Load runner classes dynamically, skipping any not yet exported."""
    import importlib

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
            print(f"[benchmark_hallucinations] Skipping {name}: {e}")
    return classes


def main():
    parser = argparse.ArgumentParser(description="Hallucination benchmark on MUSAN non-speech")
    parser.add_argument("--max-clips", type=int, default=20, help="Non-speech clips to use")
    parser.add_argument("--cache-dir", default=str(Path.home() / ".cache" / "musan"))
    parser.add_argument("--output", default="results_hallucinations.json")
    parser.add_argument("--runners", default="all")
    args = parser.parse_args()

    runner_classes = _load_runner_classes()
    musan_dir = download_musan(Path(args.cache_dir))
    clips = collect_non_speech_clips(musan_dir, max_clips=args.max_clips)
    print(f"Using {len(clips)} non-speech clips")

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
                results.append(evaluate_runner(runner, clips))
        else:
            results.append(evaluate_runner(runner, clips))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"runners": results}, f, ensure_ascii=False, indent=2)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
