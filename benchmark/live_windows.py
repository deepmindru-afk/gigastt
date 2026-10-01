#!/usr/bin/env python3
"""Run the opt-in live-window research harness and summarize fixture quality.

Uses existing INT8/sidecar models; does not download models or change defaults.
Stage measurements are elapsed time; process CPU is sampled separately on Linux.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
from pathlib import Path

from common import compute_wer, normalize_for_wer

CONFIGS = {
    "baseline": (800, 2.5, 0),
    "stride400": (400, 2.5, 0),
    "stride1200": (1200, 2.5, 0),
    "window5": (800, 5.0, 0),
    "vad": (800, 2.5, 1),
}


def summarize(raw: dict, cpu_ticks_per_second: int = 100) -> dict:
    rows = raw["rows"]
    corpus = [r for r in rows if r["file"].endswith(".wav")]
    references = errors = revisions = final_revisions = partials = 0
    first = []
    last_final_lags = []
    stages: dict[str, float] = {}
    stage_cpu: dict[str, float] = {}
    windows = []
    for row in rows:
        final_events = [e for e in row["events"] if e["final"] and e["text"].strip()]
        if final_events:
            last_final_lags.append(final_events[-1]["at_ms"] - row["audio_samples"] / 16)
        previous: list[str] = []
        first_partial = None
        for event in row["events"]:
            words = normalize_for_wer(event["text"])
            if event["final"]:
                common = 0
                for a, b in zip(previous, words):
                    if a != b:
                        break
                    common += 1
                final_revisions += len(previous) - common
                previous = []
                continue
            if not words:
                continue
            partials += 1
            if first_partial is None:
                first_partial = event["at_ms"]
            common = 0
            for a, b in zip(previous, words):
                if a != b:
                    break
                common += 1
            revisions += len(previous) - common
            previous = words
        if first_partial is not None:
            first.append(first_partial)
        for key, nanos in row["probe"]["stages_ns"].items():
            stages[key] = stages.get(key, 0) + nanos / 1e9
        for key, nanos in row["probe"].get("process_thread_cpu_ns", {}).items():
            stage_cpu[key] = stage_cpu.get(key, 0) + nanos / 1e9
        windows.extend(row["probe"]["windows"])
    for row in corpus:
        _, row_errors, words = compute_wer(row["reference"], row["text"])
        errors += row_errors
        references += words
    focused = []
    for row in rows:
        if row["file"].endswith(".wav"):
            continue
        _, row_errors, words = compute_wer(row["reference"], row["text"])
        focused.append({"file": row["file"], "worker": row["worker"], "errors": row_errors,
                        "reference_words": words, "text": row["text"],
                        "final_endpoints": [e["endpoint"] for e in row["events"] if e["final"]]})
    audio = sum(r["audio_samples"] for r in rows) / 16000
    return {
        "config": raw["config"], "corpus_clips": len(corpus),
        "errors": errors, "reference_words": references,
        "wer_percent": 100 * errors / references if references else None,
        "process_thread_cpu_during_stage_seconds": stage_cpu,
        "stage_elapsed_seconds": stages, "process_cpu_seconds": raw["process_cpu_ticks"] / cpu_ticks_per_second,
        "audio_seconds": audio, "wall_seconds": raw["elapsed_ms"] / 1000,
        "first_partial_median_ms": statistics.median(first) if first else None,
        "no_partial": len(rows) - len(first),
        "last_final_relative_audio_end_median_ms": statistics.median(last_final_lags) if last_final_lags else None,
        "final_after_stop_median_ms": statistics.median(r["final_after_stop_ms"] for r in rows),
        "partial_retracted_words": revisions, "final_retracted_words": final_revisions, "partial_events": partials,
        "peak_rss_kib": max(r["peak_rss_kib"] for r in rows),
        "decode_windows": len(windows),
        "encoded_audio_seconds": sum(w["samples"] for w in windows) / 16000,
        "max_window_seconds": max(w["samples"] for w in windows) / 16000,
        "off_grid_windows": sum(w["start"] % 160 != 0 for w in windows),
        "focused": focused,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument("--concurrency", nargs="+", type=int, choices=[1, 2], default=[1, 2])
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        model_dir = Path.home() / ".gigastt/models"
        files = ["v3_rnnt_encoder_int8.onnx", "v3_rnnt_decoder.onnx", "v3_rnnt_joint.onnx", "v3_vocab.txt",
                 "punct/rupunct_small_int8.onnx", "punct/tokenizer.json", "punct/config.json", "vad/silero_vad.onnx"]
        source_files = ["crates/gigastt-core/src/inference/engine/stream.rs", "crates/gigastt-core/src/inference/engine/infer.rs", "crates/gigastt-core/src/inference/engine/live_probe.rs", "crates/gigastt-core/src/inference/engine/live_probe/experiment.rs"]
        provenance = {"source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in source_files}, "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                      "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
                      "platform": platform.platform(), "profile": "debug", "cpu_ticks_per_second": os.sysconf("SC_CLK_TCK"),
                      "models_sha256": {name: hashlib.sha256((model_dir / name).read_bytes()).hexdigest() for name in files}}
        (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        for case in args.cases:
            stride, window, vad = CONFIGS[case]
            for concurrency in args.concurrency:
                name = f"{case}-c{concurrency}"
                env = dict(os.environ, GIGASTT_LIVE_PROBE=str(output / f"{name}.json"),
                           GIGASTT_LIVE_STRIDE_MS=str(stride), GIGASTT_LIVE_WINDOW_SECS=str(window),
                           GIGASTT_LIVE_VAD=str(vad), GIGASTT_LIVE_CONCURRENT=str(concurrency))
                print(f"Running {name}", flush=True)
                with (output / f"{name}.log").open("w") as log:
                    subprocess.run(["cargo", "test", "-p", "gigastt-core", "--lib", "benchmark_live_window_workload",
                                    "--", "--ignored", "--nocapture", "--test-threads=1"], cwd=root, env=env,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
    tick_rate = json.loads((output / "provenance.json").read_text())["cpu_ticks_per_second"]
    summaries = {path.stem: summarize(json.loads(path.read_text()), tick_rate) for path in sorted(output.glob("*-c[12].json"))}
    (output / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
