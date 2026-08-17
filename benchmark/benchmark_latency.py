#!/usr/bin/env python3
"""Measure streaming latency: TTFP / TTFS p50–p95 and finalization lag.

Single clip (legacy):
    python benchmark_latency.py --wav path.wav --port 9877

Corpus (p50/p95):
    python benchmark_latency.py --dataset golos_crowd --max-samples 100 --port 9877

The server must already be running (warm, `--pool-size 1` for the published
protocol). Clock starts on the first audio frame after Ready + configure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import load_manifest
from streaming import STREAM_CHUNK_MS, STREAM_PROTOCOL_VERSION, STREAM_SAMPLE_RATE, summarize_latency, transcribe_ws


def evaluate_gigastt(wav_path: str, port: int = 9877, chunk_ms: int = STREAM_CHUNK_MS) -> dict:
    """One clip. Keeps the legacy JSON keys used by older notes."""
    _text, _elapsed, session = transcribe_ws(wav_path, port=port, chunk_ms=chunk_ms, pace=True)
    lags = session.get("partial_lags_ms") or []
    ordered = sorted(lags)
    n = len(ordered)
    return {
        "time_to_first_partial_ms": session["ttfp_ms"],
        "first_partial_after_audio_ms": session["ttfp_ms"],
        "ttfs_ms": session["ttfs_ms"],
        "finalization_lag_ms": session["finalization_lag_ms"],
        "audio_duration_ms": session["audio_duration_ms"],
        "total_audio_sent_ms": session["total_audio_sent_ms"],
        "timed_out": session["timed_out"],
        "no_partial": session["no_partial"],
        "onset_s": session["onset_s"],
        "ttfp_ms": session["ttfp_ms"],
        "partial_lags_ms": lags,
        "partial_response_lag_ms": (
            {
                "count": n,
                "min": ordered[0],
                "median": ordered[n // 2],
                "max": ordered[-1],
            }
            if ordered
            else None
        ),
        "wav": wav_path,
        "engine": "gigastt",
        "protocol": {
            "sample_rate": STREAM_SAMPLE_RATE,
            "chunk_ms": chunk_ms,
            "version": STREAM_PROTOCOL_VERSION,
        },
    }


def evaluate_corpus(samples: list[dict], port: int, chunk_ms: int) -> dict:
    rows = []
    for idx, sample in enumerate(samples):
        wav_path = sample["filename"]
        try:
            row = evaluate_gigastt(wav_path, port=port, chunk_ms=chunk_ms)
            row["ok"] = True
        except Exception as e:
            print(f"  [{idx + 1}/{len(samples)}] ERROR {Path(wav_path).name}: {e}")
            row = {
                "wav": wav_path,
                "ttfp_ms": None,
                "ttfs_ms": None,
                "finalization_lag_ms": None,
                "timed_out": False,
                "no_partial": True,
                "partial_lags_ms": [],
                "ok": False,
                "error": str(e),
            }
        # summarize_latency reads ttfp_ms / timed_out / no_partial
        if "ttfp_ms" not in row:
            row["ttfp_ms"] = row.get("time_to_first_partial_ms")
        if row.get("partial_response_lag_ms") and "partial_lags_ms" not in row:
            # single-clip helper stores a summary; corpus rollup wants the list
            row["partial_lags_ms"] = []
        rows.append(row)
        print(
            f"  [{idx + 1}/{len(samples)}] TTFP={row.get('ttfp_ms')}  "
            f"TTFS={row.get('ttfs_ms')}  {Path(wav_path).name}"
        )
    return {
        "engine": "gigastt",
        "port": port,
        "chunk_ms": chunk_ms,
        "summary": summarize_latency(rows),
        "clips": rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Streaming latency benchmark")
    parser.add_argument("--wav", help="Single 16 kHz (or resampled) WAV")
    parser.add_argument("--dataset", help="Manifest name (e.g. golos_crowd)")
    parser.add_argument("--max-samples", type=int, default=100, help="0 = all (dataset mode)")
    parser.add_argument("--output", default="results_latency.json")
    parser.add_argument("--port", type=int, default=9877)
    parser.add_argument("--chunk-ms", type=int, default=STREAM_CHUNK_MS)
    args = parser.parse_args()

    if bool(args.wav) == bool(args.dataset):
        parser.error("provide exactly one of --wav or --dataset")

    if args.wav:
        result = evaluate_gigastt(args.wav, port=args.port, chunk_ms=args.chunk_ms)
    else:
        max_samples = args.max_samples if args.max_samples > 0 else None
        manifest = load_manifest(max_samples=max_samples, dataset=args.dataset)
        print(f"Loaded {len(manifest['samples'])} samples from '{args.dataset}'")
        result = evaluate_corpus(manifest["samples"], port=args.port, chunk_ms=args.chunk_ms)
        result["dataset"] = args.dataset

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result if args.wav else result["summary"], ensure_ascii=False, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
