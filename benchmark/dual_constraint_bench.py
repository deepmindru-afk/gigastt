#!/usr/bin/env python3
"""Warm multi-run dual-constraint measure for public gigastt (ORT lean INT8).

Produces a JSON document comparable to ``benchmark/results_dual_constraint/freeze.json``
metrics, covering:

  - warm REST multi-run BEST/mean on a ~40s concat fixture (primary RTF)
  - warm REST mean/best on five golos fixtures
  - cold-start, ps RSS after ready / after decode
  - optional macOS footprint resident
  - fixture transcripts for quality identity checks
  - lean disk class

Example::

    python3 benchmark/dual_constraint_bench.py \\
      --binary ./target/release/gigastt \\
      --output benchmark/results_dual_constraint/post.json
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIXTURES = REPO_ROOT / "crates" / "gigastt" / "tests" / "fixtures"
DEFAULT_WAVS = [DEFAULT_FIXTURES / f"golos_{i:02d}.wav" for i in range(5)]


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def wav_duration_s(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def rss_mb(pid: int) -> float | None:
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True).strip()
        return round(int(out) / 1024, 1)
    except Exception:
        return None


def footprint_mb(pid: int) -> float | None:
    if sys.platform != "darwin":
        return None
    try:
        out = subprocess.check_output(
            ["/usr/bin/footprint", str(pid)], text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        return None
    # First summary line contains "Footprint: N MB"
    for line in out.splitlines():
        if "Footprint:" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "Footprint:" and i + 1 < len(parts):
                    try:
                        return float(parts[i + 1])
                    except ValueError:
                        return None
    return None


def wait_ready(port: int, timeout_s: float = 180.0) -> float:
    url = f"http://127.0.0.1:{port}/ready"
    deadline = time.perf_counter() + timeout_s
    started = time.perf_counter()
    while time.perf_counter() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    return time.perf_counter() - started
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(0.1)
    raise RuntimeError(f"server on port {port} did not become ready within {timeout_s}s")


def rest_transcribe(port: int, data: bytes, timeout_s: float = 600.0) -> tuple[str, float]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/transcribe",
        data=data,
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8")
    elapsed = time.perf_counter() - t0
    text = json.loads(body).get("text", "").strip()
    return text, elapsed


def build_long_wav(wavs: list[Path], out: Path, repeats: int = 2) -> float:
    frames: list[bytes] = []
    sr = None
    for p in wavs:
        with wave.open(str(p), "rb") as wf:
            assert wf.getnchannels() == 1 and wf.getsampwidth() == 2
            if sr is None:
                sr = wf.getframerate()
            else:
                assert sr == wf.getframerate()
            frames.append(wf.readframes(wf.getnframes()))
    raw = b"".join(frames) * repeats
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr or 16000)
        wf.writeframes(raw)
    return len(raw) / 2 / float(sr or 16000)


def lean_disk_mb(model_dir: Path) -> float:
    files = [
        "v3_rnnt_encoder_int8.onnx",
        "v3_rnnt_decoder.onnx",
        "v3_rnnt_joint.onnx",
        "v3_vocab.txt",
    ]
    total = 0
    for name in files:
        p = model_dir / name
        if p.exists():
            total += p.stat().st_size
    return round(total / (1024 * 1024), 1)


def start_server(binary: str, port: int, pool_size: int, model_dir: str | None) -> subprocess.Popen:
    cmd = [
        binary,
        "serve",
        "--port",
        str(port),
        "--pool-size",
        str(pool_size),
        "--model-variant",
        "rnnt",
        "--punctuation",
        "off",
        "--itn",
        "off",
    ]
    if model_dir:
        cmd.extend(["--model-dir", model_dir])
    env = {**os.environ, "RUST_LOG": "error"}
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env)


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()


def measure(binary: str, model_dir: str | None, long_path: Path, batches: int) -> dict:
    port = free_port()
    proc = start_server(binary, port, pool_size=1, model_dir=model_dir)
    try:
        cold = wait_ready(port)
        rss_ready = rss_mb(proc.pid)

        # Warmup (discard)
        rest_transcribe(port, DEFAULT_WAVS[0].read_bytes())

        short_rows = []
        texts = []
        for wav in DEFAULT_WAVS:
            text, wall = rest_transcribe(port, wav.read_bytes())
            dur = wav_duration_s(wav)
            short_rows.append(
                {
                    "file": wav.name,
                    "audio_duration_sec": round(dur, 3),
                    "processing_sec": round(wall, 4),
                    "rtf": round(wall / dur, 4) if dur > 0 else None,
                    "text": text,
                }
            )
            texts.append(text)

        short_rtfs = [r["rtf"] for r in short_rows if r["rtf"] is not None]
        short_mean = sum(short_rtfs) / len(short_rtfs)
        short_best = min(short_rtfs)
        short_max = max(short_rtfs)

        # RAM snapshot after short warm decodes (matches freeze / edge protocol).
        # Long multi-run is for RTF only — ORT arenas can stick after 40s audio
        # and would unfairly inflate RSS vs the freeze edge-fixture protocol.
        rss_decode = rss_mb(proc.pid)
        resident = footprint_mb(proc.pid)

        long_data = long_path.read_bytes()
        long_dur = wav_duration_s(long_path)
        # One long warmup
        rest_transcribe(port, long_data)
        long_rtfs = []
        long_text = None
        for _ in range(batches):
            text, wall = rest_transcribe(port, long_data)
            if long_text is None:
                long_text = text
            elif text != long_text:
                raise RuntimeError("long-fixture transcript drifted across multi-run batches")
            long_rtfs.append(wall / long_dur)

        rss_after_long = rss_mb(proc.pid)

        model_root = Path(model_dir) if model_dir else Path.home() / ".gigastt" / "models"

        metrics = {
            "warm_rtf_long40_best": round(min(long_rtfs), 4),
            "warm_rtf_long40_mean": round(sum(long_rtfs) / len(long_rtfs), 4),
            "warm_rtf_long40_runs": [round(x, 4) for x in long_rtfs],
            "warm_rtf_long40_audio_s": round(long_dur, 3),
            "warm_rtf_short_fixtures_mean": round(short_mean, 4),
            "warm_rtf_short_fixtures_best": round(short_best, 4),
            "warm_rtf_short_fixtures_max": round(short_max, 4),
            "warm_rtf_multirun_overall_best": round(short_best, 4),
            "warm_rtf_multirun_mean_of_means": round(short_mean, 4),
            "rss_mb_after_ready_pool1": rss_ready,
            "rss_mb_after_decode_pool1": rss_decode,
            "rss_mb_after_long_pool1": rss_after_long,
            "resident_mb_pool1_footprint": resident,
            "cold_start_sec_pool1": round(cold, 3),
            "ttfp_ms": None,
            "lean_disk_mb": lean_disk_mb(model_root),
            "fixture_texts": texts,
            "competitive_bar_rtf": 0.030,
            "stretch_rtf_lo": 0.015,
            "stretch_rtf_hi": 0.020,
            "short_samples": short_rows,
            "long_text_preview": (long_text or "")[:120],
        }
        return {
            "schema": "gigastt.dual_constraint_measure.v1",
            "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "product_path": "lean INT8 ORT rnnt (default; no permanent F32 pack)",
            "binary": binary,
            "protocol": {
                "batches": batches,
                "pool_size": 1,
                "model_variant": "rnnt",
                "punctuation": "off",
                "itn": "off",
                "rss_sample_point": "after_short_fixtures_before_long",
                "primary_rtf": "warm_rtf_long40_best (multi-run BEST); mean must not regress",
                "long_fixture": "golos_00..04 concat x2 (~42s)",
            },
            "metrics": metrics,
        }
    finally:
        stop_server(proc)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", default=str(REPO_ROOT / "target" / "release" / "gigastt"))
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batches", type=int, default=3, help="long-fixture multi-run count")
    ap.add_argument(
        "--long-wav",
        default=None,
        help="optional path for the long fixture; built under output dir if omitted",
    )
    args = ap.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    long_path = Path(args.long_wav) if args.long_wav else out.parent / "long40s.wav"
    if not long_path.exists():
        dur = build_long_wav(DEFAULT_WAVS, long_path, repeats=2)
        print(f"built long fixture {long_path} ({dur:.2f}s)", flush=True)

    result = measure(args.binary, args.model_dir, long_path, args.batches)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    m = result["metrics"]
    print(
        f"long40 best={m['warm_rtf_long40_best']} mean={m['warm_rtf_long40_mean']}  "
        f"short mean={m['warm_rtf_short_fixtures_mean']} best={m['warm_rtf_short_fixtures_best']}  "
        f"RSS decode={m['rss_mb_after_decode_pool1']} resident={m['resident_mb_pool1_footprint']}",
        flush=True,
    )
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
