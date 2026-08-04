#!/usr/bin/env python3
"""Edge / Raspberry Pi–oriented footprint + RTF + TTFP harness for gigastt.

Collects host metadata and measures, for each selected model variant:

  - cold-start wall time (process start → HTTP 200 on /ready)
  - peak RSS after ready and after a warm decode
  - warm RTF on a small set of in-tree WAV fixtures (REST /v1/transcribe)
  - streaming TTFP on golos_00.wav (WebSocket, real-time paced) when
    the ``websockets`` package is available

This does **not** invent numbers. Run it on the target board and paste the
JSON into docs (see ``specs/edge-raspberry-pi-roadmap.md``).

Example::

    python3 benchmark/bench_edge.py \\
      --binary ./target/release/gigastt \\
      --variants rnnt \\
      --storage-label unknown \\
      --output benchmark/results_edge.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIXTURES = REPO_ROOT / "crates" / "gigastt" / "tests" / "fixtures"
DEFAULT_WAVS = [DEFAULT_FIXTURES / f"golos_{i:02d}.wav" for i in range(5)]
DEFAULT_STREAM_WAV = DEFAULT_FIXTURES / "golos_00.wav"


def _read_file(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def collect_host_metadata(storage_label: str) -> dict:
    uname = platform.uname()
    meta: dict = {
        "system": uname.system,
        "node": uname.node,
        "release": uname.release,
        "machine": uname.machine,
        "processor": uname.processor,
        "python": sys.version.split()[0],
        "storage_label": storage_label,
        "cpu_count_logical": os.cpu_count(),
    }

    # Raspberry Pi / Linux device-tree model string when present.
    for p in (
        Path("/proc/device-tree/model"),
        Path("/sys/firmware/devicetree/base/model"),
    ):
        model = _read_file(p)
        if model:
            meta["device_tree_model"] = model.replace("\x00", "").strip()
            break

    meminfo = _read_file(Path("/proc/meminfo"))
    if meminfo:
        for line in meminfo.splitlines():
            if line.startswith("MemTotal:"):
                # kB on Linux
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    meta["mem_total_mb"] = round(int(parts[1]) / 1024, 1)
                break

    if "mem_total_mb" not in meta and sys.platform == "darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            meta["mem_total_mb"] = round(int(out) / (1024 * 1024), 1)
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
            pass

    return meta


def rss_mb(pid: int) -> float | None:
    """Best-effort resident set size in MiB for *pid*."""
    try:
        import psutil  # type: ignore

        return round(psutil.Process(pid).memory_info().rss / (1024 * 1024), 1)
    except Exception:
        pass

    # Linux smaps / status
    status = _read_file(Path(f"/proc/{pid}/status"))
    if status:
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    return round(int(parts[1]) / 1024, 1)

    # macOS: ps
    if sys.platform == "darwin":
        try:
            out = subprocess.check_output(
                ["ps", "-o", "rss=", "-p", str(pid)], text=True
            ).strip()
            # ps rss is KiB on macOS
            return round(int(out) / 1024, 1)
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
            return None
    return None


def wav_duration_s(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def wait_ready(port: int, timeout_s: float = 180.0) -> float:
    """Poll /ready; return seconds from call start until first 200."""
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


def rest_transcribe(port: int, wav: Path, timeout_s: float = 600.0) -> tuple[str, float]:
    data = wav.read_bytes()
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


async def measure_ttfp(port: int, wav_path: Path, chunk_ms: int = 100) -> dict:
    """Real-time-paced WS TTFP; requires the ``websockets`` package."""
    try:
        import websockets  # type: ignore
    except ImportError:
        return {"error": "websockets package not installed; skip TTFP"}

    with wave.open(str(wav_path), "rb") as wf:
        channels, width, rate = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
        if channels != 1 or width != 2 or rate != 16000:
            return {
                "error": (
                    f"TTFP fixture must be 16 kHz mono PCM16 WAV "
                    f"(got ch={channels} width={width} rate={rate})"
                )
            }
        frames_per_chunk = int(rate * chunk_ms / 1000)
        audio_duration_ms = wf.getnframes() / rate * 1000.0
        chunks: list[bytes] = []
        while True:
            data = wf.readframes(frames_per_chunk)
            if not data:
                break
            chunks.append(data)

    first_partial_at = None
    final_at = None
    url = f"ws://127.0.0.1:{port}/v1/ws"

    async with websockets.connect(url) as ws:
        await ws.recv()  # ready
        await ws.send(json.dumps({"type": "configure", "sample_rate": 16000}))

        async def _read_loop():
            nonlocal first_partial_at, final_at
            async for msg in ws:
                now = time.perf_counter()
                obj = json.loads(msg)
                kind = obj.get("type")
                if kind == "partial" and first_partial_at is None:
                    first_partial_at = now
                elif kind == "final":
                    final_at = now
                    return

        reader = asyncio.create_task(_read_loop())
        started_at = time.perf_counter()
        for data in chunks:
            await ws.send(data)
            await asyncio.sleep(chunk_ms / 1000.0)
        await ws.send(json.dumps({"type": "stop"}))
        try:
            await asyncio.wait_for(reader, timeout=60.0)
        except asyncio.TimeoutError:
            reader.cancel()

    ttfp_ms = (
        round((first_partial_at - started_at) * 1000, 1) if first_partial_at else None
    )
    return {
        "wav": str(wav_path),
        "audio_duration_ms": round(audio_duration_ms, 1),
        "time_to_first_partial_ms": ttfp_ms,
        "finalization_lag_ms": (
            round((final_at - started_at) * 1000, 1) if final_at else None
        ),
        "note": (
            "Buffered/chunked streaming over offline decode; TTFP is not a "
            "sub-200ms claim. Compare against M1 baseline in docs/benchmarks.md."
        ),
    }


def find_binary(explicit: str | None) -> str:
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    release = REPO_ROOT / "target" / "release" / "gigastt"
    candidates.append(str(release))
    which = shutil.which("gigastt")
    if which:
        candidates.append(which)
    for c in candidates:
        try:
            subprocess.run([c, "--version"], capture_output=True, check=True)
            return c
        except Exception:
            continue
    raise SystemExit(
        "gigastt binary not found. Pass --binary or build target/release/gigastt."
    )


def start_server(
    binary: str,
    port: int,
    model_variant: str,
    pool_size: int,
    encoder_intra_threads: int | None,
    model_dir: str | None,
) -> subprocess.Popen:
    cmd = [
        binary,
        "serve",
        "--port",
        str(port),
        "--pool-size",
        str(pool_size),
        "--model-variant",
        model_variant,
        # Edge profile: no extra models in the critical path for this bench.
        "--punctuation",
        "off",
        "--itn",
        "off",
    ]
    if encoder_intra_threads is not None:
        cmd.extend(["--encoder-intra-threads", str(encoder_intra_threads)])
    if model_dir:
        cmd.extend(["--model-dir", model_dir])

    env = {**os.environ, "RUST_LOG": "error"}
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env=env,
    )


def stop_server(proc: subprocess.Popen) -> str:
    stderr = ""
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            _, err = proc.communicate(timeout=15)
            stderr = (err or b"").decode("utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            proc.kill()
            _, err = proc.communicate()
            stderr = (err or b"").decode("utf-8", errors="replace")
    else:
        _, err = proc.communicate()
        stderr = (err or b"").decode("utf-8", errors="replace")
    return stderr


def measure_variant(
    *,
    binary: str,
    variant: str,
    port: int,
    pool_size: int,
    encoder_intra_threads: int | None,
    model_dir: str | None,
    wavs: list[Path],
    stream_wav: Path,
    skip_ttfp: bool,
) -> dict:
    print(f"\n=== variant={variant} pool_size={pool_size} ===", flush=True)
    result: dict = {
        "model_variant": variant,
        "pool_size": pool_size,
        "encoder_intra_threads": encoder_intra_threads,
        "binary": binary,
    }

    proc = start_server(
        binary, port, variant, pool_size, encoder_intra_threads, model_dir
    )
    try:
        try:
            cold = wait_ready(port)
        except RuntimeError as e:
            stderr = stop_server(proc)
            result["error"] = str(e)
            result["server_stderr_tail"] = stderr[-2000:]
            print(f"  FAIL ready: {e}", flush=True)
            if stderr:
                print(f"  stderr tail:\n{stderr[-500:]}", flush=True)
            return result

        result["cold_start_sec"] = round(cold, 3)
        result["rss_mb_after_ready"] = rss_mb(proc.pid)
        print(
            f"  cold-start {result['cold_start_sec']}s  "
            f"RSS@ready {result['rss_mb_after_ready']} MiB",
            flush=True,
        )

        # One discard warm-up (not included in RTF mean).
        try:
            rest_transcribe(port, wavs[0])
        except Exception as e:
            result["warmup_error"] = str(e)
            print(f"  warmup failed: {e}", flush=True)

        samples = []
        for wav in wavs:
            try:
                text, elapsed = rest_transcribe(port, wav)
                dur = wav_duration_s(wav)
                rtf = elapsed / dur if dur > 0 else None
                row = {
                    "file": wav.name,
                    "audio_duration_sec": round(dur, 3),
                    "processing_sec": round(elapsed, 3),
                    "rtf": round(rtf, 4) if rtf is not None else None,
                    "text_preview": text[:80],
                }
                samples.append(row)
                print(
                    f"  {wav.name}: RTF={row['rtf']}  "
                    f"({row['processing_sec']}s / {row['audio_duration_sec']}s audio)",
                    flush=True,
                )
            except Exception as e:
                samples.append({"file": wav.name, "error": str(e)})
                print(f"  {wav.name}: ERROR {e}", flush=True)

        result["samples"] = samples
        rtfs = [s["rtf"] for s in samples if isinstance(s.get("rtf"), (int, float))]
        if rtfs:
            result["rtf_mean"] = round(sum(rtfs) / len(rtfs), 4)
            result["rtf_min"] = round(min(rtfs), 4)
            result["rtf_max"] = round(max(rtfs), 4)
            print(
                f"  RTF mean={result['rtf_mean']}  "
                f"min={result['rtf_min']} max={result['rtf_max']}",
                flush=True,
            )

        result["rss_mb_after_decode"] = rss_mb(proc.pid)
        print(f"  RSS@decode {result['rss_mb_after_decode']} MiB", flush=True)

        if skip_ttfp:
            result["ttfp"] = {"skipped": True}
        else:
            print("  measuring TTFP (WS, real-time paced)…", flush=True)
            result["ttfp"] = asyncio.run(measure_ttfp(port, stream_wav))
            print(f"  TTFP: {result['ttfp']}", flush=True)
    finally:
        stop_server(proc)

    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Edge / Pi RTF + footprint + TTFP harness")
    p.add_argument("--binary", default=None, help="Path to gigastt binary")
    p.add_argument(
        "--variants",
        default="rnnt",
        help="Comma-separated model variants (e.g. rnnt,ml_ctc)",
    )
    p.add_argument("--pool-size", type=int, default=1)
    p.add_argument(
        "--encoder-intra-threads",
        type=int,
        default=None,
        help="If set, pass --encoder-intra-threads N to serve",
    )
    p.add_argument("--model-dir", default=None)
    p.add_argument("--port", type=int, default=9877)
    p.add_argument(
        "--storage-label",
        default="unknown",
        choices=["microSD", "usb-ssd", "nvme", "unknown"],
        help="Where the model dir lives (for cold-start comparison)",
    )
    p.add_argument(
        "--wavs",
        default=None,
        help="Comma-separated WAV paths (default: golos_00..04 fixtures)",
    )
    p.add_argument(
        "--stream-wav",
        default=None,
        help="WAV for TTFP (default: golos_00.wav fixture)",
    )
    p.add_argument("--skip-ttfp", action="store_true")
    p.add_argument(
        "--output",
        default=str(Path(__file__).parent / "results_edge.json"),
        help="JSON output path",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    binary = find_binary(args.binary)
    try:
        ver = subprocess.check_output([binary, "--version"], text=True).strip()
    except subprocess.CalledProcessError:
        ver = "unknown"

    if args.wavs:
        wavs = [Path(x) for x in args.wavs.split(",")]
    else:
        wavs = list(DEFAULT_WAVS)
    for w in wavs:
        if not w.is_file():
            raise SystemExit(f"WAV not found: {w}")

    stream_wav = Path(args.stream_wav) if args.stream_wav else DEFAULT_STREAM_WAV
    if not args.skip_ttfp and not stream_wav.is_file():
        raise SystemExit(f"stream WAV not found: {stream_wav}")

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    host = collect_host_metadata(args.storage_label)
    host["gigastt_version"] = ver
    host["binary"] = binary

    print("Host:", json.dumps(host, ensure_ascii=False, indent=2), flush=True)

    runs = []
    for i, variant in enumerate(variants):
        port = args.port + i  # avoid TIME_WAIT collisions when chaining variants
        runs.append(
            measure_variant(
                binary=binary,
                variant=variant,
                port=port,
                pool_size=args.pool_size,
                encoder_intra_threads=args.encoder_intra_threads,
                model_dir=args.model_dir,
                wavs=wavs,
                stream_wav=stream_wav,
                skip_ttfp=args.skip_ttfp,
            )
        )

    payload = {
        "schema": "gigastt.edge_bench.v1",
        "protocol": "specs/edge-raspberry-pi-roadmap.md",
        "host": host,
        "runs": runs,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out}", flush=True)

    # Non-zero if every variant failed to start.
    if runs and all("error" in r for r in runs):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
