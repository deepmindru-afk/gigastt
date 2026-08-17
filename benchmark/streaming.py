"""Shared WebSocket streaming protocol for WER and TTFP measurement.

Canonical knobs (do not change without bumping STREAM_PROTOCOL_VERSION and
rewriting the tables in docs/benchmarks.md):

- 16 kHz mono PCM16 over `/v1/ws`
- `configure` with `sample_rate=16000` after Ready, before any audio
- real-time paced `chunk_ms=100` frames
- TTFP clock starts on the first audio frame (after Ready + configure)
- empty / whitespace `partial`s are ignored
- every `final` is kept until the socket closes (mid-stream + Stop flush)
- a clip with no counted partial before timeout stays in the corpus as `no_partial`
"""

from __future__ import annotations

import asyncio
import json
import struct
import wave
from pathlib import Path
from typing import Any, Optional

STREAM_SAMPLE_RATE = 16000
STREAM_CHUNK_MS = 100
STREAM_PROTOCOL_VERSION = "1.0"


def chunk_pcm16(
    pcm: bytes,
    chunk_ms: int = STREAM_CHUNK_MS,
    sample_rate: int = STREAM_SAMPLE_RATE,
) -> list[bytes]:
    """Split mono PCM16 into `chunk_ms` frames (last frame may be short)."""
    if not pcm:
        return []
    frame_bytes = max(1, int(sample_rate * chunk_ms / 1000) * 2)
    return [pcm[i : i + frame_bytes] for i in range(0, len(pcm), frame_bytes)]


def should_count_partial(msg: dict) -> bool:
    """True for a non-empty `partial`. Empty partials do not start the TTFP clock."""
    if msg.get("type") != "partial":
        return False
    return bool(str(msg.get("text") or "").strip())


def join_transcripts(*, finals: list[str], last_partial: str) -> str:
    """Committed streaming text: finals in order, plus a live tail if it is new.

    After a `final`, the session sets `last_partial` to that same text, so the
    tail is not double-counted. A later non-empty partial is the next
    utterance and must be kept if Stop never flushed it (timeout).
    """
    parts = [t.strip() for t in finals if t and t.strip()]
    tail = (last_partial or "").strip()
    if parts:
        if tail and tail != parts[-1]:
            parts.append(tail)
        return " ".join(parts)
    return tail


def energy_onset_s(
    pcm: bytes,
    sample_rate: int = STREAM_SAMPLE_RATE,
    frame_ms: int = 10,
    rel_threshold: float = 0.02,
    abs_threshold: float = 200.0,
) -> Optional[float]:
    """First frame whose RMS exceeds a fraction of the clip peak (or `abs_threshold`).

    Used for TTFS (time-to-first-partial after speech onset) without a VAD model.
    All-silence clips return None.
    """
    n = len(pcm) // 2
    if n == 0:
        return None
    samples = struct.unpack("<" + "h" * n, pcm[: n * 2])
    peak = max(abs(s) for s in samples)
    if peak == 0:
        return None
    thresh = max(abs_threshold, rel_threshold * peak)
    hop = max(1, int(sample_rate * frame_ms / 1000))
    for start in range(0, n, hop):
        window = samples[start : start + hop]
        if not window:
            continue
        mean_sq = sum(s * s for s in window) / len(window)
        rms = mean_sq**0.5
        if rms >= thresh:
            return start / sample_rate
    return None


def percentile(xs: list[float], p: float) -> Optional[float]:
    """Linear-interpolation percentile. `p` is in [0, 100]. Empty → None."""
    if not xs:
        return None
    ordered = sorted(xs)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (p / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def paired_delta(batch_details: list[dict], stream_details: list[dict]) -> dict:
    """WER_stream − WER_batch on files present in both runs, plus bootstrap CI on Δ."""
    by_batch = {d["file"]: d for d in batch_details}
    pairs: list[tuple[int, int, int]] = []
    for s in stream_details:
        b = by_batch.get(s["file"])
        if b is None:
            continue
        ref = int(s.get("ref_words") or b.get("ref_words") or 0)
        pairs.append((ref, int(b.get("errors") or 0), int(s.get("errors") or 0)))

    if not pairs:
        return {
            "paired": 0,
            "wer_batch": 0.0,
            "wer_stream": 0.0,
            "delta_pp": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
        }

    ref_sum = sum(p[0] for p in pairs)
    batch_err = sum(p[1] for p in pairs)
    stream_err = sum(p[2] for p in pairs)
    wer_batch = (batch_err / ref_sum * 100.0) if ref_sum else 0.0
    wer_stream = (stream_err / ref_sum * 100.0) if ref_sum else 0.0
    delta = wer_stream - wer_batch
    lo, hi = _bootstrap_delta_ci(pairs)
    return {
        "paired": len(pairs),
        "wer_batch": round(wer_batch, 2),
        "wer_stream": round(wer_stream, 2),
        "delta_pp": round(delta, 2),
        "ci_low": round(lo, 2),
        "ci_high": round(hi, 2),
    }


def _bootstrap_delta_ci(
    pairs: list[tuple[int, int, int]],
    iterations: int = 1000,
) -> tuple[float, float]:
    """95% CI for (WER_stream − WER_batch) by resampling paired clips."""
    n = len(pairs)
    rng = 123456789
    mask = (1 << 64) - 1
    deltas: list[float] = []
    for _ in range(iterations):
        ref_sum = 0
        batch_err = 0
        stream_err = 0
        for _ in range(n):
            rng = (rng * 6364136223846793005 + 1) & mask
            idx = (rng >> 32) % n
            ref, b_err, s_err = pairs[idx]
            ref_sum += ref
            batch_err += b_err
            stream_err += s_err
        if ref_sum == 0:
            deltas.append(0.0)
        else:
            deltas.append((stream_err / ref_sum - batch_err / ref_sum) * 100.0)
    deltas.sort()
    return deltas[(iterations * 25) // 1000], deltas[(iterations * 975) // 1000]


def summarize_latency(rows: list[dict]) -> dict:
    """Corpus rollup: p50/p95 TTFP/TTFS/finalization, timeout counts."""
    ttfp = [r["ttfp_ms"] for r in rows if r.get("ttfp_ms") is not None]
    ttfs = [
        r["ttfs_ms"]
        for r in rows
        if r.get("ttfs_ms") is not None and r["ttfs_ms"] >= 0
    ]
    fin = [r["finalization_lag_ms"] for r in rows if r.get("finalization_lag_ms") is not None]
    lags: list[float] = []
    for r in rows:
        lags.extend(r.get("partial_lags_ms") or [])

    def _block(xs: list[float]) -> dict:
        return {
            "n": len(xs),
            "p50": None if not xs else round(percentile(xs, 50) or 0.0, 1),
            "p95": None if not xs else round(percentile(xs, 95) or 0.0, 1),
            "max": None if not xs else round(max(xs), 1),
        }

    return {
        "n": len(rows),
        "n_timeout": sum(1 for r in rows if r.get("timed_out")),
        "n_no_partial": sum(1 for r in rows if r.get("no_partial")),
        "n_error": sum(
            1 for r in rows if r.get("ok") is False and not r.get("timed_out")
        ),
        "ttfp_ms": _block(ttfp),
        "ttfs_ms": _block(ttfs),
        "finalization_lag_ms": _block(fin),
        "partial_lag_ms": _block(lags),
        "protocol": {
            "sample_rate": STREAM_SAMPLE_RATE,
            "chunk_ms": STREAM_CHUNK_MS,
            "version": STREAM_PROTOCOL_VERSION,
        },
    }


def load_pcm16_16k(path: str) -> tuple[bytes, float]:
    """Load a WAV as mono PCM16 @ 16 kHz. Returns `(pcm, duration_s)`."""
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)
    if channels == 1 and width == 2 and rate == STREAM_SAMPLE_RATE:
        return raw, nframes / float(STREAM_SAMPLE_RATE)
    return _resample_to_pcm16_16k(path)


def _resample_to_pcm16_16k(path: str) -> tuple[bytes, float]:
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as e:
        raise ValueError(
            f"{path} is not 16 kHz mono PCM16; install soundfile+numpy to resample"
        ) from e
    samples, rate = sf.read(path, always_2d=True, dtype="float32")
    mono = samples.mean(axis=1)
    if rate != STREAM_SAMPLE_RATE and len(mono) > 1:
        n_out = max(1, int(round(len(mono) * STREAM_SAMPLE_RATE / rate)))
        x = np.linspace(0.0, 1.0, num=len(mono), endpoint=False)
        xi = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        mono = np.interp(xi, x, mono)
    clipped = np.clip(mono, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2").tobytes()
    return pcm, len(clipped) / float(STREAM_SAMPLE_RATE)


async def stream_session(
    url: str,
    pcm: bytes,
    *,
    chunk_ms: int = STREAM_CHUNK_MS,
    pace: bool = True,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """One WS session: Ready → configure → paced PCM → Stop → collect finals.

    Returns hypothesis text plus latency fields. Does not raise on a missing
    final: `timed_out` / `no_partial` are set instead so the clip stays in p95.
    """
    import websockets
    from websockets.exceptions import ConnectionClosed

    chunks = chunk_pcm16(pcm, chunk_ms=chunk_ms)
    duration_s = (len(pcm) // 2) / float(STREAM_SAMPLE_RATE)
    onset_s = energy_onset_s(pcm)

    first_partial_at: Optional[float] = None
    final_at: Optional[float] = None
    last_sent_at: Optional[float] = None
    partial_lags: list[float] = []
    finals: list[str] = []
    last_partial = ""
    empty_skipped = 0
    timed_out = False
    started_at: Optional[float] = None
    send_done_at: Optional[float] = None

    try:
        async with websockets.connect(url, open_timeout=min(10.0, timeout_s)) as ws:
            try:
                ready_raw = await asyncio.wait_for(ws.recv(), timeout=min(10.0, timeout_s))
            except asyncio.TimeoutError as e:
                raise RuntimeError("timed out waiting for Ready") from e
            ready = json.loads(ready_raw if isinstance(ready_raw, str) else ready_raw.decode())
            if ready.get("type") != "ready":
                raise RuntimeError(f"expected Ready, got {ready!r}")
            await ws.send(json.dumps({"type": "configure", "sample_rate": STREAM_SAMPLE_RATE}))

            async def _read() -> None:
                # Mid-stream finals are committed utterances (VAD / blank). Stop
                # then emits one more Final (possibly empty) and closes. Collect
                # every final until the socket ends — returning on the first one
                # drops the rest of the clip from WER. The server Breaks after
                # the Stop Final and may drop TCP without a close frame.
                nonlocal first_partial_at, final_at, last_partial, empty_skipped
                try:
                    async for raw in ws:
                        now = asyncio.get_running_loop().time()
                        if isinstance(raw, bytes):
                            try:
                                raw = raw.decode()
                            except UnicodeDecodeError:
                                continue
                        try:
                            obj = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        kind = obj.get("type")
                        if kind == "partial":
                            if not should_count_partial(obj):
                                empty_skipped += 1
                                continue
                            last_partial = str(obj.get("text") or "")
                            if last_sent_at is not None:
                                partial_lags.append(now - last_sent_at)
                            if first_partial_at is None:
                                first_partial_at = now
                        elif kind == "final":
                            text = str(obj.get("text") or "").strip()
                            if text:
                                finals.append(text)
                                last_partial = text
                            final_at = now
                        elif kind == "error":
                            raise RuntimeError(
                                obj.get("message") or obj.get("code") or "ws error"
                            )
                except ConnectionClosed:
                    return

            reader = asyncio.create_task(_read())
            loop = asyncio.get_running_loop()
            started_at = loop.time()
            last_sent_at = started_at
            try:
                for chunk in chunks:
                    await ws.send(chunk)
                    last_sent_at = loop.time()
                    if pace:
                        await asyncio.sleep(chunk_ms / 1000.0)
                await ws.send(json.dumps({"type": "stop"}))
                await asyncio.wait_for(reader, timeout=timeout_s)
            except asyncio.TimeoutError:
                timed_out = True
                reader.cancel()
                try:
                    await reader
                except (asyncio.CancelledError, ConnectionClosed):
                    pass
            except Exception:
                reader.cancel()
                try:
                    await reader
                except (asyncio.CancelledError, ConnectionClosed):
                    pass
                raise
            send_done_at = loop.time()
    except ConnectionClosed:
        if started_at is None:
            raise

    ttfp_ms = (
        round((first_partial_at - started_at) * 1000.0, 1)
        if first_partial_at is not None and started_at is not None
        else None
    )
    ttfs_ms = None
    if ttfp_ms is not None and onset_s is not None:
        raw_ttfs = ttfp_ms - onset_s * 1000.0
        # Energy onset after the first partial (common on far-field / noise)
        # is not a latency; leave TTFS unset instead of publishing a negative.
        if raw_ttfs >= 0:
            ttfs_ms = round(raw_ttfs, 1)
    no_partial = first_partial_at is None
    return {
        "text": join_transcripts(finals=finals, last_partial=last_partial),
        "ttfp_ms": ttfp_ms,
        "ttfs_ms": ttfs_ms,
        "finalization_lag_ms": (
            round((final_at - started_at) * 1000.0, 1)
            if final_at is not None and started_at is not None
            else None
        ),
        "audio_duration_ms": round(duration_s * 1000.0, 1),
        "total_audio_sent_ms": (
            round((send_done_at - started_at) * 1000.0, 1) if started_at is not None else None
        ),
        "partial_count": len(partial_lags) if first_partial_at is None else max(len(partial_lags), 1),
        "empty_partials_skipped": empty_skipped,
        "partial_lags_ms": [round(x * 1000.0, 1) for x in partial_lags],
        "onset_s": None if onset_s is None else round(onset_s, 3),
        "timed_out": timed_out,
        "no_partial": no_partial,
    }


def transcribe_ws(
    wav_path: str,
    *,
    port: int = 9877,
    chunk_ms: int = STREAM_CHUNK_MS,
    pace: bool = True,
) -> tuple[str, float, dict[str, Any]]:
    """Sync wrapper: `(hypothesis, wall_s, session_metrics)`."""
    pcm, duration_s = load_pcm16_16k(wav_path)
    url = f"ws://127.0.0.1:{port}/v1/ws"
    timeout_s = max(30.0, duration_s + 30.0)
    import time

    start = time.perf_counter()
    session = asyncio.run(
        stream_session(url, pcm, chunk_ms=chunk_ms, pace=pace, timeout_s=timeout_s)
    )
    elapsed = time.perf_counter() - start
    if session["no_partial"] and not session["text"]:
        raise RuntimeError(f"no streaming transcript for {Path(wav_path).name}")
    return session["text"], elapsed, session
