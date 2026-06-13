#!/usr/bin/env python3
"""Measure streaming latency: time-to-first-partial and finalization lag."""

import argparse
import asyncio
import json
import time
import wave
from pathlib import Path

import websockets


class GigasttLatencyClient:
    def __init__(self, port: int = 9877):
        self.port = port
        self.url = f"ws://127.0.0.1:{port}/v1/ws"

    async def measure(self, wav_path: str, chunk_ms: int = 100) -> dict:
        # Pre-load and validate the clip up front so the sender just paces already
        # decoded PCM frames out over the socket while the reader runs concurrently.
        with wave.open(wav_path, "rb") as wf:
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            rate = wf.getframerate()
            if channels != 1 or width != 2 or rate != 16000:
                raise ValueError("gigastt WebSocket streaming expects 16kHz mono 16-bit WAV")
            frames_per_chunk = int(rate * chunk_ms / 1000)
            audio_duration_ms = wf.getnframes() / rate * 1000.0
            chunks = []
            while True:
                data = wf.readframes(frames_per_chunk)
                if not data:
                    break
                chunks.append(data)

        first_partial_at = None
        final_at = None
        # Wall-clock when the most recent audio chunk left the client. Because the sender is
        # real-time paced, it also marks that audio's position in the stream, so a partial's
        # delay relative to it approximates the server's per-chunk response (compute) lag.
        last_sent_at = None
        partial_lags = []  # seconds: (partial arrival - last_sent_at), one per partial

        async with websockets.connect(self.url) as ws:
            # Consume the ready message and tell the server we are streaming 16kHz PCM.
            await ws.recv()
            await ws.send(json.dumps({"type": "configure", "sample_rate": 16000}))

            async def _read_loop():
                # Runs concurrently with the sender, so a partial emitted mid-stream is
                # timestamped when it actually arrives — not after the whole clip is sent.
                nonlocal first_partial_at, final_at
                async for msg in ws:
                    now = time.perf_counter()
                    obj = json.loads(msg)
                    kind = obj.get("type")
                    if kind == "partial":
                        if last_sent_at is not None:
                            partial_lags.append(now - last_sent_at)
                        if first_partial_at is None:
                            first_partial_at = now
                    elif kind == "final":
                        final_at = now
                        return

            # Start the reader before any audio leaves the client, then stamp started_at
            # on the first chunk (handshake + configure already done) so TTFP is measured
            # against the audio stream, not the connection setup.
            reader = asyncio.create_task(_read_loop())
            started_at = time.perf_counter()
            last_sent_at = started_at
            for data in chunks:
                await ws.send(data)
                last_sent_at = time.perf_counter()
                await asyncio.sleep(chunk_ms / 1000.0)
            await ws.send(json.dumps({"type": "stop"}))
            send_done_at = time.perf_counter()

            # The sender is real-time paced (~clip length); give the reader a bounded
            # window after `stop` to observe the final segment, then stop waiting.
            try:
                await asyncio.wait_for(reader, timeout=30.0)
            except asyncio.TimeoutError:
                reader.cancel()

        ttfp_ms = round((first_partial_at - started_at) * 1000, 1) if first_partial_at else None
        result = {
            "time_to_first_partial_ms": ttfp_ms,
            "first_partial_after_audio_ms": ttfp_ms,
            "finalization_lag_ms": round((final_at - started_at) * 1000, 1) if final_at else None,
            "audio_duration_ms": round(audio_duration_ms, 1),
            "total_audio_sent_ms": round((send_done_at - started_at) * 1000, 1),
        }
        # Per-chunk server response lag: delay of each partial relative to the most recently
        # sent audio chunk. Isolates compute (+queue) latency from real-time pacing and from
        # where the first word happens to fall in the clip — this is the number comparable to
        # "incremental streaming latency". NOTE: it is an UPPER-bounded approximation and is
        # under-estimated when per-chunk compute >= chunk_ms (a newer chunk is sent before the
        # prior partial arrives, resetting last_sent_at); cross-check against the server log's
        # `encoder_inference elapsed_ms`.
        if partial_lags:
            ordered = sorted(partial_lags)
            n = len(ordered)
            result["partial_response_lag_ms"] = {
                "count": n,
                "min": round(ordered[0] * 1000, 1),
                "median": round(ordered[n // 2] * 1000, 1),
                "max": round(ordered[-1] * 1000, 1),
            }
        else:
            result["partial_response_lag_ms"] = None
        return result


def evaluate_gigastt(wav_path: str, port: int = 9877) -> dict:
    return asyncio.run(GigasttLatencyClient(port).measure(wav_path))


def main():
    parser = argparse.ArgumentParser(description="Streaming latency benchmark")
    parser.add_argument("--wav", required=True, help="16kHz mono 16-bit WAV file")
    parser.add_argument("--output", default="results_latency.json")
    parser.add_argument("--port", type=int, default=9877)
    args = parser.parse_args()

    result = evaluate_gigastt(args.wav, port=args.port)
    result["engine"] = "gigastt"
    result["wav"] = args.wav

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
