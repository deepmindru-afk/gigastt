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
        first_partial_at = None
        final_at = None
        started_at = time.perf_counter()
        async with websockets.connect(self.url) as ws:
            # Consume the ready message and tell the server we are streaming 16kHz PCM.
            await ws.recv()
            await ws.send(json.dumps({"type": "configure", "sample_rate": 16000}))

            with wave.open(wav_path, "rb") as wf:
                channels = wf.getnchannels()
                width = wf.getsampwidth()
                rate = wf.getframerate()
                if channels != 1 or width != 2 or rate != 16000:
                    raise ValueError("gigastt WebSocket streaming expects 16kHz mono 16-bit WAV")
                frames_per_chunk = int(rate * chunk_ms / 1000)
                while True:
                    data = wf.readframes(frames_per_chunk)
                    if not data:
                        break
                    await ws.send(data)
                    await asyncio.sleep(chunk_ms / 1000.0)
                await ws.send(json.dumps({"type": "stop"}))
                async for msg in ws:
                    obj = json.loads(msg)
                    if obj.get("type") == "partial" and first_partial_at is None:
                        first_partial_at = time.perf_counter()
                    if obj.get("type") == "final":
                        final_at = time.perf_counter()
                        break
        return {
            "time_to_first_partial_ms": round((first_partial_at - started_at) * 1000, 1) if first_partial_at else None,
            "finalization_lag_ms": round((final_at - started_at) * 1000, 1) if final_at else None,
        }


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
