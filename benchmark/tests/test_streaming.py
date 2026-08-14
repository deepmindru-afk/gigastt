"""Unit tests for the streaming WER / TTFP protocol helpers."""

import asyncio
import json
import struct
from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest

from streaming import (
    STREAM_CHUNK_MS,
    STREAM_SAMPLE_RATE,
    chunk_pcm16,
    energy_onset_s,
    join_transcripts,
    paired_delta,
    percentile,
    should_count_partial,
    stream_session,
    summarize_latency,
)


def _pcm16_silence(n_samples: int) -> bytes:
    return b"\x00\x00" * n_samples


def _pcm16_tone(n_samples: int, amplitude: int = 8000) -> bytes:
    return struct.pack("<" + "h" * n_samples, *([amplitude] * n_samples))


def test_chunk_pcm16_splits_on_100ms_frames():
    # 0.35 s @ 16 kHz = 5600 samples → 3 full 100 ms chunks + 50 ms remainder.
    pcm = _pcm16_silence(5600)
    chunks = chunk_pcm16(pcm, chunk_ms=100)
    assert [len(c) for c in chunks] == [3200, 3200, 3200, 1600]


def test_chunk_pcm16_empty_is_empty():
    assert chunk_pcm16(b"") == []


def test_should_count_partial_skips_empty_and_whitespace():
    assert should_count_partial({"type": "partial", "text": ""}) is False
    assert should_count_partial({"type": "partial", "text": "   "}) is False
    assert should_count_partial({"type": "partial", "text": "привет"}) is True
    assert should_count_partial({"type": "final", "text": "привет"}) is False


def test_join_transcripts_prefers_finals_in_order():
    text = join_transcripts(
        finals=["шестьдесят тысяч", "тенге"],
        last_partial="тенге",
    )
    assert text == "шестьдесят тысяч тенге"


def test_join_transcripts_appends_live_tail_after_final():
    text = join_transcripts(
        finals=["шестьдесят тысяч"],
        last_partial="тенге",
    )
    assert text == "шестьдесят тысяч тенге"


def test_join_transcripts_falls_back_to_last_partial():
    assert join_transcripts(finals=[], last_partial="  привет мир  ") == "привет мир"
    assert join_transcripts(finals=[], last_partial="") == ""


def test_energy_onset_finds_first_speech_frame():
    # 200 ms silence + 100 ms tone @ 16 kHz.
    pcm = _pcm16_silence(3200) + _pcm16_tone(1600)
    onset = energy_onset_s(pcm)
    assert onset is not None
    assert 0.18 <= onset <= 0.22


def test_energy_onset_all_silence_is_none():
    assert energy_onset_s(_pcm16_silence(1600)) is None


def test_percentile_p50_p95():
    xs = [float(i) for i in range(1, 101)]
    assert percentile(xs, 50) == pytest.approx(50.5, abs=0.6)
    assert percentile(xs, 95) == pytest.approx(95.05, abs=1.0)
    assert percentile([], 50) is None


def test_paired_delta_stream_minus_batch():
    batch = [
        {"file": "a.wav", "ref_words": 10, "errors": 1, "failed": False},
        {"file": "b.wav", "ref_words": 10, "errors": 0, "failed": False},
    ]
    stream = [
        {"file": "a.wav", "ref_words": 10, "errors": 2, "failed": False},
        {"file": "b.wav", "ref_words": 10, "errors": 1, "failed": False},
    ]
    delta = paired_delta(batch, stream)
    assert delta["paired"] == 2
    assert delta["wer_batch"] == pytest.approx(5.0)
    assert delta["wer_stream"] == pytest.approx(15.0)
    assert delta["delta_pp"] == pytest.approx(10.0)
    assert delta["ci_low"] <= delta["delta_pp"] <= delta["ci_high"]


def test_paired_delta_skips_unmatched_files():
    batch = [{"file": "a.wav", "ref_words": 4, "errors": 0, "failed": False}]
    stream = [{"file": "b.wav", "ref_words": 4, "errors": 1, "failed": False}]
    delta = paired_delta(batch, stream)
    assert delta["paired"] == 0
    assert delta["delta_pp"] == 0.0


def test_summarize_latency_reports_p50_p95_and_timeouts():
    rows = [
        {"ttfp_ms": 400.0, "ttfs_ms": 200.0, "finalization_lag_ms": 4100.0, "timed_out": False, "no_partial": False},
        {"ttfp_ms": 800.0, "ttfs_ms": 300.0, "finalization_lag_ms": 4200.0, "timed_out": False, "no_partial": False},
        {"ttfp_ms": 1200.0, "ttfs_ms": 400.0, "finalization_lag_ms": 4300.0, "timed_out": False, "no_partial": False},
        {"ttfp_ms": None, "ttfs_ms": None, "finalization_lag_ms": None, "timed_out": True, "no_partial": True},
    ]
    summary = summarize_latency(rows)
    assert summary["n"] == 4
    assert summary["n_timeout"] == 1
    assert summary["n_no_partial"] == 1
    assert summary["n_error"] == 0
    assert summary["ttfp_ms"]["n"] == 3
    assert summary["ttfp_ms"]["p50"] == pytest.approx(800.0, abs=1.0)
    assert summary["ttfp_ms"]["p95"] >= summary["ttfp_ms"]["p50"]
    assert STREAM_CHUNK_MS == 100
    assert STREAM_SAMPLE_RATE == 16000


def test_summarize_latency_drops_negative_ttfs():
    rows = [
        {"ttfp_ms": 800.0, "ttfs_ms": 200.0, "finalization_lag_ms": 4000.0, "timed_out": False, "no_partial": False},
        {"ttfp_ms": 800.0, "ttfs_ms": -300.0, "finalization_lag_ms": 4000.0, "timed_out": False, "no_partial": False},
    ]
    summary = summarize_latency(rows)
    assert summary["ttfs_ms"]["n"] == 1
    assert summary["ttfs_ms"]["p50"] == pytest.approx(200.0, abs=0.1)


class _ScriptedWS:
    """Minimal websockets stand-in: Ready, then scripted replies as audio/Stop arrive."""

    def __init__(self, *, hang_after_stop: bool = False, error_after_audio: bool = False):
        self.sent: list = []
        self._out: asyncio.Queue = asyncio.Queue()
        self._audio = 0
        self._hang_after_stop = hang_after_stop
        self._error_after_audio = error_after_audio
        self._out.put_nowait(
            json.dumps(
                {
                    "type": "ready",
                    "model": "mock",
                    "sample_rate": 16000,
                    "version": "1.0",
                    "max_session_secs": 0,
                    "idle_timeout_secs": 300,
                }
            )
        )

    async def recv(self):
        return await self._out.get()

    async def send(self, data):
        self.sent.append(data)
        if isinstance(data, (bytes, bytearray)):
            self._audio += 1
            if self._audio == 1:
                await self._out.put(json.dumps({"type": "partial", "text": "   "}))
                await self._out.put(json.dumps({"type": "partial", "text": "шестьдесят"}))
            elif self._audio == 2:
                await self._out.put(json.dumps({"type": "final", "text": "шестьдесят тысяч"}))
            if self._error_after_audio and self._audio == 1:
                await self._out.put(json.dumps({"type": "error", "code": "inference_error", "message": "boom"}))
            return
        try:
            obj = json.loads(data)
        except (TypeError, json.JSONDecodeError):
            return
        if obj.get("type") == "stop" and not self._hang_after_stop:
            await self._out.put(json.dumps({"type": "final", "text": "тенге"}))
            await self._out.put(None)

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self._out.get()
        if item is None:
            raise StopAsyncIteration
        return item

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _patch_connect(ws: _ScriptedWS):
    @asynccontextmanager
    async def _connect(*_args, **_kwargs):
        yield ws

    return _connect


def test_stream_session_collects_midstream_and_stop_finals():
    ws = _ScriptedWS()
    pcm = _pcm16_tone(1600 * 3)

    with patch("websockets.connect", _patch_connect(ws)):
        result = asyncio.run(stream_session("ws://127.0.0.1:9/v1/ws", pcm, pace=False, timeout_s=2.0))

    assert result["text"] == "шестьдесят тысяч тенге"
    assert result["empty_partials_skipped"] == 1
    assert result["ttfp_ms"] is not None
    assert result["timed_out"] is False
    assert result["no_partial"] is False
    assert any(isinstance(s, (bytes, bytearray)) for s in ws.sent)
    stop_msgs = [json.loads(s) for s in ws.sent if isinstance(s, str)]
    assert {"type": "configure", "sample_rate": 16000} in stop_msgs
    assert {"type": "stop"} in stop_msgs


def test_stream_session_timeout_keeps_clip_in_corpus():
    ws = _ScriptedWS(hang_after_stop=True)
    pcm = _pcm16_tone(1600)

    with patch("websockets.connect", _patch_connect(ws)):
        result = asyncio.run(stream_session("ws://127.0.0.1:9/v1/ws", pcm, pace=False, timeout_s=0.05))

    assert result["timed_out"] is True
    assert result["text"] == "шестьдесят"
    assert result["no_partial"] is False


def test_stream_session_ready_timeout_raises():
    class _HangReady:
        async def recv(self):
            await asyncio.sleep(10)

        async def send(self, data):
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    @asynccontextmanager
    async def _connect(*_args, **_kwargs):
        yield _HangReady()

    with patch("websockets.connect", _connect):
        with pytest.raises(RuntimeError, match="Ready"):
            asyncio.run(
                stream_session("ws://127.0.0.1:9/v1/ws", _pcm16_tone(1600), pace=False, timeout_s=0.05)
            )


def test_stream_session_treats_unclean_close_as_end():
    """Server Stop path often drops TCP without a close frame."""
    from websockets.exceptions import ConnectionClosedError

    class _CloseAfterStop(_ScriptedWS):
        async def __anext__(self):
            item = await self._out.get()
            if item is None:
                raise ConnectionClosedError(None, None)
            return item

    ws = _CloseAfterStop()
    pcm = _pcm16_tone(1600 * 3)

    with patch("websockets.connect", _patch_connect(ws)):
        result = asyncio.run(stream_session("ws://127.0.0.1:9/v1/ws", pcm, pace=False, timeout_s=2.0))

    assert result["text"] == "шестьдесят тысяч тенге"
    assert result["timed_out"] is False


def test_stream_session_error_raises():
    ws = _ScriptedWS(error_after_audio=True)
    pcm = _pcm16_tone(1600)

    with patch("websockets.connect", _patch_connect(ws)):
        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(stream_session("ws://127.0.0.1:9/v1/ws", pcm, pace=False, timeout_s=1.0))
