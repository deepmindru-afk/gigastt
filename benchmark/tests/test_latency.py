"""Unit tests for benchmark_latency.py CLI and corpus rollup."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import benchmark_latency


def test_main_requires_wav_xor_dataset(tmp_path, capsys):
    with patch("sys.argv", ["benchmark_latency.py"]):
        with pytest.raises(SystemExit):
            benchmark_latency.main()
    err = capsys.readouterr().err
    assert "--wav" in err and "--dataset" in err

    with patch("sys.argv", [
        "benchmark_latency.py",
        "--wav", "a.wav",
        "--dataset", "golos_crowd",
    ]):
        with pytest.raises(SystemExit):
            benchmark_latency.main()


def test_main_single_wav_writes_legacy_keys(tmp_path):
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"fake")
    out = tmp_path / "lat.json"
    session = {
        "ttfp_ms": 400.0,
        "ttfs_ms": 200.0,
        "finalization_lag_ms": 4100.0,
        "audio_duration_ms": 4000.0,
        "total_audio_sent_ms": 4050.0,
        "timed_out": False,
        "no_partial": False,
        "onset_s": 0.2,
        "partial_lags_ms": [80.0, 90.0],
    }

    with patch.object(benchmark_latency, "transcribe_ws", return_value=("привет", 4.1, session)):
        with patch("sys.argv", [
            "benchmark_latency.py",
            "--wav", str(wav),
            "--output", str(out),
            "--port", "9877",
        ]):
            benchmark_latency.main()

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["time_to_first_partial_ms"] == 400.0
    assert payload["ttfp_ms"] == 400.0
    assert payload["ttfs_ms"] == 200.0
    assert payload["engine"] == "gigastt"
    assert payload["partial_response_lag_ms"]["count"] == 2
    assert payload["protocol"]["version"] == "1.0"
    assert payload["protocol"]["chunk_ms"] == 100


def test_evaluate_corpus_keeps_errors_in_n_not_as_timeouts():
    samples = [
        {"filename": "/tmp/a.wav"},
        {"filename": "/tmp/b.wav"},
    ]
    ok = {
        "ttfp_ms": 500.0,
        "ttfs_ms": 300.0,
        "finalization_lag_ms": 4000.0,
        "timed_out": False,
        "no_partial": False,
        "partial_lags_ms": [70.0],
        "time_to_first_partial_ms": 500.0,
        "partial_response_lag_ms": {"count": 1, "min": 70.0, "median": 70.0, "max": 70.0},
        "wav": "/tmp/a.wav",
        "engine": "gigastt",
    }

    def _one(path, port=9877, chunk_ms=100):
        if path.endswith("b.wav"):
            raise RuntimeError("connection refused")
        return ok

    with patch.object(benchmark_latency, "evaluate_gigastt", side_effect=_one):
        result = benchmark_latency.evaluate_corpus(samples, port=9877, chunk_ms=100)

    assert result["summary"]["n"] == 2
    assert result["summary"]["n_timeout"] == 0
    assert result["summary"]["n_error"] == 1
    assert result["summary"]["n_no_partial"] == 1
    assert result["summary"]["ttfp_ms"]["n"] == 1
    assert result["clips"][1]["ok"] is False
    assert result["clips"][1]["timed_out"] is False
    assert Path(result["clips"][0]["wav"]).name == "a.wav"
