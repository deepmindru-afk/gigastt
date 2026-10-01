"""Research rollups keep silence, rewrites and CPU units explicit."""
from live_windows import summarize


def test_live_summary_counts_retractions_and_keeps_empty_hypotheses():
    row = {
        "file": "clip.wav", "reference": "заказать яблоки", "text": "",
        "worker": 0, "audio_samples": 16000, "final_after_stop_ms": 20,
        "peak_rss_kib": 1234,
        "probe": {"stages_ns": {"mel": 2_000_000}, "windows": [
            {"start": 1, "samples": 16000, "pending": 16000, "context": 0}]},
        "events": [
            {"final": False, "text": "заказать груши", "at_ms": 800},
            {"final": False, "text": "заказать яблоки", "at_ms": 900},
            {"final": True, "text": "заказать", "at_ms": 1020, "endpoint": "stop"},
        ],
    }
    result = summarize({"rows": [row], "config": {}, "process_cpu_ticks": 250, "elapsed_ms": 1020}, cpu_ticks_per_second=250)
    assert result["wer_percent"] == 100
    assert result["partial_retracted_words"] == 1
    assert result["final_retracted_words"] == 1
    assert result["first_partial_median_ms"] == 800
    assert result["stage_elapsed_seconds"]["mel"] == 0.002
    assert result["off_grid_windows"] == 1
    assert result["process_cpu_seconds"] == 1


def test_live_summary_includes_silence_hallucinations_separately():
    row = {
        "file": "silence_only", "reference": "", "text": "привет", "worker": 0,
        "audio_samples": 16000, "final_after_stop_ms": 1, "peak_rss_kib": 100,
        "probe": {"stages_ns": {}, "windows": [{"start": 0, "samples": 16000}]},
        "events": [{"final": True, "text": "привет", "at_ms": 1000, "endpoint": "stop"}],
    }
    result = summarize({"rows": [row], "config": {}, "process_cpu_ticks": 0, "elapsed_ms": 1000})
    assert result["wer_percent"] is None
    assert result["no_partial"] == 1
    assert result["focused"][0]["errors"] == 1
    assert result["focused"][0]["reference_words"] == 0
