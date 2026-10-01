from sidecar_contention import distribution


def test_nearest_rank_percentiles_keep_sample_count_and_units():
    stats = distribution([i * 1_000_000 for i in range(1, 21)])
    assert stats == {"n": 20, "p50_ms": 10.5, "p95_ms": 19, "sum_seconds": .21}
    assert distribution([]) == {"n": 0, "p50_ms": None, "p95_ms": None, "sum_seconds": 0}


def test_opaque_speaker_wait_and_recorded_cpu_units():
    from sidecar_contention import summarize
    raw = {"case":"speaker","pool":1,"observations":[
        {"stage":"speaker_embedding_total","wait_ns":None,"execution_ns":2_000_000}],
        "rows":[{"kind":"batch","latency_ns":3_000_000,"checkout_ns":1_000_000,"rss_kib":1024}],
        "elapsed_ns":1_000_000_000,"process_cpu_ticks":250,"audio_seconds":2,
        "after_load_rss_kib":512,"load_ns":1,"cold_batch":{"latency_ns":1},
        "cold_interactive":{"latency_ns":1}}
    result = summarize(raw, 250)
    assert result["cpu_seconds"] == 1
    assert result["stages"]["speaker_embedding_total"]["wait"]["n"] == 0
    assert result["stages"]["speaker_embedding_total"]["wait"]["p50_ms"] is None
    assert result["stages"]["speaker_embedding_total"]["execution"]["p50_ms"] == 2
