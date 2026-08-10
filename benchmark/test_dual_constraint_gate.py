#!/usr/bin/env python3
"""Unit tests for dual_constraint_gate.compare (no model / server required)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "dual_constraint_gate", HERE / "dual_constraint_gate.py"
)
assert spec and spec.loader
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def _doc(metrics: dict) -> dict:
    return {"schema": "t", "metrics": metrics}


def test_pass_when_strictly_better_rtf_and_ram_holds():
    freeze = {
        "schema": "t",
        "protocol": {
            "batches": 3,
            "rss_sample_point": "after_short_fixtures_before_long",
            "pool_size": 1,
            "model_variant": "rnnt",
        },
        "metrics": {
            "fixture_texts": ["a", "b"],
            "warm_rtf_long40_best": 0.0296,
            "warm_rtf_long40_mean": 0.0310,
            "warm_rtf_short_fixtures_mean": 0.0343,
            "rss_mb_after_decode_pool1": 283.0,
            "rss_mb_after_ready_pool1": 266.0,
            "resident_mb_pool1_footprint": 54.0,
            "cold_start_sec_pool1": 1.2,
            "competitive_bar_rtf": 0.030,
            "stretch_rtf_lo": 0.015,
            "stretch_rtf_hi": 0.020,
        },
    }
    post = {
        "schema": "t",
        "protocol": {
            "batches": 3,
            "rss_sample_point": "after_short_fixtures_before_long",
            "pool_size": 1,
            "model_variant": "rnnt",
        },
        "metrics": {
            "fixture_texts": ["a", "b"],
            "warm_rtf_long40_best": 0.0280,
            "warm_rtf_long40_mean": 0.0305,
            "warm_rtf_short_fixtures_mean": 0.0330,
            "rss_mb_after_decode_pool1": 280.0,
            "rss_mb_after_ready_pool1": 265.0,
            "resident_mb_pool1_footprint": 53.0,
            "cold_start_sec_pool1": 1.1,
            "competitive_bar_rtf": 0.030,
        },
    }
    assert gate.compare(freeze, post) == []


def test_fail_when_long40_mean_regresses():
    freeze = _doc(
        {
            "fixture_texts": ["x"],
            "warm_rtf_long40_best": 0.0300,
            "warm_rtf_long40_mean": 0.0310,
            "rss_mb_after_decode_pool1": 280.0,
            "competitive_bar_rtf": 0.030,
        }
    )
    post = _doc(
        {
            "fixture_texts": ["x"],
            "warm_rtf_long40_best": 0.0280,
            "warm_rtf_long40_mean": 0.0340,
            "rss_mb_after_decode_pool1": 280.0,
            "competitive_bar_rtf": 0.030,
        }
    )
    fails = gate.compare(freeze, post)
    assert any("MEAN regressed" in f for f in fails)


def test_fail_on_quality_regression():
    freeze = _doc(
        {
            "fixture_texts": ["hello"],
            "warm_rtf_long40_best": 0.0296,
            "rss_mb_after_decode_pool1": 280.0,
            "competitive_bar_rtf": 0.030,
        }
    )
    post = _doc(
        {
            "fixture_texts": ["goodbye"],
            "warm_rtf_long40_best": 0.0200,
            "rss_mb_after_decode_pool1": 270.0,
            "competitive_bar_rtf": 0.030,
        }
    )
    fails = gate.compare(freeze, post)
    assert any("quality" in f for f in fails)


def test_fail_when_rtf_not_strictly_better():
    freeze = _doc(
        {
            "fixture_texts": ["x"],
            "warm_rtf_long40_best": 0.0296,
            "rss_mb_after_decode_pool1": 280.0,
            "competitive_bar_rtf": 0.030,
        }
    )
    post = _doc(
        {
            "fixture_texts": ["x"],
            "warm_rtf_long40_best": 0.0296,
            "rss_mb_after_decode_pool1": 280.0,
            "competitive_bar_rtf": 0.030,
        }
    )
    fails = gate.compare(freeze, post)
    assert any("not strictly improved" in f for f in fails)


def test_fail_when_competitive_bar_missed_absolute():
    # Absolute competitive always required — even if freeze also misses the bar.
    freeze = _doc(
        {
            "fixture_texts": ["x"],
            "warm_rtf_long40_best": 0.0474,
            "warm_rtf_long40_mean": 0.0490,
            "rss_mb_after_decode_pool1": 285.0,
            "competitive_bar_rtf": 0.030,
        }
    )
    post = _doc(
        {
            "fixture_texts": ["x"],
            "warm_rtf_long40_best": 0.0423,
            "warm_rtf_long40_mean": 0.0468,
            "rss_mb_after_decode_pool1": 285.0,
            "competitive_bar_rtf": 0.030,
        }
    )
    fails = gate.compare(freeze, post)
    assert any("competitive bar" in f for f in fails)

    # Quiet-host style: absolute competitive met + relative win.
    freeze_q = _doc(
        {
            "fixture_texts": ["x"],
            "warm_rtf_long40_best": 0.0296,
            "warm_rtf_long40_mean": 0.0310,
            "rss_mb_after_decode_pool1": 283.0,
            "competitive_bar_rtf": 0.030,
        }
    )
    post_q = _doc(
        {
            "fixture_texts": ["x"],
            "warm_rtf_long40_best": 0.0280,
            "warm_rtf_long40_mean": 0.0300,
            "rss_mb_after_decode_pool1": 283.0,
            "competitive_bar_rtf": 0.030,
        }
    )
    assert gate.compare(freeze_q, post_q) == []


if __name__ == "__main__":
    test_pass_when_strictly_better_rtf_and_ram_holds()
    test_fail_when_long40_mean_regresses()
    test_fail_on_quality_regression()
    test_fail_when_rtf_not_strictly_better()
    test_fail_when_competitive_bar_missed_absolute()
    print("ok")
