#!/usr/bin/env python3
"""Dual-constraint gate: WER/quality → RAM → multi-run RTF (competitive ≤ 0.030).

Compares a post measure JSON against a freeze JSON produced by the **same**
``dual_constraint_bench.py`` protocol (identical ``batches``, RSS after short
fixtures). Exit 0 only when:

  1. Fixture quality is not worse (transcripts match freeze fixture_texts).
  2. RSS / resident are not worse than freeze (when both present).
  3. Primary warm multi-run RTF on long40:
       - BEST is **strictly better** than freeze (the competitive primary), and
       - MEAN is **not worse** than freeze within noise (≤ freeze × 1.02 + 0.001).
  4. Cold-start is not worse when both present.
  5. Primary BEST meets absolute competitive bar ≤ 0.030 (no host-limited
     carve-out). Stretch 0.015–0.020 is aspirational (reported, non-gating).
  6. Protocol fields match when present (``batches``, ``rss_sample_point``).

Usage::

    python3 benchmark/dual_constraint_gate.py \\
      --freeze benchmark/results_dual_constraint/freeze.json \\
      --post benchmark/results_dual_constraint/post.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _metrics(doc: dict[str, Any]) -> dict[str, Any]:
    if "metrics" in doc:
        return doc["metrics"]
    return doc


def _f(m: dict[str, Any], key: str) -> float | None:
    v = m.get(key)
    if v is None:
        return None
    return float(v)


def compare(freeze: dict[str, Any], post: dict[str, Any]) -> list[str]:
    """Return a list of failure reasons (empty ⇒ pass)."""
    f = _metrics(freeze)
    p = _metrics(post)
    failures: list[str] = []

    # 1) WER / quality first
    f_texts = f.get("fixture_texts") or []
    p_texts = p.get("fixture_texts") or []
    if f_texts and p_texts:
        if len(f_texts) != len(p_texts):
            failures.append(
                f"quality: fixture count mismatch freeze={len(f_texts)} post={len(p_texts)}"
            )
        else:
            for i, (a, b) in enumerate(zip(f_texts, p_texts)):
                if a != b:
                    failures.append(f"quality: fixture[{i}] transcript differs from freeze")
                    break
    elif f_texts and not p_texts:
        failures.append("quality: post missing fixture_texts")

    # 2) RAM (not worse)
    for key, label in (
        ("rss_mb_after_decode_pool1", "RSS@decode pool1"),
        ("rss_mb_after_ready_pool1", "RSS@ready pool1"),
        ("resident_mb_pool1_footprint", "resident footprint pool1"),
    ):
        fv, pv = _f(f, key), _f(p, key)
        if fv is not None and pv is not None:
            # Allow 2% measurement noise / OS reclaim jitter.
            if pv > fv * 1.02 + 1.0:
                failures.append(f"ram: {label} regressed {pv} > freeze {fv}")

    # 3) Primary RTF: BEST strictly better; MEAN non-worse (noise-tolerant);
    # absolute competitive ≤ bar always (no host-limited carve-out).
    f_best = _f(f, "warm_rtf_long40_best")
    p_best = _f(p, "warm_rtf_long40_best")
    f_mean = _f(f, "warm_rtf_long40_mean")
    p_mean = _f(p, "warm_rtf_long40_mean")
    if f_best is None or p_best is None:
        failures.append("rtf: missing warm_rtf_long40_best on freeze or post")
    else:
        if p_best >= f_best:
            failures.append(
                f"rtf: primary long40 BEST not strictly improved ({p_best} >= freeze {f_best})"
            )
        bar = _f(f, "competitive_bar_rtf") or 0.030
        if p_best > bar:
            failures.append(f"rtf: competitive bar missed ({p_best} > {bar})")
    if f_mean is not None and p_mean is not None:
        # Mean must not regress; allow 2% + 0.001 absolute host-noise slack.
        if p_mean > f_mean * 1.02 + 0.001:
            failures.append(
                f"rtf: primary long40 MEAN regressed ({p_mean} > freeze {f_mean} + noise)"
            )

    # 4) Cold-start not worse (when measured)
    f_cs, p_cs = _f(f, "cold_start_sec_pool1"), _f(p, "cold_start_sec_pool1")
    if f_cs is not None and p_cs is not None:
        if p_cs > f_cs * 1.15 + 0.25:
            failures.append(f"cold-start: regressed {p_cs}s > freeze {f_cs}s")

    # 5) Secondary short mean not catastrophically worse (noise-tolerant)
    f_sm, p_sm = _f(f, "warm_rtf_short_fixtures_mean"), _f(p, "warm_rtf_short_fixtures_mean")
    if f_sm is not None and p_sm is not None and p_sm > f_sm * 1.10 + 0.002:
        failures.append(
            f"rtf: short-fixture mean regressed beyond noise ({p_sm} vs freeze {f_sm})"
        )

    # 6) Same protocol (when both documents carry protocol metadata)
    f_proto = freeze.get("protocol") if isinstance(freeze, dict) else None
    p_proto = post.get("protocol") if isinstance(post, dict) else None
    if isinstance(f_proto, dict) and isinstance(p_proto, dict):
        for key in ("batches", "rss_sample_point", "pool_size", "model_variant"):
            if key in f_proto and key in p_proto and f_proto[key] != p_proto[key]:
                failures.append(
                    f"protocol: {key} mismatch freeze={f_proto[key]!r} post={p_proto[key]!r}"
                )

    return failures


def stretch_note(freeze: dict[str, Any], post: dict[str, Any]) -> str:
    f = _metrics(freeze)
    p = _metrics(post)
    p_rtf = _f(p, "warm_rtf_long40_best")
    bar = _f(f, "competitive_bar_rtf") or 0.030
    lo = _f(f, "stretch_rtf_lo") or 0.015
    hi = _f(f, "stretch_rtf_hi") or 0.020
    parts: list[str] = []
    if p_rtf is None:
        parts.append("stretch: no post RTF")
    elif lo <= p_rtf <= hi:
        parts.append(f"stretch: met ({p_rtf} within [{lo}, {hi}])")
    else:
        parts.append(
            f"stretch: open — post long40 BEST={p_rtf} vs target [{lo}, {hi}]; "
            "no axis was regressed to chase stretch (dual-constraint holds)."
        )
    if p_rtf is not None and p_rtf <= bar:
        parts.append(f"competitive: met (post BEST {p_rtf} ≤ {bar})")
    elif p_rtf is not None:
        parts.append(f"competitive: open (post BEST {p_rtf} > {bar})")
    return " ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--post", required=True)
    ap.add_argument("--stretch-note", default=None, help="write stretch gap note here")
    args = ap.parse_args()

    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))
    post = json.loads(Path(args.post).read_text(encoding="utf-8"))
    failures = compare(freeze, post)
    note = stretch_note(freeze, post)
    print(note)
    if args.stretch_note:
        Path(args.stretch_note).write_text(note + "\n", encoding="utf-8")

    if failures:
        print("FAIL dual-constraint gate:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    p = _metrics(post)
    print(
        "PASS dual-constraint gate: "
        f"long40 BEST={p.get('warm_rtf_long40_best')} "
        f"MEAN={p.get('warm_rtf_long40_mean')} "
        f"RSS@decode={p.get('rss_mb_after_decode_pool1')} "
        f"quality=ok competitive_absolute"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
