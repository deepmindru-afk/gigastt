#!/usr/bin/env python3
"""Analyze WER benchmark results from JSON output."""

import json
import sys
from collections import Counter


def main(path: str):
    with open(path) as f:
        data = json.load(f)

    print(f"Samples:   {data['samples']}")
    print(f"WER:       {data['wer']}%")
    print(f"95% CI:    [{data.get('ci_low', 'N/A')}%, {data.get('ci_high', 'N/A')}%]")
    print(f"Errors:    {data['total_errors']} / {data['total_words']} words")
    print(f"Score:     {data['score']}")
    print()

    details = data.get("details", [])
    if not details:
        print("No per-sample details available.")
        return

    wers = [d["wer"] for d in details]
    print(f"WER distribution:")
    print(f"  Min:    {min(wers):.1f}%")
    print(f"  Max:    {max(wers):.1f}%")
    print(f"  Mean:   {sum(wers)/len(wers):.1f}%")
    print(f"  Median: {sorted(wers)[len(wers)//2]:.1f}%")
    print()

    high_wer = [d for d in details if d["wer"] >= 50.0]
    print(f"Samples with WER >= 50%: {len(high_wer)}")
    for d in high_wer[:20]:
        print(f"  {d['wer']:5.1f}% | ref: {d['ref_norm']}")
        print(f"            hyp: {d['hyp_norm']}")
    print()

    # Common word-level error patterns
    error_counter = Counter()
    for d in details:
        ref = d["ref_norm"].split()
        hyp = d["hyp_norm"].split()
        m = min(len(ref), len(hyp))
        for i in range(m):
            if ref[i] != hyp[i]:
                error_counter[(ref[i], hyp[i])] += 1
        if len(hyp) > len(ref):
            for w in hyp[len(ref):]:
                error_counter[("(del)", w)] += 1
        if len(ref) > len(hyp):
            for w in ref[len(hyp):]:
                error_counter[(w, "(ins)")] += 1

    print("Top 20 error pairs (ref -> hyp):")
    for (ref, hyp), count in error_counter.most_common(20):
        print(f"  {count:3d} | {ref:25s} -> {hyp}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/benchmark_full.json"
    main(path)
