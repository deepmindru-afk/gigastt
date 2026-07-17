#!/usr/bin/env python3
"""Recompute verbatim WER with a Unicode-aware normalization from a benchmark result JSON.

The cross-engine harness's WER normalization (``benchmark/common.py``) keeps only
``[a-zа-я0-9]`` — tuned for Russian/English. That strips the extra Turkic Cyrillic
letters used by Kazakh / Kyrgyz (``ә ғ қ ң ө ұ ү һ і``), which would distort their
WER. This recomputes a verbatim WER that keeps *every* Unicode letter and digit —
the correct metric for the GigaAM Multilingual heads' non-Russian languages
(Kazakh, Kyrgyz, Uzbek). Same "verbatim, no ITN" spirit as the harness's
``naive_wer``, just script-complete.

Usage:
    python scripts/wer_unicode.py benchmark/results_full/fleurs_kk_gigastt_ml_ctc.json ...
"""

import json
import random
import sys


# Apostrophe / tutuq-belgisi variants. Uzbek Latin writes `oʻ` / `gʻ` with the
# modifier letter U+02BB (which `str.isalnum` counts as a letter), while the
# model emits an ASCII `'` — folding them avoids a spurious mismatch on every
# such word. ASCII `'` is already dropped by the isalnum filter; these must be
# dropped explicitly because they are "letters".
_APOSTROPHES = "ʻʼ‘’ʹ′´`"


def normalize(text: str) -> list[str]:
    """Lowercase, fold apostrophe variants, drop non-alphanumerics, split.

    Matches the harness's ``naive`` normalization (``benchmark/common.py``:
    strip punctuation, no ITN) but is Unicode-complete: ``str.isalnum`` keeps
    Latin, Cyrillic, and the Turkic Cyrillic extensions alike, whereas the
    harness's ``[a-zа-я0-9]`` range drops the Kazakh/Kyrgyz-specific letters.
    Apostrophe variants are folded so Uzbek ``oʻ`` (U+02BB) and the model's
    ``o'`` (U+0027) compare equal.
    """
    text = text.lower()
    text = "".join("" if c in _APOSTROPHES else c for c in text)
    out = [c for c in text if c.isalnum() or c.isspace()]
    return "".join(out).split()


def word_edit_distance(ref: list[str], hyp: list[str]) -> int:
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ref[i - 1] != hyp[j - 1]))
            prev = cur
    return dp[m]


def _has_digit(text: str) -> bool:
    return any(c.isdigit() for c in text)


def _wer_ci(per_sample: list[tuple[int, int]]) -> tuple[float, float, float]:
    """Aggregate WER% + 95% bootstrap CI over per-sample (ref_words, errors)."""
    tot_err = sum(e for _, e in per_sample)
    tot_ref = sum(n for n, _ in per_sample)
    wer = 100.0 * tot_err / tot_ref if tot_ref else 0.0
    rng = random.Random(42)
    k = len(per_sample)
    boots = []
    for _ in range(1000):
        se = sr = 0
        for _ in range(k):
            n, e = per_sample[rng.randrange(k)]
            se += e
            sr += n
        boots.append(100.0 * se / sr if sr else 0.0)
    boots.sort()
    return wer, boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]


def report(path: str) -> None:
    runner = json.load(open(path, encoding="utf-8"))["runners"][0]
    details = runner.get("details", [])

    full, digit_free = [], []
    for s in details:
        ref = s.get("reference", "")
        errors = word_edit_distance(normalize(ref), normalize(s.get("hypothesis", "")))
        pair = (len(normalize(ref)), errors)
        full.append(pair)
        # The charwise-CTC heads spell numbers out while some references keep
        # digits, and there is no reliable words<->digits ITN for kk/ky/uz
        # (num2words has no support). Report a digit-free subset that removes
        # this normalization confound, alongside the full (upper-bound) WER.
        if not _has_digit(ref):
            digit_free.append(pair)

    fw, flo, fhi = _wer_ci(full)
    dw, dlo, dhi = _wer_ci(digit_free)
    print(
        f"{runner['name']:24s} "
        f"full: WER={fw:6.2f} ({flo:.1f}-{fhi:.1f}) n={len(full):4d}  |  "
        f"digit-free: WER={dw:6.2f} ({dlo:.1f}-{dhi:.1f}) n={len(digit_free):4d}"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: wer_unicode.py <result.json> [...]", file=sys.stderr)
        sys.exit(2)
    for p in sys.argv[1:]:
        report(p)
