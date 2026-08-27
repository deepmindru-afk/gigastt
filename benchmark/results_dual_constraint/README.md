# Dual-constraint freeze / post / stretch results

Product path: **lean INT8 ORT** (no permanent F32 pack).

| File | Role |
|------|------|
| `freeze.json` / `post.json` | Dual-constraint competitive on **`rnnt`** (quality default) |
| `stretch_ml_ctc.json` | Stretch RTF **0.015–0.020** on **`ml_ctc`** (speed SKU) |
| `compare_freeze_post.json` | rnnt freeze vs post deltas |

## Protocol

```sh
# Competitive (rnnt, default quality head)
python3 benchmark/dual_constraint_bench.py \
  --binary ./target/release/gigastt --batches 3 \
  --model-variant rnnt --output post.json

# Stretch (ml_ctc speed head — encoder-only, ~1.5× RTF)
python3 benchmark/dual_constraint_bench.py \
  --binary ./target/release/gigastt --batches 9 \
  --model-variant ml_ctc --output stretch_ml_ctc.json
```

- **batches:** competitive uses 3; stretch multi-run uses ≥7 for BEST
- **RSS sample:** after short fixtures, before long40
- **Competitive (rnnt):** long40 BEST ≤ **0.030**, dual-constraint vs freeze
- **Stretch (ml_ctc):** long40 BEST in **[0.015, 0.020]**
- **Primary:** BEST is the stretch/competitive scalar; MEAN must not regress vs freeze on competitive path

## What moved between the freeze and post snapshots

Point-in-time comparison: both snapshots were captured on 2026-08-10 (see
`date_utc` in each JSON); this is a freeze-vs-post delta, not a running tally.

1. Sparse HTK mel filterbank
2. Full-core auto encoder threads (`logical_cpus / pool_slots`)
3. Dual-constraint harness + gate
