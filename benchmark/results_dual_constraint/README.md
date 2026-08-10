# Dual-constraint freeze / post results

Product path: **lean INT8 ORT `rnnt`** (no permanent F32 pack).

| File | Role |
|------|------|
| `freeze.json` | Baseline from parent commit binary (`2c09b7e`), same harness |
| `post.json` | HEAD with sparse mel + single-session thread reserve |
| `compare_freeze_post.json` | Freeze vs post deltas |

## Protocol (identical for freeze and post)

```sh
# freeze binary: parent main release build
# post binary:   this branch release build
python3 benchmark/dual_constraint_bench.py \
  --binary "$BINARY" \
  --batches 3 \
  --output measure.json
python3 benchmark/dual_constraint_gate.py \
  --freeze freeze.json --post post.json
python3 benchmark/test_dual_constraint_gate.py
```

- **batches:** 3 (multi-run long40)
- **RSS sample point:** after short golos fixtures, **before** long40 (avoids sticky ORT arena)
- **Primary RTF:** long40 **BEST** must be strictly better; long40 **MEAN** must not regress (2%+0.001 noise slack)
- **Competitive bar:** absolute BEST ≤ 0.030 when freeze itself is competitive-class; if freeze BEST also misses the bar under the same host load, gate requires relative dual-constraint only and reports `host-limited-competitive`
- **Stretch:** 0.015–0.020 aspirational

## First win

1. Sparse HTK mel filterbank (feature-identical to dense)
2. Auto encoder threads: single-session multi-core reserves one core for OS/I/O
