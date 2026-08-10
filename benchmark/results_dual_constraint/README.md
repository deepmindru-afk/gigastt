# Dual-constraint freeze / post results

Product path: **lean INT8 ORT `rnnt`** (no permanent F32 pack).

| File | Role |
|------|------|
| `freeze.json` | Parent commit binary (`2c09b7e`), same harness |
| `post.json` | HEAD: sparse mel + single-session thread reserve |
| `compare_freeze_post.json` | Freeze vs post deltas |

## Protocol (identical freeze and post)

```sh
python3 benchmark/dual_constraint_bench.py \
  --binary "$BINARY" --batches 3 --output measure.json
python3 benchmark/dual_constraint_gate.py \
  --freeze freeze.json --post post.json
```

- **batches:** 3
- **RSS sample:** after short golos fixtures, **before** long40
- **Primary:** long40 **BEST** strictly better; long40 **MEAN** non-worse
- **Competitive:** absolute BEST ≤ **0.030** (no host-limited carve-out)
- **Stretch:** 0.015–0.020 aspirational

## First win

1. Sparse HTK mel filterbank (feature-identical to dense)
2. Auto encoder threads: single-session multi-core reserves one core for OS/I/O
