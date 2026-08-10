# Dual-constraint freeze / post results

Product path: **lean INT8 ORT `rnnt`** (no permanent F32 pack).

| File | Role |
|------|------|
| `freeze.json` | Baseline metrics + measure commands |
| `post.json` | Same protocol after dual-constraint win |
| `compare_freeze_post.json` | Freeze vs post deltas |

## Protocol

```sh
cargo build --release -p gigastt
python3 benchmark/dual_constraint_bench.py \
  --binary ./target/release/gigastt \
  --output benchmark/results_dual_constraint/measure.json
python3 benchmark/dual_constraint_gate.py \
  --freeze benchmark/results_dual_constraint/freeze.json \
  --post benchmark/results_dual_constraint/post.json
python3 benchmark/test_dual_constraint_gate.py
```

Gate order: **WER/quality → RAM → multi-run RTF**.

- **Primary RTF:** warm REST multi-run BEST on ~42 s concat (`golos_00..04` ×2), pool=1.
- **Competitive bar:** long40 BEST ≤ **0.030**.
- **Stretch:** 0.015–0.020 (aspirational; dual-constraint may block further gains).
- **RSS/resident:** sampled after short golos fixtures (not after long40 sticky arena).

## First win (this branch)

1. **Sparse HTK mel filterbank** — triangle bands applied as contiguous sparse slices (~33× fewer MACs on the mel apply; features match dense within float noise).
2. **Auto encoder threads** — single-session pool budget on ≥4-core hosts reserves one core for OS/I/O (`cpus - 1`), multi-slot pools unchanged.
