#!/usr/bin/env bash
# Edge / Raspberry Pi measurement wrapper.
# See specs/edge-raspberry-pi-roadmap.md for the full protocol.
#
# Usage (on the Pi, 64-bit OS):
#   ./scripts/bench_edge_pi.sh --storage-label microSD --variants rnnt,ml_ctc
#   ./scripts/bench_edge_pi.sh --storage-label usb-ssd  --variants rnnt
#
# Prefer a prebuilt arm64 binary or GHCR image; do not cargo-install on-device
# unless you have swap and patience.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

STORAGE_LABEL="unknown"
VARIANTS="rnnt"
OUTPUT="benchmark/results_edge.json"
BINARY=""
POOL_SIZE=1
THREADS=""
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --storage-label)
      STORAGE_LABEL="${2:?}"
      shift 2
      ;;
    --variants)
      VARIANTS="${2:?}"
      shift 2
      ;;
    --output)
      OUTPUT="${2:?}"
      shift 2
      ;;
    --binary)
      BINARY="${2:?}"
      shift 2
      ;;
    --pool-size)
      POOL_SIZE="${2:?}"
      shift 2
      ;;
    --encoder-intra-threads)
      THREADS="${2:?}"
      shift 2
      ;;
    --help|-h)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      EXTRA+=("$1")
      shift
      ;;
  esac
done

if [[ -n "$BINARY" ]]; then
  BIN_ARG=(--binary "$BINARY")
elif [[ -x "$ROOT/target/release/gigastt" ]]; then
  BIN_ARG=(--binary "$ROOT/target/release/gigastt")
elif command -v gigastt >/dev/null 2>&1; then
  BIN_ARG=(--binary "$(command -v gigastt)")
else
  echo "error: gigastt binary not found. Install arm64 release or pass --binary." >&2
  exit 1
fi

THREAD_ARG=()
if [[ -n "$THREADS" ]]; then
  THREAD_ARG=(--encoder-intra-threads "$THREADS")
fi

# Show board identity when available (Pi).
if [[ -r /proc/device-tree/model ]]; then
  echo "device-tree model: $(tr -d '\0' </proc/device-tree/model)"
fi
uname -a
echo "storage_label=$STORAGE_LABEL variants=$VARIANTS pool_size=$POOL_SIZE"

exec python3 "$ROOT/benchmark/bench_edge.py" \
  "${BIN_ARG[@]}" \
  --storage-label "$STORAGE_LABEL" \
  --variants "$VARIANTS" \
  --pool-size "$POOL_SIZE" \
  "${THREAD_ARG[@]}" \
  --output "$OUTPUT" \
  "${EXTRA[@]}"
