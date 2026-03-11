#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TAU_PY="${TAU_PY:-$HOME/micromamba/envs/prt39/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/hack4sages-output/crossgen-biosignatures/extensions/tau_20260311_200000}"
TAUREX_KTABLE_DIR="${TAUREX_KTABLE_DIR:-$HOME/.cache/crossgen-taurex-ktables}"
TAU_WORKERS="${TAU_WORKERS:-24}"
TAU_SHARD_SIZE="${TAU_SHARD_SIZE:-1000}"
TAU_EXTENSION_COUNT="${TAU_EXTENSION_COUNT:-200000}"
TAU_EXTENSION_START_ORDINAL="${TAU_EXTENSION_START_ORDINAL:-41423}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

export CROSSGEN_TAUREX_KTABLE_DIR="$TAUREX_KTABLE_DIR"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYTHONUNBUFFERED

echo "TauREx extension run"
echo "  repo_root=$REPO_ROOT"
echo "  output_root=$OUTPUT_ROOT"
echo "  workers=$TAU_WORKERS"
echo "  shard_size=$TAU_SHARD_SIZE"
echo "  count=$TAU_EXTENSION_COUNT"
echo "  start_ordinal=$TAU_EXTENSION_START_ORDINAL"
echo "  taurex_python=$TAU_PY"

"$TAU_PY" "$REPO_ROOT/data/scripts/generate_crossgen_tau_extension.py" \
  --output-root "$OUTPUT_ROOT" \
  --count "$TAU_EXTENSION_COUNT" \
  --start-ordinal "$TAU_EXTENSION_START_ORDINAL" \
  --workers "$TAU_WORKERS" \
  --shard-size "$TAU_SHARD_SIZE"
