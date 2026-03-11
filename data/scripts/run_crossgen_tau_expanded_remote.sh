#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TAU_PY="${TAU_PY:-$HOME/micromamba/envs/prt39/bin/python}"
BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-$HOME/hack4sages-output/crossgen-biosignatures}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/hack4sages-output/crossgen-biosignatures-expanded/20260311_tau241423_poseidon685}"
TAUREX_KTABLE_DIR="${TAUREX_KTABLE_DIR:-$HOME/.cache/crossgen-taurex-ktables}"
TAU_WORKERS="${TAU_WORKERS:-24}"
TAU_SHARD_SIZE="${TAU_SHARD_SIZE:-256}"
NEW_TAU_COUNT="${NEW_TAU_COUNT:-200000}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

export CROSSGEN_TAUREX_KTABLE_DIR="$TAUREX_KTABLE_DIR"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYTHONUNBUFFERED

echo "Expanded cross-generator run"
echo "  repo_root=$REPO_ROOT"
echo "  base_output_root=$BASE_OUTPUT_ROOT"
echo "  output_root=$OUTPUT_ROOT"
echo "  workers=$TAU_WORKERS"
echo "  shard_size=$TAU_SHARD_SIZE"
echo "  new_tau_count=$NEW_TAU_COUNT"
echo "  taurex_python=$TAU_PY"

"$TAU_PY" "$REPO_ROOT/data/scripts/generate_crossgen_tau_expanded_dataset.py" \
  --base-output-root "$BASE_OUTPUT_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --new-tau-count "$NEW_TAU_COUNT" \
  --workers "$TAU_WORKERS" \
  --shard-size "$TAU_SHARD_SIZE"
