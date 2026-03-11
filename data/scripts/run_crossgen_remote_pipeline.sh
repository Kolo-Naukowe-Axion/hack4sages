#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/hack4sages-output/crossgen-biosignatures}"
TAU_PY="${TAU_PY:-$HOME/micromamba/envs/prt39/bin/python}"
POSEIDON_PY="${POSEIDON_PY:-$HOME/.venvs/poseidon39/bin/python}"
NEUTRAL_PY="${NEUTRAL_PY:-$HOME/.venvs/crossgen-neutral/bin/python}"
TAUREX_KTABLE_DIR="${TAUREX_KTABLE_DIR:-$HOME/.cache/crossgen-taurex-ktables}"
POSEIDON_INPUT_DATA="${POSEIDON_INPUT_DATA:-$HOME/poseidon-inputs}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/crossgen-matplotlib}"
TAU_WORKERS="${TAU_WORKERS:-8}"
TAU_SHARD_SIZE="${TAU_SHARD_SIZE:-256}"
POSEIDON_WORKERS="${POSEIDON_WORKERS:-1}"
POSEIDON_SHARD_SIZE="${POSEIDON_SHARD_SIZE:-685}"

export CROSSGEN_TAUREX_KTABLE_DIR="$TAUREX_KTABLE_DIR"
export POSEIDON_input_data="$POSEIDON_INPUT_DATA"
export MPLCONFIGDIR

"$TAU_PY" "$REPO_ROOT/data/scripts/generate_crossgen_biosignatures.py" \
  --output-root "$OUTPUT_ROOT" \
  --mode tau \
  --workers "$TAU_WORKERS" \
  --shard-size "$TAU_SHARD_SIZE"

"$POSEIDON_PY" "$REPO_ROOT/data/scripts/generate_crossgen_biosignatures.py" \
  --output-root "$OUTPUT_ROOT" \
  --mode poseidon \
  --workers "$POSEIDON_WORKERS" \
  --shard-size "$POSEIDON_SHARD_SIZE"

"$NEUTRAL_PY" "$REPO_ROOT/data/scripts/generate_crossgen_biosignatures.py" \
  --output-root "$OUTPUT_ROOT" \
  --mode assemble

"$NEUTRAL_PY" "$REPO_ROOT/data/scripts/validate_crossgen_biosignatures.py" \
  --output-root "$OUTPUT_ROOT"

"$NEUTRAL_PY" "$REPO_ROOT/data/scripts/run_crossgen_baseline.py" \
  --output-root "$OUTPUT_ROOT"
