#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TAU_PY="${TAU_PY:-$HOME/micromamba/envs/prt39/bin/python}"
POSEIDON_ENV="${POSEIDON_ENV:-$HOME/.venvs/poseidon39}"
NEUTRAL_ENV="${NEUTRAL_ENV:-$HOME/.venvs/crossgen-neutral}"
TAUREX_KTABLE_DIR="${TAUREX_KTABLE_DIR:-$HOME/.cache/crossgen-taurex-ktables}"
TAUREX_KTABLE_SOURCE_ROOT="${TAUREX_KTABLE_SOURCE_ROOT:-$HOME/hack4sages/input_data/opacities/lines/corr_k}"

if [[ ! -x "$TAU_PY" ]]; then
  echo "TauREx Python not found at $TAU_PY" >&2
  exit 1
fi

"$TAU_PY" -m pip install --upgrade pip
"$TAU_PY" -m pip install -r "$REPO_ROOT/data/crossgen_biosignatures/env/taurex-requirements.txt"

mkdir -p "$TAUREX_KTABLE_DIR"
declare -A TAUREX_KTABLE_SOURCES=(
  ["H2O.h5"]="$TAUREX_KTABLE_SOURCE_ROOT/H2O_HITEMP_R_400/H2O_HITEMP_R_400.h5"
  ["CO2.h5"]="$TAUREX_KTABLE_SOURCE_ROOT/CO2_R_400/CO2_R_400.h5"
  ["CO.h5"]="$TAUREX_KTABLE_SOURCE_ROOT/CO_all_iso_HITEMP_R_400/CO_all_iso_HITEMP_R_400.h5"
  ["CH4.h5"]="$TAUREX_KTABLE_SOURCE_ROOT/CH4_R_400/CH4_R_400.h5"
  ["NH3.h5"]="$TAUREX_KTABLE_SOURCE_ROOT/NH3_R_400/NH3_R_400.h5"
)

for target_name in "${!TAUREX_KTABLE_SOURCES[@]}"; do
  source_path="${TAUREX_KTABLE_SOURCES[$target_name]}"
  if [[ ! -f "$source_path" ]]; then
    echo "Required TauREx k-table source not found: $source_path" >&2
    exit 1
  fi
  ln -sfn "$source_path" "$TAUREX_KTABLE_DIR/$target_name"
done

if [[ ! -x "$POSEIDON_ENV/bin/python" ]]; then
  "$TAU_PY" -m venv "$POSEIDON_ENV"
fi
"$POSEIDON_ENV/bin/python" -m pip install --upgrade pip
"$POSEIDON_ENV/bin/python" -m pip install -r "$REPO_ROOT/data/crossgen_biosignatures/env/poseidon-requirements.txt"

if [[ ! -x "$NEUTRAL_ENV/bin/python" ]]; then
  python3 -m venv "$NEUTRAL_ENV"
fi
"$NEUTRAL_ENV/bin/python" -m pip install --upgrade pip
"$NEUTRAL_ENV/bin/python" -m pip install -r "$REPO_ROOT/data/crossgen_biosignatures/env/neutral-requirements.txt"
