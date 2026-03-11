#!/usr/bin/env bash
set -euo pipefail

WORKFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$WORKFLOW_DIR/../.." && pwd)"

DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/full-ariel}"
PREPARED_DATA="${PREPARED_DATA:-$PROJECT_ROOT/data/generated-data/ariel_winner_nf_prepared}"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/local_runs/ariel_winner_nf_$(date +%Y%m%d_%H%M%S)}"
SETTINGS_FILE="${SETTINGS_FILE:-$WORKFLOW_DIR/settings/winner_noised_independent_nsf.yaml}"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv-ariel-winner-nf}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

required_files=(
  "$DATA_ROOT/TrainingData/AuxillaryTable.csv"
  "$DATA_ROOT/TrainingData/Ground Truth Package/FM_Parameter_Table.csv"
  "$DATA_ROOT/TrainingData/SpectralData.hdf5"
  "$DATA_ROOT/TestData/AuxillaryTable.csv"
  "$DATA_ROOT/TestData/SpectralData.hdf5"
)
for path in "${required_files[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "ADC2023 dataset is incomplete. Missing: $path" >&2
    exit 1
  fi
done

mkdir -p "$RUN_DIR"
if [[ ! -x "$VENV_DIR/bin/python3" ]]; then
  rm -rf "$VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
cd "$PROJECT_ROOT"

python -m pip install --upgrade pip
python -m pip install --index-url "$PYTORCH_INDEX_URL" torch
python -m pip install -r "$WORKFLOW_DIR/requirements-vast.txt"

if [[ "${PREPARED_FORCE_OVERWRITE:-0}" == "1" || ! -f "$PREPARED_DATA/manifest.json" ]]; then
  python -m models.ariel_winner_nf.prepare_dataset \
    --data-root "$DATA_ROOT" \
    --output "$PREPARED_DATA" \
    --overwrite
else
  echo "Reusing prepared dataset at $PREPARED_DATA"
fi

python -u -m models.ariel_winner_nf.train \
  --settings "$SETTINGS_FILE" \
  --prepared-data "$PREPARED_DATA" \
  --run-dir "$RUN_DIR" \
  --resume auto \
  | tee "$RUN_DIR/train.log"

