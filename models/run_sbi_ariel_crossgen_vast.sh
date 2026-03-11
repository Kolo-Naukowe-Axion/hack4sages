#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DATASET="${SOURCE_DATASET:-$PROJECT_ROOT/data/generated-data/crossgen_biosignatures_20260311}"
PREPARED_DATA="${PREPARED_DATA:-$PROJECT_ROOT/data/generated-data/crossgen_biosignatures_20260311_prepared_sbi_ariel}"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/local_runs/sbi_ariel_crossgen_h100_$(date +%Y%m%d_%H%M%S)}"
SETTINGS_FILE="${SETTINGS_FILE:-$PROJECT_ROOT/models/sbi_ariel_crossgen/settings/crossgen_h100.yaml}"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv-sbi-crossgen}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

mkdir -p "$RUN_DIR"
ulimit -n "${NOFILE_LIMIT:-65535}" || true
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install --index-url "$PYTORCH_INDEX_URL" torch
python -m pip install --no-deps dingo-gw==0.8.3
python -m pip install -r "$PROJECT_ROOT/models/sbi_ariel_crossgen/requirements-vast.txt"

TRAIN_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  wandb login --relogin "$WANDB_API_KEY"
else
  echo "WANDB_API_KEY is not set; using local live logs instead of W&B." >&2
  TRAIN_ARGS+=(--no-wandb)
fi

python -m models.sbi_ariel_crossgen.prepare_dataset \
  --source "$SOURCE_DATASET" \
  --output "$PREPARED_DATA" \
  --overwrite

python -u -m models.sbi_ariel_crossgen.train \
  --settings "$SETTINGS_FILE" \
  --prepared-data "$PREPARED_DATA" \
  --run-dir "$RUN_DIR" \
  --resume auto \
  "${TRAIN_ARGS[@]}" \
  | tee "$RUN_DIR/train.log"
