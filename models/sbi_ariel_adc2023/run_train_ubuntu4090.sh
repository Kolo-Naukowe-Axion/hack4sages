#!/usr/bin/env bash
set -euo pipefail

WORKFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$WORKFLOW_DIR/../.." && pwd)"

DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/full-ariel}"
PREPARED_DATA="${PREPARED_DATA:-$PROJECT_ROOT/data/generated-data/adc2023_fivegas_fmpe_prepared}"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/local_runs/sbi_ariel_adc2023_rtx4090_$(date +%Y%m%d_%H%M%S)}"
SETTINGS_FILE="${SETTINGS_FILE:-$WORKFLOW_DIR/settings/adc2023_rtx4090.yaml}"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv-sbi-adc2023}"
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
ulimit -n "${NOFILE_LIMIT:-65535}" || true
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
cd "$PROJECT_ROOT"

python -m pip install --upgrade pip
python -m pip install --index-url "$PYTORCH_INDEX_URL" torch
python -m pip install --no-deps dingo-gw==0.8.3
python -m pip install \
  "astropy>=7,<8" \
  "bilby>=2.8,<3" \
  "configargparse>=1.7,<2" \
  "corner>=2.2,<3" \
  "glasflow>=0.4,<0.5" \
  "h5py>=3.10,<4" \
  "numpy>=1.26,<3" \
  "pandas>=2.2,<3" \
  "pyarrow>=17,<20" \
  "PyYAML>=6,<7" \
  "scikit-learn>=1.5,<2" \
  "torchdiffeq>=0.2.5,<0.3" \
  "tqdm>=4.66,<5" \
  "wandb>=0.19,<0.21" \
  "threadpoolctl>=3.5,<4"

TRAIN_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  wandb login --relogin "$WANDB_API_KEY"
else
  TRAIN_ARGS+=(--no-wandb)
fi

if [[ "${PREPARED_FORCE_OVERWRITE:-0}" == "1" || ! -f "$PREPARED_DATA/manifest.json" ]]; then
  python -m models.sbi_ariel_adc2023.prepare_dataset \
    --data-root "$DATA_ROOT" \
    --output "$PREPARED_DATA" \
    --overwrite
else
  echo "Reusing prepared dataset at $PREPARED_DATA"
fi

if [[ -f "$RUN_DIR/best_model_by_mrmse.pt" ]]; then
  python -u -m models.sbi_ariel_adc2023.evaluate \
    --run-dir "$RUN_DIR" \
    --prepared-data "$PREPARED_DATA"
  exit 0
fi

python -u -m models.sbi_ariel_adc2023.train \
  --settings "$SETTINGS_FILE" \
  --prepared-data "$PREPARED_DATA" \
  --run-dir "$RUN_DIR" \
  --resume auto \
  "${TRAIN_ARGS[@]}" \
  | tee "$RUN_DIR/train.log"
