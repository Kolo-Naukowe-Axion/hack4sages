#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-$HOME/.venvs/ariel-quantum-regression/bin/python}"
DATA_ROOT="${1:-${PROJECT_ROOT}/data/ariel-ml-dataset}"
OUTPUT_ROOT="${2:-${PROJECT_ROOT}/outputs/ariel_quantum_regression_cv_two_stage}"
FOLD_NUMBERS="${3:-${AQR_FOLD_NUMBERS:-1}}"
NUM_FOLDS="${NUM_FOLDS:-5}"
SEED="${SEED:-42}"

STAGE1_ROOT="${OUTPUT_ROOT}_stage1"
FINAL_ROOT="${OUTPUT_ROOT}"

mkdir -p "${STAGE1_ROOT}" "${FINAL_ROOT}"

echo "Project root: ${PROJECT_ROOT}"
echo "Data root: ${DATA_ROOT}"
echo "Stage 1 output root: ${STAGE1_ROOT}"
echo "Final output root: ${FINAL_ROOT}"
echo "Fold numbers: ${FOLD_NUMBERS}"
echo "Python: ${PYTHON_BIN}"
echo "Started: $(date --iso-8601=seconds)"

stage1_cmd=(
  "${PYTHON_BIN}" -u "${PROJECT_ROOT}/models/ariel_quantum_regression/run_ariel_quantum_cross_validation.py"
  --project-root "${PROJECT_ROOT}"
  --data-root "${DATA_ROOT}"
  --output-dir "${STAGE1_ROOT}"
  --seed "${SEED}"
  --num-folds "${NUM_FOLDS}"
  --fold-numbers "${FOLD_NUMBERS}"
  --classical-only
  --batch-size 256
  --eval-batch-size 512
  --max-epochs 45
  --early-stop-patience 10
  --scheduler-patience 2
  --scheduler-factor 0.5
  --classical-lr 0.001
  --weight-decay 0.0001
  --gradient-clip-norm 5.0
  --dropout 0.05
  --loss-name mse
  --log-every-batches 20
)

echo
echo "=== Stage 1: classical per-fold pretraining ==="
printf '%q ' "${stage1_cmd[@]}"
printf '\n'
"${stage1_cmd[@]}"

stage2_cmd=(
  "${PYTHON_BIN}" -u "${PROJECT_ROOT}/models/ariel_quantum_regression/run_ariel_quantum_cross_validation.py"
  --project-root "${PROJECT_ROOT}"
  --data-root "${DATA_ROOT}"
  --output-dir "${FINAL_ROOT}"
  --init-checkpoint-path "${STAGE1_ROOT}/fold_$(printf '%02d' "${FOLD_NUMBERS}")/best_model.pt"
  --seed "${SEED}"
  --num-folds "${NUM_FOLDS}"
  --fold-numbers "${FOLD_NUMBERS}"
  --batch-size 16
  --eval-batch-size 32
  --max-epochs 30
  --early-stop-patience 8
  --scheduler-patience 3
  --scheduler-factor 0.5
  --classical-lr 0.00005
  --quantum-lr 0.0002
  --weight-decay 0.0001
  --gradient-clip-norm 5.0
  --dropout 0.05
  --loss-name mse
  --qnn-init-scale 0.1
  --quantum-warmup-epochs 0
  --quantum-ramp-epochs 12
  --quantum-backbone-freeze-epochs 6
  --qnn-qubits 8
  --qnn-depth 2
  --quantum-device lightning.gpu
  --log-every-batches 20
)

echo
echo "=== Stage 2: hybrid per-fold fine-tune ==="
printf '%q ' "${stage2_cmd[@]}"
printf '\n'
"${stage2_cmd[@]}"

echo
echo "Finished: $(date --iso-8601=seconds)"
echo "Stage 1 root: ${STAGE1_ROOT}"
echo "Final root: ${FINAL_ROOT}"
