#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-$HOME/.venvs/five-qubit-exobiome/bin/python}"
DATA_ROOT="${1:-${PROJECT_ROOT}/data/ariel-ml-dataset}"
OUTPUT_ROOT="${2:-${PROJECT_ROOT}/outputs/five_qubit_exobiome_two_stage_$(date +%Y%m%d_%H%M%S)}"

CACHE_DIR="${OUTPUT_ROOT}/prepared_cache"
STAGE1_DIR="${OUTPUT_ROOT}/stage1_classical"
STAGE2_DIR="${OUTPUT_ROOT}/stage2_hybrid"

mkdir -p "${OUTPUT_ROOT}" "${STAGE1_DIR}" "${STAGE2_DIR}" "${CACHE_DIR}"

echo "Project root: ${PROJECT_ROOT}"
echo "Data root: ${DATA_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Started: $(date --iso-8601=seconds)"

stage1_cmd=(
  "${PYTHON_BIN}" -u "${PROJECT_ROOT}/models/five_qubit_exobiome/run_five_qubit_exobiome.py"
  --project-root "${PROJECT_ROOT}"
  --data-root "${DATA_ROOT}"
  --output-dir "${STAGE1_DIR}"
  --prepared-cache-dir "${CACHE_DIR}"
  --seed 42
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

printf '%q ' "${stage1_cmd[@]}" > "${STAGE1_DIR}/command.sh"
printf '\n' >> "${STAGE1_DIR}/command.sh"

echo
echo "=== Stage 1: classical backbone ==="
"${stage1_cmd[@]}" 2>&1 | tee "${STAGE1_DIR}/train.log"

stage2_cmd=(
  "${PYTHON_BIN}" -u "${PROJECT_ROOT}/models/five_qubit_exobiome/run_five_qubit_exobiome.py"
  --project-root "${PROJECT_ROOT}"
  --data-root "${DATA_ROOT}"
  --output-dir "${STAGE2_DIR}"
  --prepared-cache-dir "${CACHE_DIR}"
  --init-checkpoint-path "${STAGE1_DIR}/best_model.pt"
  --seed 42
  --batch-size 16
  --eval-batch-size 64
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
  --qnn-qubits 5
  --qnn-depth 2
  --quantum-device lightning.gpu
  --log-every-batches 20
)

printf '%q ' "${stage2_cmd[@]}" > "${STAGE2_DIR}/command.sh"
printf '\n' >> "${STAGE2_DIR}/command.sh"

echo
echo "=== Stage 2: hybrid fine-tune ==="
"${stage2_cmd[@]}" 2>&1 | tee "${STAGE2_DIR}/train.log"

echo
echo "Finished: $(date --iso-8601=seconds)"
echo "Artifacts:"
echo "  Stage 1: ${STAGE1_DIR}"
echo "  Stage 2: ${STAGE2_DIR}"
