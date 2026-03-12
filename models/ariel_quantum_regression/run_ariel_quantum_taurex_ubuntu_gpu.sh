#!/usr/bin/env bash
set -euo pipefail

WORKFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${WORKFLOW_DIR}/../.." && pwd)"

DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/TauREx_set}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/ariel_quantum_taurex_$(date +%Y%m%d_%H%M%S)}"
ENV_DIR="${ENV_DIR:-/opt/aqr-taurex}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_DIR/bin/python}"
SEED="${SEED:-42}"

STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-1024}"
STAGE1_EVAL_BATCH_SIZE="${STAGE1_EVAL_BATCH_SIZE:-2048}"
STAGE1_MAX_EPOCHS="${STAGE1_MAX_EPOCHS:-30}"
STAGE1_EARLY_STOP="${STAGE1_EARLY_STOP:-6}"

STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-128}"
STAGE2_EVAL_BATCH_SIZE="${STAGE2_EVAL_BATCH_SIZE:-512}"
STAGE2_MAX_EPOCHS="${STAGE2_MAX_EPOCHS:-20}"
STAGE2_EARLY_STOP="${STAGE2_EARLY_STOP:-5}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-32}"
export PYTHONUTF8=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR:-1}"

required_files=(
  "$DATA_ROOT/labels.parquet"
  "$DATA_ROOT/spectra.h5"
)
for path in "${required_files[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "TauREx dataset is incomplete. Missing: $path" >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_ROOT"

if [[ ! -x "$PYTHON_BIN" ]]; then
  rm -rf "$ENV_DIR"
  python3 -m venv "$ENV_DIR"
fi

source "$ENV_DIR/bin/activate"
cd "$PROJECT_ROOT"

python -m pip install --upgrade pip
python -m pip install --index-url "$PYTORCH_INDEX_URL" torch torchvision torchaudio
python -m pip install custatevec-cu12
export CUQUANTUM_SDK="$(
  python - <<'PYSDK'
import site
print(f"{site.getsitepackages()[0]}/cuquantum")
PYSDK
)"
export LD_LIBRARY_PATH="${CUQUANTUM_SDK}/lib:${LD_LIBRARY_PATH:-}"
python -m pip install -r "$WORKFLOW_DIR/requirements-vast.txt"

python - <<'PYCHECK'
import numpy as np
import pennylane as qml
import torch

from models.ariel_quantum_regression.training import default_quantum_device

print("pennylane", qml.__version__, flush=True)
print("torch", torch.__version__, flush=True)
print("cuda_available", torch.cuda.is_available(), flush=True)
print("cuda_device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none", flush=True)
print("default_quantum_device", default_quantum_device(), flush=True)
dev = qml.device("lightning.gpu", wires=2, c_dtype=np.complex64, shots=None)

@qml.qnode(dev, interface="torch", diff_method="adjoint")
def circuit(inputs, weights):
    qml.RY(inputs[0], wires=0)
    qml.RY(inputs[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(weights[0], wires=0)
    return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

out = circuit(
    torch.tensor([0.1, 0.2], dtype=torch.float32),
    torch.tensor([0.3], dtype=torch.float32, requires_grad=True),
)
print("smoke_output", [float(value) for value in out], flush=True)
print("lightning_gpu_smoke_ok", flush=True)
PYCHECK

CACHE_DIR="${OUTPUT_ROOT}/prepared_cache"
STAGE1_DIR="${OUTPUT_ROOT}/stage1_classical"
STAGE2_DIR="${OUTPUT_ROOT}/stage2_hybrid"
mkdir -p "$CACHE_DIR" "$STAGE1_DIR" "$STAGE2_DIR"

echo "Project root: $PROJECT_ROOT"
echo "Data root: $DATA_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "Python: $PYTHON_BIN"
echo "Started: $(date --iso-8601=seconds)"

stage1_cmd=(
  "$PYTHON_BIN" -u "$PROJECT_ROOT/models/ariel_quantum_regression/run_ariel_quantum_regression.py"
  --project-root "$PROJECT_ROOT"
  --data-root "$DATA_ROOT"
  --dataset-format taurex
  --output-dir "$STAGE1_DIR"
  --prepared-cache-dir "$CACHE_DIR"
  --seed "$SEED"
  --classical-only
  --batch-size "$STAGE1_BATCH_SIZE"
  --eval-batch-size "$STAGE1_EVAL_BATCH_SIZE"
  --max-epochs "$STAGE1_MAX_EPOCHS"
  --early-stop-patience "$STAGE1_EARLY_STOP"
  --scheduler-patience 2
  --scheduler-factor 0.5
  --classical-lr 0.001
  --weight-decay 0.0001
  --gradient-clip-norm 5.0
  --dropout 0.05
  --loss-name mse
  --log-every-batches 10
)

printf '%q ' "${stage1_cmd[@]}" > "$STAGE1_DIR/command.sh"
printf '\n' >> "$STAGE1_DIR/command.sh"

echo
echo "=== Stage 1: classical backbone on TauREx ==="
"${stage1_cmd[@]}" 2>&1 | tee "$STAGE1_DIR/train.log"

stage2_cmd=(
  "$PYTHON_BIN" -u "$PROJECT_ROOT/models/ariel_quantum_regression/run_ariel_quantum_regression.py"
  --project-root "$PROJECT_ROOT"
  --data-root "$DATA_ROOT"
  --dataset-format taurex
  --output-dir "$STAGE2_DIR"
  --prepared-cache-dir "$CACHE_DIR"
  --init-checkpoint-path "$STAGE1_DIR/best_model.pt"
  --seed "$SEED"
  --batch-size "$STAGE2_BATCH_SIZE"
  --eval-batch-size "$STAGE2_EVAL_BATCH_SIZE"
  --max-epochs "$STAGE2_MAX_EPOCHS"
  --early-stop-patience "$STAGE2_EARLY_STOP"
  --scheduler-patience 2
  --scheduler-factor 0.5
  --classical-lr 0.00005
  --quantum-lr 0.0002
  --weight-decay 0.0001
  --gradient-clip-norm 5.0
  --dropout 0.05
  --loss-name mse
  --qnn-init-scale 0.1
  --quantum-warmup-epochs 0
  --quantum-ramp-epochs 8
  --quantum-backbone-freeze-epochs 4
  --qnn-qubits 8
  --qnn-depth 2
  --quantum-device lightning.gpu
  --quantum-use-async
  --log-every-batches 10
)

printf '%q ' "${stage2_cmd[@]}" > "$STAGE2_DIR/command.sh"
printf '\n' >> "$STAGE2_DIR/command.sh"

echo
echo "=== Stage 2: hybrid fine-tune on TauREx ==="
"${stage2_cmd[@]}" 2>&1 | tee "$STAGE2_DIR/train.log"

echo
echo "Finished: $(date --iso-8601=seconds)"
echo "Artifacts:"
echo "  Stage 1: $STAGE1_DIR"
echo "  Stage 2: $STAGE2_DIR"
