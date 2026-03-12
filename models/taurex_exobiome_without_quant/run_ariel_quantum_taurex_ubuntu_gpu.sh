#!/usr/bin/env bash
set -euo pipefail

WORKFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${WORKFLOW_DIR}/../.." && pwd)"

DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/TauREx set}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/taurex_exobiome_without_quant_$(date +%Y%m%d_%H%M%S)}"
ENV_DIR="${ENV_DIR:-/opt/taurex-noquant}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
PYTHON_BIN="${PYTHON_BIN:-}"
UPGRADE_PIP="${UPGRADE_PIP:-0}"
SEED="${SEED:-42}"

BATCH_SIZE="${BATCH_SIZE:-}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-}"
MAX_EPOCHS="${MAX_EPOCHS:-60}"
EARLY_STOP="${EARLY_STOP:-10}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-48}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-48}"
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

use_managed_env=0
if [[ -z "$PYTHON_BIN" || "$PYTHON_BIN" == "$ENV_DIR/bin/python" ]]; then
  PYTHON_BIN="$ENV_DIR/bin/python"
  use_managed_env=1
fi

if [[ "$use_managed_env" == "1" ]]; then
  if [[ ! -x "$PYTHON_BIN" ]]; then
    rm -rf "$ENV_DIR"
    python3 -m venv "$ENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$ENV_DIR/bin/activate"
  PYTHON_BIN="$ENV_DIR/bin/python"
fi

cd "$PROJECT_ROOT"

python_has_modules() {
  local modules_literal="$1"
  "$PYTHON_BIN" - <<PY
import importlib.util

modules = ${modules_literal}
missing = [name for name in modules if importlib.util.find_spec(name) is None]
raise SystemExit(0 if not missing else 1)
PY
}

if [[ "$UPGRADE_PIP" == "1" ]]; then
  "$PYTHON_BIN" -m pip install --upgrade pip
fi

if ! python_has_modules "['torch', 'torchvision', 'torchaudio']"; then
  "$PYTHON_BIN" -m pip install --index-url "$PYTORCH_INDEX_URL" torch torchvision torchaudio
fi

if ! python_has_modules "['numpy', 'pandas', 'h5py', 'pyarrow', 'sklearn']"; then
  "$PYTHON_BIN" -m pip install -r "$WORKFLOW_DIR/requirements-vast.txt"
fi

GPU_MEM_MB="$("$PYTHON_BIN" - <<'PY'
import subprocess

try:
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().splitlines()
    print(int(raw[0]) if raw else 0)
except Exception:
    print(0)
PY
)"

if [[ -z "$BATCH_SIZE" ]]; then
  if (( GPU_MEM_MB >= 130000 )); then
    BATCH_SIZE=4096
  elif (( GPU_MEM_MB >= 70000 )); then
    BATCH_SIZE=2048
  else
    BATCH_SIZE=1024
  fi
fi

if [[ -z "$EVAL_BATCH_SIZE" ]]; then
  if (( GPU_MEM_MB >= 130000 )); then
    EVAL_BATCH_SIZE=12288
  elif (( GPU_MEM_MB >= 70000 )); then
    EVAL_BATCH_SIZE=6144
  else
    EVAL_BATCH_SIZE=2048
  fi
fi

"$PYTHON_BIN" - <<'PYCHECK'
import torch

print("torch", torch.__version__, flush=True)
print("cuda_available", torch.cuda.is_available(), flush=True)
print("cuda_device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none", flush=True)
print("bf16_supported", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False, flush=True)
PYCHECK

CACHE_DIR="${OUTPUT_ROOT}/prepared_cache"
RUN_DIR="${OUTPUT_ROOT}/train"
mkdir -p "$CACHE_DIR" "$RUN_DIR"

echo "Project root: $PROJECT_ROOT"
echo "Data root: $DATA_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "Python: $PYTHON_BIN"
echo "GPU memory (MiB): $GPU_MEM_MB"
echo "Batch size: $BATCH_SIZE"
echo "Eval batch size: $EVAL_BATCH_SIZE"
echo "Started: $(date --iso-8601=seconds)"

train_cmd=(
  "$PYTHON_BIN" -u "$PROJECT_ROOT/models/taurex_exobiome_without_quant/run_ariel_quantum_regression.py"
  --project-root "$PROJECT_ROOT"
  --data-root "$DATA_ROOT"
  --dataset-format taurex
  --ignore-poseidon
  --output-dir "$RUN_DIR"
  --prepared-cache-dir "$CACHE_DIR"
  --seed "$SEED"
  --batch-size "$BATCH_SIZE"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --max-epochs "$MAX_EPOCHS"
  --early-stop-patience "$EARLY_STOP"
  --scheduler-patience 3
  --scheduler-factor 0.5
  --classical-lr 0.0015
  --quantum-lr 0.0008
  --weight-decay 0.0001
  --gradient-clip-norm 5.0
  --dropout 0.08
  --loss-name mrmse
  --quantum-warmup-epochs 0
  --quantum-ramp-epochs 1
  --quantum-backbone-freeze-epochs 0
  --qnn-qubits 8
  --qnn-depth 3
  --log-every-batches 5
)

printf '%q ' "${train_cmd[@]}" > "$RUN_DIR/command.sh"
printf '\n' >> "$RUN_DIR/command.sh"

echo
echo "=== Training non-quantum TauREx regressor ==="
"${train_cmd[@]}" 2>&1 | tee "$RUN_DIR/train.log"

echo
echo "Finished: $(date --iso-8601=seconds)"
echo "Artifacts: $RUN_DIR"
