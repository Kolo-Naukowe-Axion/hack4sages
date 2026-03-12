#!/usr/bin/env bash
set -euo pipefail

WORKFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${WORKFLOW_DIR}/../.." && pwd)"
ENV_DIR="${ENV_DIR:-$HOME/.venvs/ariel-quantum-regression}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

"$PYTHON_BIN" -m venv "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install --upgrade pip
"$ENV_DIR/bin/python" -m pip install \
  "numpy>=1.26" \
  "pandas>=2.2" \
  "h5py>=3.10" \
  "pyarrow>=17" \
  "scikit-learn>=1.5" \
  "torch>=2.3" \
  "pennylane>=0.44,<0.45" \
  "pennylane-lightning>=0.44,<0.45"
"$ENV_DIR/bin/python" - <<'PY'
import importlib

for name in ("torch", "h5py", "pandas", "pennylane"):
    mod = importlib.import_module(name)
    print(f"{name}: {getattr(mod, '__version__', 'unknown')}")

try:
    import torch
    print("CUDA available:", torch.cuda.is_available())
    print("MPS available:", bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()))
except Exception as exc:
    print("Torch backend probe failed:", exc)
PY

cat <<EOF

Environment ready.
Run hybrid training with live progress:
$ENV_DIR/bin/python -u $REPO_ROOT/models/taurex_exobiome_without_quant/run_ariel_quantum_regression.py \\
  --data-root "$REPO_ROOT/data/TauREx set" \\
  --dataset-format taurex \\
  --output-dir $REPO_ROOT/outputs/taurex_exobiome_without_quant_live \\
  --batch-size 64 \\
  --eval-batch-size 128 \\
  --max-epochs 30 \\
  --log-every-batches 20

Run a faster Mac baseline first:
$ENV_DIR/bin/python -u $REPO_ROOT/models/taurex_exobiome_without_quant/run_ariel_quantum_regression.py \\
  --data-root "$REPO_ROOT/data/TauREx set" \\
  --dataset-format taurex \\
  --output-dir $REPO_ROOT/outputs/taurex_exobiome_without_quant_classical_mac \\
  --classical-only \\
  --batch-size 128 \\
  --eval-batch-size 256 \\
  --max-epochs 30 \\
  --log-every-batches 20
EOF
