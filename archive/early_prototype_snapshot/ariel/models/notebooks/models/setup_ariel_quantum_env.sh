#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${ENV_DIR:-$HOME/.venvs/ariel-quantum-regression}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

"$PYTHON_BIN" -m venv "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install --upgrade pip
"$ENV_DIR/bin/python" -m pip install -r "$REPO_ROOT/models/requirements-ariel-quantum.txt"
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
$ENV_DIR/bin/python -u $REPO_ROOT/models/run_ariel_quantum_regression.py \\
  --data-root $REPO_ROOT/data/ariel-ml-dataset \\
  --output-dir $REPO_ROOT/outputs/ariel_quantum_regression_live \\
  --batch-size 64 \\
  --eval-batch-size 128 \\
  --max-epochs 30 \\
  --log-every-batches 20

Run a faster Mac baseline first:
$ENV_DIR/bin/python -u $REPO_ROOT/models/run_ariel_quantum_regression.py \\
  --data-root $REPO_ROOT/data/ariel-ml-dataset \\
  --output-dir $REPO_ROOT/outputs/ariel_quantum_regression_classical_mac \\
  --classical-only \\
  --batch-size 128 \\
  --eval-batch-size 256 \\
  --max-epochs 30 \\
  --log-every-batches 20
EOF
