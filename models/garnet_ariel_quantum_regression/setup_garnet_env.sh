#!/usr/bin/env bash
set -euo pipefail

WORKFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${WORKFLOW_DIR}/../.." && pwd)"
ENV_DIR="${ENV_DIR:-$HOME/.venvs/garnet-ariel-quantum}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

"$PYTHON_BIN" -m venv "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install --upgrade pip
"$ENV_DIR/bin/python" -m pip install \
  "numpy>=1.26" \
  "pandas>=2.2" \
  "h5py>=3.10" \
  "scikit-learn>=1.5" \
  "torch>=2.3" \
  "qiskit>=2.0,<3.0" \
  "qiskit-aer>=0.15,<0.18" \
  "iqm-client[qiskit]==33.0.5"

cat <<EOF

Environment ready.

Dry-run tutorial notebook:
$ENV_DIR/bin/python -m jupyter lab $REPO_ROOT/models/garnet_ariel_quantum_regression/garnet_port_tutorial.ipynb

The notebook defaults to:
- no live IQM submission
- local statevector checks
- IQM fake/facade transpilation and dry evaluation

EOF
