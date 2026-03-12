#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${ENV_DIR:-$HOME/.venvs/adc-crossgen-tf}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

"$PYTHON_BIN" -m venv "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install --upgrade pip
"$ENV_DIR/bin/python" -m pip install -r "$REPO_ROOT/models/requirements-tf-quantum.txt"
"$ENV_DIR/bin/python" - <<'PY'
import tensorflow as tf

print("TensorFlow:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("Visible GPUs:", [gpu.name for gpu in gpus])
PY

cat <<EOF

Environment ready.
Run training with:
$ENV_DIR/bin/python $REPO_ROOT/models/run_adc_crossgen_tf.py \\
  --data-root $REPO_ROOT/data/generated-data/crossgen_biosignatures_20260311 \\
  --output-dir $REPO_ROOT/outputs/adc_crossgen_tf
EOF
