#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-iwo@100.103.127.124}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/iwo/hack4sages-crossgen}"
ENV_DIR="${ENV_DIR:-/home/iwo/.venvs/five-qubit-codex}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_DIR/bin/python}"
DATA_ROOT="${DATA_ROOT:-$REMOTE_ROOT/data/ariel-ml-dataset}"
SESSION_NAME="${SESSION_NAME:-five_qubit_codex}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REMOTE_ROOT/outputs/five_qubit_codex_two_stage_$(date +%Y%m%d_%H%M%S)}"

remote_script=$(cat <<'EOF'
set -euo pipefail

if [[ ! -d "\$REMOTE_ROOT" ]]; then
  echo "Missing remote repo root: \$REMOTE_ROOT" >&2
  exit 1
fi

if [[ ! -x "\$PYTHON_BIN" ]]; then
  echo "Missing remote Python interpreter: \$PYTHON_BIN" >&2
  echo "Bootstrap the environment first:" >&2
  echo "  ENV_DIR=\$ENV_DIR PYTHON_BIN=python3.11 bash \$REMOTE_ROOT/models/five_qubit_codex/setup_five_qubit_codex_env.sh" >&2
  exit 1
fi

mkdir -p "\$OUTPUT_ROOT"
cd "\$REMOTE_ROOT"

tmux kill-session -t "\$SESSION_NAME" 2>/dev/null || true
tmux new-session -d -s "\$SESSION_NAME" \
  "cd \$REMOTE_ROOT && PYTHON_BIN=\$PYTHON_BIN bash models/five_qubit_codex/run_five_qubit_codex_two_stage.sh \$DATA_ROOT \$OUTPUT_ROOT"

echo "Started remote session: \$SESSION_NAME"
echo "Output root: \$OUTPUT_ROOT"
echo "Attach with: tmux attach -t \$SESSION_NAME"
echo "Primary log: \$OUTPUT_ROOT/stage1_classical/train.log"
echo "Stage-2 log: \$OUTPUT_ROOT/stage2_hybrid/train.log"
EOF
)

ssh -t "$REMOTE_HOST" \
  "REMOTE_ROOT=$(printf '%q' "$REMOTE_ROOT") ENV_DIR=$(printf '%q' "$ENV_DIR") PYTHON_BIN=$(printf '%q' "$PYTHON_BIN") DATA_ROOT=$(printf '%q' "$DATA_ROOT") SESSION_NAME=$(printf '%q' "$SESSION_NAME") OUTPUT_ROOT=$(printf '%q' "$OUTPUT_ROOT") bash -lc $(printf '%q' "$remote_script")"
