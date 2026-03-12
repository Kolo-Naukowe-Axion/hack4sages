#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:?Set REMOTE_HOST to the Vast SSH target before launching.}"
REMOTE_SSH_ARGS="${REMOTE_SSH_ARGS:-}"
REMOTE_ROOT="${REMOTE_ROOT:-/workspace/hack4sages}"
ENV_DIR="${ENV_DIR:-/opt/aqr-taurex}"
DATA_ROOT="${DATA_ROOT:-$REMOTE_ROOT/data/TauREx set}"
SESSION_NAME="${SESSION_NAME:-aqr_taurex_train}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REMOTE_ROOT/outputs/ariel_quantum_taurex_$(date +%Y%m%d_%H%M%S)}"

remote_script=$(cat <<'EOF'
set -euo pipefail

if [[ ! -d "$REMOTE_ROOT" ]]; then
  echo "Missing remote repo root: $REMOTE_ROOT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"
cd "$REMOTE_ROOT"

if command -v tmux >/dev/null 2>&1; then
  tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
  tmux new-session -d -s "$SESSION_NAME" \
    "cd $REMOTE_ROOT && DATA_ROOT=$(printf '%q' "$DATA_ROOT") OUTPUT_ROOT=$(printf '%q' "$OUTPUT_ROOT") ENV_DIR=$(printf '%q' "$ENV_DIR") bash models/taurex_exobiome_without_quant/run_ariel_quantum_taurex_ubuntu_gpu.sh"
  echo "Started remote tmux session: $SESSION_NAME"
  echo "Attach with: tmux attach -t $SESSION_NAME"
else
  setsid -f bash -lc \
    "cd $(printf '%q' "$REMOTE_ROOT") && \
     exec env DATA_ROOT=$(printf '%q' "$DATA_ROOT") OUTPUT_ROOT=$(printf '%q' "$OUTPUT_ROOT") ENV_DIR=$(printf '%q' "$ENV_DIR") \
     bash $(printf '%q' "$REMOTE_ROOT/models/taurex_exobiome_without_quant/run_ariel_quantum_taurex_ubuntu_gpu.sh") \
     > $(printf '%q' "$OUTPUT_ROOT/launcher.log") 2>&1 < /dev/null"
  echo "Started remote background process without tmux."
fi

echo "Run directory: $OUTPUT_ROOT"
echo "Stage 1 log: $OUTPUT_ROOT/stage1_classical/train.log"
echo "Stage 2 log: $OUTPUT_ROOT/stage2_hybrid/train.log"
EOF
)

ssh ${REMOTE_SSH_ARGS} -t "$REMOTE_HOST" \
  "REMOTE_ROOT=$(printf '%q' "$REMOTE_ROOT") ENV_DIR=$(printf '%q' "$ENV_DIR") DATA_ROOT=$(printf '%q' "$DATA_ROOT") SESSION_NAME=$(printf '%q' "$SESSION_NAME") OUTPUT_ROOT=$(printf '%q' "$OUTPUT_ROOT") bash -lc $(printf '%q' "$remote_script")"
