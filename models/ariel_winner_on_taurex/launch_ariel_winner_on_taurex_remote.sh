#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-iwo@100.103.127.124}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/iwo/hack4sages-crossgen}"
ENV_DIR="${ENV_DIR:-/home/iwo/.venvs/ariel-winner-on-taurex}"
DATA_ROOT="${DATA_ROOT:-$REMOTE_ROOT/data/TauREx set}"
PREPARED_DATA="${PREPARED_DATA:-$REMOTE_ROOT/data/generated-data/ariel_winner_on_taurex_prepared}"
SETTINGS_FILE="${SETTINGS_FILE:-$REMOTE_ROOT/models/ariel_winner_on_taurex/settings/winner_noised_independent_nsf.yaml}"
SESSION_NAME="${SESSION_NAME:-ariel_winner_on_taurex}"
RUN_DIR="${RUN_DIR:-$REMOTE_ROOT/local_runs/ariel_winner_on_taurex_$(date +%Y%m%d_%H%M%S)}"

remote_script=$(cat <<'EOF'
set -euo pipefail

if [[ ! -d "$REMOTE_ROOT" ]]; then
  echo "Missing remote repo root: $REMOTE_ROOT" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"
cd "$REMOTE_ROOT"

if command -v tmux >/dev/null 2>&1; then
  tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
  tmux new-session -d -s "$SESSION_NAME" \
    "cd $REMOTE_ROOT && DATA_ROOT=$(printf '%q' "$DATA_ROOT") PREPARED_DATA=$(printf '%q' "$PREPARED_DATA") SETTINGS_FILE=$(printf '%q' "$SETTINGS_FILE") RUN_DIR=$(printf '%q' "$RUN_DIR") VENV_DIR=$(printf '%q' "$ENV_DIR") bash models/ariel_winner_on_taurex/run_train_ubuntu4090.sh"
  echo "Started remote tmux session: $SESSION_NAME"
else
  setsid -f bash -lc \
    "cd $(printf '%q' "$REMOTE_ROOT") && \
     echo \$\$ > $(printf '%q' "$RUN_DIR/train.pid") && \
     exec env \
       DATA_ROOT=$(printf '%q' "$DATA_ROOT") \
       PREPARED_DATA=$(printf '%q' "$PREPARED_DATA") \
       SETTINGS_FILE=$(printf '%q' "$SETTINGS_FILE") \
       RUN_DIR=$(printf '%q' "$RUN_DIR") \
       VENV_DIR=$(printf '%q' "$ENV_DIR") \
       bash $(printf '%q' "$REMOTE_ROOT/models/ariel_winner_on_taurex/run_train_ubuntu4090.sh") \
       > $(printf '%q' "$RUN_DIR/launcher.log") 2>&1 < /dev/null"
  sleep 1
  echo "Started remote background process without tmux."
  if [[ -f "$RUN_DIR/train.pid" ]]; then
    echo "PID: $(cat "$RUN_DIR/train.pid")"
  else
    echo "PID file was not created yet."
  fi
fi

echo "Run directory: $RUN_DIR"
if command -v tmux >/dev/null 2>&1; then
  echo "Attach with: tmux attach -t $SESSION_NAME"
fi
echo "Primary log: $RUN_DIR/train.log"
echo "Bootstrap log: $RUN_DIR/launcher.log"
EOF
)

ssh -t "$REMOTE_HOST" \
  "REMOTE_ROOT=$(printf '%q' "$REMOTE_ROOT") ENV_DIR=$(printf '%q' "$ENV_DIR") DATA_ROOT=$(printf '%q' "$DATA_ROOT") PREPARED_DATA=$(printf '%q' "$PREPARED_DATA") SETTINGS_FILE=$(printf '%q' "$SETTINGS_FILE") SESSION_NAME=$(printf '%q' "$SESSION_NAME") RUN_DIR=$(printf '%q' "$RUN_DIR") bash -lc $(printf '%q' "$remote_script")"
