#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-tail}"
REMOTE_HOST="${REMOTE_HOST:-iwo@100.103.127.124}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/iwo/hack4sages-crossgen}"
SESSION_NAME="${SESSION_NAME:-ariel_winner_on_taurex}"
RUN_DIR="${RUN_DIR:-}"

remote_script=$(cat <<'EOF'
set -euo pipefail

resolve_run_dir() {
  if [[ -n "$RUN_DIR" ]]; then
    printf '%s\n' "$RUN_DIR"
    return
  fi
  ls -td "$REMOTE_ROOT"/local_runs/ariel_winner_on_taurex_* 2>/dev/null | head -n 1
}

case "$MODE" in
  attach)
    if ! command -v tmux >/dev/null 2>&1; then
      echo "tmux is not installed on the remote host." >&2
      exit 1
    fi
    exec tmux attach -t "$SESSION_NAME"
    ;;
  tail)
    run_dir="$(resolve_run_dir)"
    if [[ -z "$run_dir" ]]; then
      echo "No ariel_winner_on_taurex run directory found under $REMOTE_ROOT/local_runs" >&2
      exit 1
    fi
    if [[ -f "$run_dir/train.log" ]]; then
      log_path="$run_dir/train.log"
    else
      log_path="$run_dir/launcher.log"
    fi
    if [[ ! -f "$log_path" ]]; then
      echo "No log found yet under $run_dir" >&2
      exit 1
    fi
    echo "Previewing $log_path" >&2
    exec tail -F "$log_path"
    ;;
  status)
    run_dir="$(resolve_run_dir)"
    echo "Session: $SESSION_NAME"
    echo "Run directory: ${run_dir:-missing}"
    if command -v tmux >/dev/null 2>&1; then
      tmux list-sessions 2>/dev/null | grep -F "$SESSION_NAME" || true
    fi
    if [[ -n "$run_dir" && -f "$run_dir/train.pid" ]]; then
      pid="$(cat "$run_dir/train.pid")"
      echo "PID: $pid"
      ps -p "$pid" -o pid=,etime=,cmd= || true
    fi
    ;;
  *)
    echo "Usage: monitor_ariel_winner_on_taurex_remote.sh [tail|attach|status]" >&2
    exit 1
    ;;
esac
EOF
)

ssh -t "$REMOTE_HOST" \
  "MODE=$(printf '%q' "$MODE") REMOTE_ROOT=$(printf '%q' "$REMOTE_ROOT") SESSION_NAME=$(printf '%q' "$SESSION_NAME") RUN_DIR=$(printf '%q' "$RUN_DIR") bash -lc $(printf '%q' "$remote_script")"
