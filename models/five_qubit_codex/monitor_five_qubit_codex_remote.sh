#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-tail}"
REMOTE_HOST="${REMOTE_HOST:-iwo@100.103.127.124}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/iwo/hack4sages-crossgen}"
SESSION_NAME="${SESSION_NAME:-five_qubit_codex}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"

remote_script=$(cat <<'EOF'
set -euo pipefail

resolve_output_root() {
  if [[ -n "\$OUTPUT_ROOT" ]]; then
    printf '%s\n' "\$OUTPUT_ROOT"
    return
  fi
  ls -td "\$REMOTE_ROOT"/outputs/five_qubit_codex_two_stage_* 2>/dev/null | head -n 1
}

case "\$MODE" in
  attach)
    exec tmux attach -t "\$SESSION_NAME"
    ;;
  tail)
    output_root="\$(resolve_output_root)"
    if [[ -z "\$output_root" ]]; then
      echo "No five_qubit_codex run directory found under \$REMOTE_ROOT/outputs" >&2
      exit 1
    fi
    if [[ -f "\$output_root/stage2_hybrid/train.log" ]]; then
      log_path="\$output_root/stage2_hybrid/train.log"
    elif [[ -f "\$output_root/stage1_classical/train.log" ]]; then
      log_path="\$output_root/stage1_classical/train.log"
    else
      echo "No train.log found yet under \$output_root" >&2
      exit 1
    fi
    echo "Previewing \$log_path" >&2
    exec tail -F "\$log_path"
    ;;
  status)
    output_root="\$(resolve_output_root)"
    echo "Session: \$SESSION_NAME"
    echo "Output root: \${output_root:-missing}"
    tmux list-sessions 2>/dev/null | grep -F "\$SESSION_NAME" || true
    ;;
  *)
    echo "Usage: monitor_five_qubit_codex_remote.sh [tail|attach|status]" >&2
    exit 1
    ;;
esac
EOF
)

ssh -t "$REMOTE_HOST" \
  "MODE=$(printf '%q' "$MODE") REMOTE_ROOT=$(printf '%q' "$REMOTE_ROOT") SESSION_NAME=$(printf '%q' "$SESSION_NAME") OUTPUT_ROOT=$(printf '%q' "$OUTPUT_ROOT") bash -lc $(printf '%q' "$remote_script")"
