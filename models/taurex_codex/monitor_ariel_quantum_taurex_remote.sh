#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:?Set REMOTE_HOST to the Vast SSH target before monitoring.}"
REMOTE_SSH_ARGS="${REMOTE_SSH_ARGS:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT to the remote run directory first.}"
TARGET="${1:-stage2}"

case "$TARGET" in
  stage1)
    LOG_PATH="$OUTPUT_ROOT/stage1_classical/train.log"
    ;;
  stage2)
    LOG_PATH="$OUTPUT_ROOT/stage2_hybrid/train.log"
    ;;
  launcher)
    LOG_PATH="$OUTPUT_ROOT/launcher.log"
    ;;
  *)
    echo "Unknown target: $TARGET" >&2
    echo "Expected one of: stage1, stage2, launcher" >&2
    exit 1
    ;;
esac

ssh ${REMOTE_SSH_ARGS} -t "$REMOTE_HOST" "bash -lc 'if [[ -f \"$LOG_PATH\" ]]; then tail -n 200 -f \"$LOG_PATH\"; else echo \"Missing log: $LOG_PATH\" >&2; exit 1; fi'"
