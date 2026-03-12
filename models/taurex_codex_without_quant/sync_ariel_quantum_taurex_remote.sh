#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

REMOTE_HOST="${REMOTE_HOST:?Set REMOTE_HOST to the Vast SSH target, for example root@ssh7.vast.ai -p 40123 via ~/.ssh/config or ssh alias.}"
REMOTE_SSH_ARGS="${REMOTE_SSH_ARGS:-}"
REMOTE_ROOT="${REMOTE_ROOT:-/workspace/hack4sages}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-$REMOTE_ROOT/data/TauREx set}"

ssh ${REMOTE_SSH_ARGS} "$REMOTE_HOST" "mkdir -p '$REMOTE_ROOT/models' '$REMOTE_ROOT/data'"

rsync -a --delete --info=progress2 -e "ssh ${REMOTE_SSH_ARGS}" \
  "${PROJECT_ROOT}/models/taurex_codex_without_quant/" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/models/taurex_codex_without_quant/"

rsync -a --delete --info=progress2 -e "ssh ${REMOTE_SSH_ARGS}" \
  "${PROJECT_ROOT}/data/TauREx set/" \
  "${REMOTE_HOST}:${REMOTE_DATA_ROOT}/"

echo "Synced quantum model -> ${REMOTE_HOST}:${REMOTE_ROOT}/models/taurex_codex_without_quant"
echo "Synced TauREx bundle -> ${REMOTE_HOST}:${REMOTE_DATA_ROOT}"
