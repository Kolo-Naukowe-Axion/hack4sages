#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

REMOTE_HOST="${REMOTE_HOST:-iwo@100.103.127.124}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/iwo/hack4sages-crossgen}"

rsync -az \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '.venv-*' \
  --exclude '.codex-venv' \
  --exclude '.local-prt' \
  --exclude '__pycache__' \
  --exclude '.DS_Store' \
  --exclude '.pytest_cache' \
  --exclude '.mypy_cache' \
  "${PROJECT_ROOT}/" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/"

echo "Synced ${PROJECT_ROOT} -> ${REMOTE_HOST}:${REMOTE_ROOT}"
