#!/usr/bin/env bash
set -euo pipefail

REMOTE_TARGET="${1:-iwo@100.103.127.124:~/hack4sages-crossgen}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

rsync -az \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '.venv-*' \
  --exclude '.exobiome-venv' \
  --exclude '.local-prt' \
  --exclude '__pycache__' \
  --exclude '.DS_Store' \
  --exclude '.pytest_cache' \
  --exclude '.mypy_cache' \
  "$LOCAL_ROOT/" \
  "$REMOTE_TARGET/"
