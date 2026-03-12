#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PREPARED_DIR="${PREPARED_DIR:-data/generated-data/ariel_winner_trace_nf_prepared}"
RUN_DIR="${RUN_DIR:-local_runs/ariel_winner_trace_nf}"
SETTINGS_PATH="${SETTINGS_PATH:-models/ariel_winner_trace_nf/settings/winner_trace_independent_nsf.yaml}"

python3 -m models.ariel_winner_trace_nf.prepare_dataset \
  --data-root data/full-ariel \
  --split-source data/val_dataset \
  --output "$PREPARED_DIR" \
  --overwrite

python3 -m models.ariel_winner_trace_nf.train \
  --settings "$SETTINGS_PATH" \
  --prepared-data "$PREPARED_DIR" \
  --run-dir "$RUN_DIR"
