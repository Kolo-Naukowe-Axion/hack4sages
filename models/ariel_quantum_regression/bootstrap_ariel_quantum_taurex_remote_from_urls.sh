#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:?Set REMOTE_HOST to the Vast SSH target before bootstrapping.}"
SRC_URL="${SRC_URL:?Set SRC_URL to a tarball containing models/ariel_quantum_regression.}"
LABELS_URL="${LABELS_URL:?Set LABELS_URL to the TauREx labels.parquet download URL.}"
SPECTRA_URL="${SPECTRA_URL:?Set SPECTRA_URL to the TauREx spectra.h5 download URL.}"

REMOTE_SSH_ARGS="${REMOTE_SSH_ARGS:-}"
REMOTE_ROOT="${REMOTE_ROOT:-/workspace/hack4sages}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-$REMOTE_ROOT/data/TauREx_set}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REMOTE_ROOT/outputs/ariel_quantum_taurex_$(date +%Y%m%d_%H%M%S)}"
SESSION_NAME="${SESSION_NAME:-aqr_taurex_train}"
ENV_DIR="${ENV_DIR:-/opt/aqr-taurex}"

remote_script=$(cat <<'EOF'
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y python3-venv git curl ca-certificates tmux rsync aria2

mkdir -p "$REMOTE_ROOT" "$REMOTE_ROOT/models" "$REMOTE_DATA_ROOT" "$REMOTE_ROOT/outputs"
rm -rf "$REMOTE_ROOT/models/ariel_quantum_regression"

download_file() {
  local url="$1"
  local destination="$2"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c \
      --allow-overwrite=true \
      --auto-file-renaming=false \
      --file-allocation=none \
      --max-connection-per-server=16 \
      --split=16 \
      --min-split-size=1M \
      --dir="$(dirname "$destination")" \
      --out="$(basename "$destination")" \
      "$url"
  else
    curl -fL --retry 5 --retry-all-errors "$url" -o "$destination"
  fi
}

download_file "$SRC_URL" /tmp/ariel_quantum_regression_remote.tgz
download_file "$LABELS_URL" "$REMOTE_DATA_ROOT/labels.parquet"
download_file "$SPECTRA_URL" "$REMOTE_DATA_ROOT/spectra.h5"
tar -xzf /tmp/ariel_quantum_regression_remote.tgz -C "$REMOTE_ROOT"

ls -lh "$REMOTE_DATA_ROOT/labels.parquet" "$REMOTE_DATA_ROOT/spectra.h5"
mkdir -p "$OUTPUT_ROOT"

if command -v tmux >/dev/null 2>&1; then
  tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
  tmux new-session -d -s "$SESSION_NAME" \
    "cd $REMOTE_ROOT && DATA_ROOT=$(printf '%q' "$REMOTE_DATA_ROOT") OUTPUT_ROOT=$(printf '%q' "$OUTPUT_ROOT") ENV_DIR=$(printf '%q' "$ENV_DIR") bash models/ariel_quantum_regression/run_ariel_quantum_taurex_ubuntu_gpu.sh"
  echo "Started remote tmux session: $SESSION_NAME"
  echo "Attach with: tmux attach -t $SESSION_NAME"
else
  setsid -f bash -lc \
    "cd $(printf '%q' "$REMOTE_ROOT") && \
     exec env DATA_ROOT=$(printf '%q' "$REMOTE_DATA_ROOT") OUTPUT_ROOT=$(printf '%q' "$OUTPUT_ROOT") ENV_DIR=$(printf '%q' "$ENV_DIR") \
     bash $(printf '%q' "$REMOTE_ROOT/models/ariel_quantum_regression/run_ariel_quantum_taurex_ubuntu_gpu.sh") \
     > $(printf '%q' "$OUTPUT_ROOT/launcher.log") 2>&1 < /dev/null"
  echo "Started remote background process without tmux."
fi

echo "Run directory: $OUTPUT_ROOT"
echo "Stage 1 log: $OUTPUT_ROOT/stage1_classical/train.log"
echo "Stage 2 log: $OUTPUT_ROOT/stage2_hybrid/train.log"
EOF
)

ssh ${REMOTE_SSH_ARGS} -t "$REMOTE_HOST" \
  "REMOTE_ROOT=$(printf '%q' "$REMOTE_ROOT") REMOTE_DATA_ROOT=$(printf '%q' "$REMOTE_DATA_ROOT") OUTPUT_ROOT=$(printf '%q' "$OUTPUT_ROOT") SESSION_NAME=$(printf '%q' "$SESSION_NAME") ENV_DIR=$(printf '%q' "$ENV_DIR") SRC_URL=$(printf '%q' "$SRC_URL") LABELS_URL=$(printf '%q' "$LABELS_URL") SPECTRA_URL=$(printf '%q' "$SPECTRA_URL") bash -lc $(printf '%q' "$remote_script")"
