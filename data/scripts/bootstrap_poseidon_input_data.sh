#!/usr/bin/env bash
set -euo pipefail

POSEIDON_INPUT_DATA="${POSEIDON_INPUT_DATA:-$HOME/poseidon-inputs}"
POSEIDON_DOWNLOAD_DIR="${POSEIDON_DOWNLOAD_DIR:-$HOME/poseidon-inputs-download}"
POSEIDON_INPUTS_URL="${POSEIDON_INPUTS_URL:-https://zenodo.org/records/16107813/files/inputs.zip?download=1}"
POSEIDON_ZIP_PATH="$POSEIDON_DOWNLOAD_DIR/inputs.zip"
POSEIDON_UNPACK_DIR="$POSEIDON_DOWNLOAD_DIR/unpacked"

if [[ -f "$POSEIDON_INPUT_DATA/opacity/Opacity_database_v1.3.hdf5" ]]; then
  exit 0
fi

mkdir -p "$POSEIDON_DOWNLOAD_DIR"
curl -L -C - "$POSEIDON_INPUTS_URL" -o "$POSEIDON_ZIP_PATH"

rm -rf "$POSEIDON_UNPACK_DIR"
mkdir -p "$POSEIDON_UNPACK_DIR"
unzip -q -o "$POSEIDON_ZIP_PATH" -d "$POSEIDON_UNPACK_DIR"

rm -rf "$POSEIDON_INPUT_DATA"
if [[ -d "$POSEIDON_UNPACK_DIR/inputs" ]]; then
  mv "$POSEIDON_UNPACK_DIR/inputs" "$POSEIDON_INPUT_DATA"
  rmdir "$POSEIDON_UNPACK_DIR" || true
else
  mv "$POSEIDON_UNPACK_DIR" "$POSEIDON_INPUT_DATA"
fi

rm -f "$POSEIDON_ZIP_PATH"
