#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUTPUT_ROOT="${1:-${REPO_ROOT}/local_runs/full_20000_mac_$(date +%Y%m%d_%H%M%S)}"
VENV_PATH="${PRT_ADC_VENV_PATH:-${REPO_ROOT}/.venv-prt311}"
PYTHON_BIN="${VENV_PATH}/bin/python"
BUNDLE_PATH="${PRT_ADC_BUNDLE_PATH:-${REPO_ROOT}/data/reference_data/adc2023_reference_bundle.npz}"
INPUT_DATA_PATH="${PRT_ADC_INPUT_DATA_PATH:-${REPO_ROOT}/.local-prt/input_data}"
SAMPLE_COUNT="${PRT_ADC_SAMPLE_COUNT:-20000}"
SHARD_SIZE="${PRT_ADC_SHARD_SIZE:-250}"
WORKERS="${PRT_ADC_WORKERS:-3}"
PROGRESS_UPDATE_SECONDS="${PRT_ADC_PROGRESS_UPDATE_SECONDS:-10}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python interpreter at ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "${BUNDLE_PATH}" ]]; then
  echo "Missing reference bundle at ${BUNDLE_PATH}" >&2
  exit 1
fi

if [[ ! -d "${INPUT_DATA_PATH}" ]]; then
  echo "Missing pRT input data directory at ${INPUT_DATA_PATH}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}/work"
LOG_PATH="${OUTPUT_ROOT}/run.log"
PID_PATH="${OUTPUT_ROOT}/work/generator.pid"

CMD=(
  "${PYTHON_BIN}"
  "${REPO_ROOT}/scripts/generate_validation_set.py"
  "--output-root" "${OUTPUT_ROOT}"
  "--bundle-path" "${BUNDLE_PATH}"
  "--p-rt-input-data-path" "${INPUT_DATA_PATH}"
  "--sample-count" "${SAMPLE_COUNT}"
  "--shard-size" "${SHARD_SIZE}"
  "--workers" "${WORKERS}"
  "--progress-update-seconds" "${PROGRESS_UPDATE_SECONDS}"
  "--skip-existing"
)

RUNNER=()
if command -v caffeinate >/dev/null 2>&1; then
  RUNNER=(caffeinate -dimsu)
fi

nohup env \
  PRT_ADC_LOG_PATH="${LOG_PATH}" \
  MPLCONFIGDIR="${OUTPUT_ROOT}/work/mplconfig" \
  "${RUNNER[@]}" \
  "${CMD[@]}" >"${LOG_PATH}" 2>&1 &

PID="$!"
echo "${PID}" > "${PID_PATH}"

echo "Launched local generation"
echo "  output_root: ${OUTPUT_ROOT}"
echo "  pid: ${PID}"
echo "  log: ${LOG_PATH}"
echo "  progress: ${OUTPUT_ROOT}/work/progress.json"
echo
echo "Monitor once:"
echo "  ${PYTHON_BIN} ${REPO_ROOT}/scripts/check_generation_status.py --output-root ${OUTPUT_ROOT}"
echo
echo "Monitor live:"
echo "  ${PYTHON_BIN} ${REPO_ROOT}/scripts/check_generation_status.py --output-root ${OUTPUT_ROOT} --watch"
