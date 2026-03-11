#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${DATA_ROOT}/.." && pwd)"

OUTPUT_ROOT="${1:-${DATA_ROOT}/generated-data/transmission_benchmark_$(date +%Y%m%d_%H%M%S)}"
VENV_PATH="${PRT_BENCH_VENV_PATH:-${WORKSPACE_ROOT}/.venv-prt311}"
PYTHON_BIN="${VENV_PATH}/bin/python"
INPUT_DATA_PATH="${PRT_BENCH_INPUT_DATA_PATH:-${WORKSPACE_ROOT}/.local-prt/input_data}"
SAMPLE_COUNT="${PRT_BENCH_SAMPLE_COUNT:-150000}"
SHARD_SIZE="${PRT_BENCH_SHARD_SIZE:-512}"
WORKERS="${PRT_BENCH_WORKERS:-16}"
PROGRESS_UPDATE_SECONDS="${PRT_BENCH_PROGRESS_UPDATE_SECONDS:-10}"
GENERATE_ONLY="${PRT_BENCH_GENERATE_ONLY:-0}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python interpreter at ${PYTHON_BIN}" >&2
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
  "${DATA_ROOT}/scripts/generate_transmission_benchmark.py"
  "--output-root" "${OUTPUT_ROOT}"
  "--p-rt-input-data-path" "${INPUT_DATA_PATH}"
  "--sample-count" "${SAMPLE_COUNT}"
  "--shard-size" "${SHARD_SIZE}"
  "--workers" "${WORKERS}"
  "--progress-update-seconds" "${PROGRESS_UPDATE_SECONDS}"
  "--skip-existing"
)

if [[ "${GENERATE_ONLY}" == "1" ]]; then
  CMD+=("--generate-only")
fi

RUNNER=()
if command -v caffeinate >/dev/null 2>&1; then
  RUNNER=(caffeinate -dimsu)
fi

nohup env \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  PRT_BENCH_LOG_PATH="${LOG_PATH}" \
  MPLCONFIGDIR="${OUTPUT_ROOT}/work/mplconfig" \
  "${RUNNER[@]}" \
  "${CMD[@]}" >"${LOG_PATH}" 2>&1 &

PID="$!"
echo "${PID}" > "${PID_PATH}"

echo "Launched transmission benchmark generation"
echo "  output_root: ${OUTPUT_ROOT}"
echo "  pid: ${PID}"
echo "  log: ${LOG_PATH}"
echo "  progress: ${OUTPUT_ROOT}/work/progress.json"
echo
echo "Monitor once:"
echo "  ${PYTHON_BIN} ${DATA_ROOT}/scripts/check_generation_status.py --output-root ${OUTPUT_ROOT}"
echo
echo "Monitor live:"
echo "  ${PYTHON_BIN} ${DATA_ROOT}/scripts/check_generation_status.py --output-root ${OUTPUT_ROOT} --watch"
