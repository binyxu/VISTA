#!/usr/bin/env bash
set -euo pipefail

METHOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWE_BENCH_ROOT="$(cd "${METHOD_DIR}/../.." && pwd)"

export SM_STRICT_LONG_CONTEXT=0
export SM_BETTER_DASHBOARD=0
export LOCA_OPENAI_BASE_URL="${LOCA_OPENAI_BASE_URL:?Set LOCA_OPENAI_BASE_URL}"
export LOCA_OPENAI_API_KEY="${LOCA_OPENAI_API_KEY:?Set LOCA_OPENAI_API_KEY}"

MODEL="${MODEL:-deepseek-v4-pro}"
DATASET_NAME="${DATASET_NAME:-SWE-bench/SWE-bench_Verified}"
SPLIT="${SPLIT:-test}"
RUN_ID="${RUN_ID:-react_baseline}"
OUTPUT_DIR="${OUTPUT_DIR:-${SWE_BENCH_ROOT}/outputs/${RUN_ID}}"
MAX_CONTEXT_SIZE="${MAX_CONTEXT_SIZE:-128000}"
MAX_WORKERS="${MAX_WORKERS:-1}"
PYTHON_BIN="${PYTHON:-}"

if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "Could not find python3 or python. Set PYTHON=/path/to/python." >&2
    exit 127
  fi
fi

mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" "${METHOD_DIR}/strict_lc_better_dashboard.py" \
  --dataset_name "${DATASET_NAME}" \
  --split "${SPLIT}" \
  --model "${MODEL}" \
  --output_file "${OUTPUT_DIR}/predictions.jsonl" \
  --max_context_size "${MAX_CONTEXT_SIZE}" \
  --config "${METHOD_DIR}/configs/react_baseline.json" \
  "$@"

cat <<EOF

Predictions written to:
  ${OUTPUT_DIR}/predictions.jsonl

Evaluate with:
  cd "${SWE_BENCH_ROOT}"
  ${PYTHON_BIN} -m swebench.harness.run_evaluation \\
    --dataset_name "${DATASET_NAME}" \\
    --split "${SPLIT}" \\
    --predictions_path "${OUTPUT_DIR}/predictions.jsonl" \\
    --max_workers "${MAX_WORKERS}" \\
    --run_id "${RUN_ID}"

On Apple Silicon, add:
    --namespace ''
EOF
