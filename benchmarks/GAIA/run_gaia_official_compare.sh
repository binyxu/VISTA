#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

BROWSECOMP_ROOT="${BROWSECOMP_ROOT:-../BrowseComp-Plus}"
PYTHON="${PYTHON:-${BROWSECOMP_ROOT}/.venv/bin/python}"

if [[ -f "${BROWSECOMP_ROOT}/.vista_env" ]]; then
  set -a; . "${BROWSECOMP_ROOT}/.vista_env"; set +a
fi
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-${LOCA_OPENAI_BASE_URL:-}}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-${LOCA_OPENAI_API_KEY:-}}"
export LOCABENCH_ROOT="${LOCABENCH_ROOT:-$(cd .. && pwd)/LOCAbench}"

MODEL="${MODEL:-deepseek-v4-pro-qcloud}"
W="${W:-12288}"
B="${B:-80000}"
LIMIT="${LIMIT:-5}"
PORT="${PORT:-8111}"
THREADS="${THREADS:-5}"
MAX_STEPS="${MAX_STEPS:-54}"
MAX_RESULT_CONTEXT_FRACTION="${MAX_RESULT_CONTEXT_FRACTION:-0.80}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-3600}"
RUN_ID="${RUN_ID:-gaia_official_L${LIMIT}_W${W}_B${B}_$(date +%Y%m%d_%H%M%S)}"
DATA_DIR="${DATA_DIR:-data/official/validation_${LIMIT}}"
ATTACHMENTS_DIR="${ATTACHMENTS_DIR:-data/attachments}"
TASK_IDS="${TASK_IDS:-}"
SAMPLE_SIZE="${SAMPLE_SIZE:-0}"
SEED="${SEED:-42}"

if [[ -z "${OPENAI_BASE_URL}" || -z "${OPENAI_API_KEY}" ]]; then
  echo "Set OPENAI_BASE_URL/OPENAI_API_KEY or LOCA_OPENAI_BASE_URL/LOCA_OPENAI_API_KEY." >&2
  exit 1
fi

BUILD_ARGS=(--input data/gaia_2023_all_validation.jsonl --out-dir "$DATA_DIR" --limit "$LIMIT" --sample-size "$SAMPLE_SIZE" --seed "$SEED")
if [[ -n "$TASK_IDS" ]]; then
  BUILD_ARGS+=(--task-ids "$TASK_IDS")
fi
"$PYTHON" scripts/build_official_dataset.py "${BUILD_ARGS[@]}"

if [[ ! -d "$ATTACHMENTS_DIR" ]]; then
  mkdir -p "$ATTACHMENTS_DIR"
fi

if lsof -iTCP:"$PORT" -sTCP:LISTEN -n -P >/dev/null 2>&1; then
  echo "Port $PORT already in use; choose another PORT." >&2
  exit 1
fi

"$PYTHON" gaia_tools/mcp_server.py \
  --attachments-dir "$ATTACHMENTS_DIR" \
  --work-dir "runs/${RUN_ID}_tool_workspace" \
  --port "$PORT" \
  > "runs/${RUN_ID}_mcp.log" 2>&1 &
MCP_PID=$!
cleanup() { kill "$MCP_PID" 2>/dev/null || true; }
trap cleanup EXIT

for _ in {1..60}; do
  lsof -iTCP:"$PORT" -sTCP:LISTEN -n -P >/dev/null 2>&1 && break
  sleep 1
done
lsof -iTCP:"$PORT" -sTCP:LISTEN -n -P >/dev/null 2>&1 || {
  echo "MCP server failed to listen; see runs/${RUN_ID}_mcp.log" >&2
  exit 1
}

echo "RUN_ID=$RUN_ID MODEL=$MODEL W=$W B=$B LIMIT=$LIMIT"

"$PYTHON" "${BROWSECOMP_ROOT}/search_agent/gemini_vista_client.py" \
  --mode vista \
  --model "$MODEL" \
  --mcp-url "http://127.0.0.1:${PORT}/mcp" \
  --query "${DATA_DIR}/queries.tsv" \
  --output-dir "runs/${RUN_ID}_vista" \
  --max-context-size "$W" \
  --max-total-tokens "$B" \
  --max-result-context-fraction "$MAX_RESULT_CONTEXT_FRACTION" \
  --max-tokens 10000 \
  --max-steps "$MAX_STEPS" \
  --timeout-seconds "$TIMEOUT_SECONDS" \
  --num-threads "$THREADS" \
  --query-template GAIA_TEMPLATE \
  --verbose

"$PYTHON" "${BROWSECOMP_ROOT}/search_agent/gemini_vista_client.py" \
  --mode react \
  --react-cm truncate \
  --model "$MODEL" \
  --mcp-url "http://127.0.0.1:${PORT}/mcp" \
  --query "${DATA_DIR}/queries.tsv" \
  --output-dir "runs/${RUN_ID}_react" \
  --max-context-size "$W" \
  --max-total-tokens "$B" \
  --max-result-context-fraction "$MAX_RESULT_CONTEXT_FRACTION" \
  --max-tokens 10000 \
  --max-steps "$MAX_STEPS" \
  --timeout-seconds "$TIMEOUT_SECONDS" \
  --num-threads "$THREADS" \
  --query-template GAIA_TEMPLATE \
  --verbose

"$PYTHON" scripts/score_gaia_official.py \
  --input-dir "runs/${RUN_ID}_vista" \
  --ground-truth "${DATA_DIR}/ground_truth.jsonl" \
  --out "runs/${RUN_ID}_vista/official_score.json"

"$PYTHON" scripts/score_gaia_official.py \
  --input-dir "runs/${RUN_ID}_react" \
  --ground-truth "${DATA_DIR}/ground_truth.jsonl" \
  --out "runs/${RUN_ID}_react/official_score.json"

echo "DONE RUN_ID=$RUN_ID"
