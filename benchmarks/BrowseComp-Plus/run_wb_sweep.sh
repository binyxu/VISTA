#!/usr/bin/env bash
# Parametrized react-vs-vista sweep. Override via env:
#   W (max-context-size), B (max-total-tokens), MODEL, PORT, QUERIES, RUN_ID, THREADS,
#   TIMEOUT_SECONDS (per-query wall-clock timeout; 0 disables),
#   TOOL_TIMEOUT_SECONDS (per-MCP-tool-attempt timeout), TOOL_RETRIES, QUERY_RETRIES
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BROWSECOMP_ROOT="${BROWSECOMP_ROOT:-${SCRIPT_DIR}}"
cd "$BROWSECOMP_ROOT"

if [[ -f ./.env ]]; then
  set -a
  . ./.env
  set +a
fi
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-${LOCA_OPENAI_BASE_URL:-}}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-${LOCA_OPENAI_API_KEY:-}}"

W="${W:-12288}"
B="${B:-163840}"
MODEL="${MODEL:-deepseek-v4-pro-qcloud}"
PORT="${PORT:-8084}"
QUERIES="${QUERIES:-topics-qrels/queries_subset30.tsv}"
THREADS="${THREADS:-5}"
RUN_ID="${RUN_ID:-wb_$(date +%Y%m%d_%H%M%S)}"
ARCHIVE_PLACEHOLDER_STYLE="${ARCHIVE_PLACEHOLDER_STYLE:-metadata}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-0}"
TOOL_TIMEOUT_SECONDS="${TOOL_TIMEOUT_SECONDS:-180}"
TOOL_RETRIES="${TOOL_RETRIES:-2}"
QUERY_RETRIES="${QUERY_RETRIES:-1}"

mkdir -p runs

if [[ "$RUN_ID" == *"_budget_"* && "$W" == "12288" && "$B" != "163840" ]]; then
  exec bash run_wb_budget_memoized.sh
fi

if lsof -iTCP:"$PORT" -sTCP:LISTEN -n -P >/dev/null 2>&1; then
  echo "Port $PORT already in use; pick another PORT." >&2
  exit 1
fi

.venv/bin/python searcher/mcp_server_api_dense.py \
  --index-path 'indexes/qwen3-embedding-8b/corpus.shard*.pkl' \
  --port "$PORT" --k 5 --snippet-max-tokens 512 \
  > "runs/${RUN_ID}_mcp.log" 2>&1 &
MCP_PID=$!
echo "MCP_PID=$MCP_PID RUN_ID=$RUN_ID W=$W B=$B MODEL=$MODEL PORT=$PORT TIMEOUT_SECONDS=$TIMEOUT_SECONDS TOOL_TIMEOUT_SECONDS=$TOOL_TIMEOUT_SECONDS TOOL_RETRIES=$TOOL_RETRIES QUERY_RETRIES=$QUERY_RETRIES"

cleanup() { echo "Stopping MCP server $MCP_PID"; kill "$MCP_PID" 2>/dev/null || true; }
trap cleanup EXIT

for _ in {1..60}; do
  lsof -iTCP:"$PORT" -sTCP:LISTEN -n -P >/dev/null 2>&1 && { echo "MCP listening on $PORT"; break; }
  sleep 2
done
lsof -iTCP:"$PORT" -sTCP:LISTEN -n -P >/dev/null 2>&1 || { echo "MCP failed to listen" >&2; exit 1; }

for MODE in react vista; do
  if [[ -f "runs/${RUN_ID}_${MODE}/judge_summary.json" ]]; then
    echo "[skip] ${RUN_ID}_${MODE} already judged"
    continue
  fi
  .venv/bin/python search_agent/gemini_vista_client.py \
    --mode "$MODE" \
    --model "$MODEL" \
    --mcp-url "http://127.0.0.1:${PORT}/mcp" \
    --query "$QUERIES" \
    --output-dir "runs/${RUN_ID}_${MODE}" \
    --max-context-size "$W" \
    --max-total-tokens "$B" \
    --max-tokens 10000 \
    --max-steps 0 \
    --timeout-seconds "$TIMEOUT_SECONDS" \
    --tool-timeout-seconds "$TOOL_TIMEOUT_SECONDS" \
    --tool-retries "$TOOL_RETRIES" \
    --query-retries "$QUERY_RETRIES" \
    --archive-placeholder-style "$ARCHIVE_PLACEHOLDER_STYLE" \
    --num-threads "$THREADS" \
    --verbose
  .venv/bin/python scripts_evaluation/judge_api.py \
    --input_dir "runs/${RUN_ID}_${MODE}" \
    --label "${RUN_ID}_${MODE}"
done

echo "DONE RUN_ID=$RUN_ID"
