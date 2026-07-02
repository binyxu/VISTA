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
B="${B:-196608}"
LIMIT="${LIMIT:-20}"
PORT="${PORT:-8091}"
THREADS="${THREADS:-5}"
RUN_ID="${RUN_ID:-gaia_W${W}_B${B}_$(date +%Y%m%d_%H%M%S)}"
CORPUS_DIR="${CORPUS_DIR:-data/stress}"

if [[ -z "${OPENAI_BASE_URL}" || -z "${OPENAI_API_KEY}" ]]; then
  echo "Set OPENAI_BASE_URL/OPENAI_API_KEY or LOCA_OPENAI_BASE_URL/LOCA_OPENAI_API_KEY." >&2
  exit 1
fi

if [[ ! -f data/gaia_2023_all_validation.jsonl ]]; then
  echo "Missing data/gaia_2023_all_validation.jsonl. Run: HF_TOKEN=... ./scripts/download_gaia.py" >&2
  exit 1
fi

"$PYTHON" scripts/build_stress_corpus.py \
  --input data/gaia_2023_all_validation.jsonl \
  --out-dir "$CORPUS_DIR" \
  --limit "$LIMIT" \
  --window "$W"

if lsof -iTCP:"$PORT" -sTCP:LISTEN -n -P >/dev/null 2>&1; then
  echo "Port $PORT already in use; choose another PORT." >&2
  exit 1
fi

"$PYTHON" searcher/local_gaia_mcp_server.py \
  --corpus "${CORPUS_DIR}/corpus.jsonl" \
  --port "$PORT" \
  --k 4 \
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
  --query "${CORPUS_DIR}/queries.tsv" \
  --output-dir "runs/${RUN_ID}_vista" \
  --max-context-size "$W" \
  --max-total-tokens "$B" \
  --max-tokens 10000 \
  --max-steps 0 \
  --num-threads "$THREADS" \
  --query-template QUERY_TEMPLATE \
  --verbose

"$PYTHON" "${BROWSECOMP_ROOT}/search_agent/gemini_vista_client.py" \
  --mode react \
  --react-cm truncate \
  --model "$MODEL" \
  --mcp-url "http://127.0.0.1:${PORT}/mcp" \
  --query "${CORPUS_DIR}/queries.tsv" \
  --output-dir "runs/${RUN_ID}_react" \
  --max-context-size "$W" \
  --max-total-tokens "$B" \
  --max-tokens 10000 \
  --max-steps 0 \
  --num-threads "$THREADS" \
  --query-template QUERY_TEMPLATE \
  --verbose

"$PYTHON" scripts/judge_gaia_api.py \
  --input-dir "runs/${RUN_ID}_vista" \
  --ground-truth "${CORPUS_DIR}/ground_truth.jsonl" \
  --label "${RUN_ID}_vista" \
  --out "runs/${RUN_ID}_vista/judge_summary.json"

"$PYTHON" scripts/judge_gaia_api.py \
  --input-dir "runs/${RUN_ID}_react" \
  --ground-truth "${CORPUS_DIR}/ground_truth.jsonl" \
  --label "${RUN_ID}_react" \
  --out "runs/${RUN_ID}_react/judge_summary.json"

echo "DONE RUN_ID=$RUN_ID"
