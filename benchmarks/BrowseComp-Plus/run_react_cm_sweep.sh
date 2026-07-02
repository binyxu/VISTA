#!/usr/bin/env bash
# Fixed-policy ReAct-family baselines for BrowseComp-Plus (Table 1).
# Runs react with --react-cm in {clear, mask, skeleton} at the same W/B/model as
# the main react-vs-vista sweep, then judges each. Override via env:
#   W, B, MODEL, PORT, QUERIES, THREADS, RUN_ID, CMS
set -euo pipefail
cd ${BROWSECOMP_ROOT:-.}

if [[ -f ./.env ]]; then if [[ -f ./.env ]]; then set -a; . ./.env; set +a; fi; fi
export OPENAI_BASE_URL="$LOCA_OPENAI_BASE_URL"
export OPENAI_API_KEY="$LOCA_OPENAI_API_KEY"

W="${W:-12288}"
B="${B:-163840}"
MODEL="${MODEL:-deepseek-v4-pro-qcloud}"
PORT="${PORT:-8087}"
QUERIES="${QUERIES:-topics-qrels/queries_model8.tsv}"
THREADS="${THREADS:-4}"
RUN_ID="${RUN_ID:-reactcm_$(date +%Y%m%d_%H%M%S)}"
CMS="${CMS:-clear mask skeleton}"

mkdir -p runs

if lsof -iTCP:"$PORT" -sTCP:LISTEN -n -P >/dev/null 2>&1; then
  echo "Port $PORT already in use; pick another PORT." >&2; exit 1
fi

.venv/bin/python searcher/mcp_server_api_dense.py \
  --index-path 'indexes/qwen3-embedding-8b/corpus.shard*.pkl' \
  --port "$PORT" --k 5 --snippet-max-tokens 512 \
  > "runs/${RUN_ID}_mcp.log" 2>&1 &
MCP_PID=$!
echo "MCP_PID=$MCP_PID RUN_ID=$RUN_ID W=$W B=$B MODEL=$MODEL PORT=$PORT CMS='$CMS'"
cleanup() { echo "Stopping MCP server $MCP_PID"; kill "$MCP_PID" 2>/dev/null || true; }
trap cleanup EXIT

for _ in {1..60}; do
  lsof -iTCP:"$PORT" -sTCP:LISTEN -n -P >/dev/null 2>&1 && { echo "MCP listening on $PORT"; break; }
  sleep 2
done
lsof -iTCP:"$PORT" -sTCP:LISTEN -n -P >/dev/null 2>&1 || { echo "MCP failed to listen" >&2; exit 1; }

for CM in $CMS; do
  echo "########## react-cm=$CM ##########"
  .venv/bin/python search_agent/gemini_vista_client.py \
    --mode react --react-cm "$CM" \
    --model "$MODEL" \
    --mcp-url "http://127.0.0.1:${PORT}/mcp" \
    --query "$QUERIES" \
    --output-dir "runs/${RUN_ID}_${CM}" \
    --max-context-size "$W" \
    --max-total-tokens "$B" \
    --max-tokens 10000 --max-steps 0 \
    --num-threads "$THREADS" --verbose
  .venv/bin/python scripts_evaluation/judge_api.py \
    --input_dir "runs/${RUN_ID}_${CM}" \
    --label "${RUN_ID}_${CM}" \
    --out "runs/${RUN_ID}_${CM}/judge_summary.json"
done

echo "REACTCM_DONE RUN_ID=$RUN_ID"
