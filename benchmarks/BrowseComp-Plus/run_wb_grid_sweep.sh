#!/usr/bin/env bash
# BrowseComp-Plus W/B sensitivity sweep.
# Keeps the original subset30 setting fixed except for either context window W
# or total trajectory budget B. Existing judged points are skipped.
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

MODEL="${MODEL:-deepseek-v4-pro-qcloud}"
QUERIES="${QUERIES:-topics-qrels/queries_subset30.tsv}"
THREADS="${THREADS:-5}"
BASE_PORT="${BASE_PORT:-8090}"
ARCHIVE_PLACEHOLDER_STYLE="${ARCHIVE_PLACEHOLDER_STYLE:-metadata}"
RUN_PREFIX="${RUN_PREFIX:-wb_grid_subset30_$(date +%Y%m%d_%H%M%S)}"
SUMMARY="runs/${RUN_PREFIX}_summary.csv"

mkdir -p runs

write_summary_header() {
  if [[ ! -f "$SUMMARY" ]]; then
    echo "axis,W,B,mode,n,accuracy,completion,avg_total_tokens,archive,recover,output_dir" > "$SUMMARY"
  fi
}

append_summary() {
  local axis="$1" w="$2" b="$3" mode="$4" out="$5"
  .venv/bin/python - "$axis" "$w" "$b" "$mode" "$out" "$SUMMARY" <<'PY'
import csv, json, sys
axis, w, b, mode, out, summary = sys.argv[1:7]
p = f"{out}/judge_summary.json"
with open(p, encoding="utf-8") as f:
    d = json.load(f)
row = {
    "axis": axis,
    "W": w,
    "B": b,
    "mode": mode,
    "n": d.get("n", 0),
    "accuracy": d.get("accuracy", 0),
    "completion": d.get("completion", 0),
    "avg_total_tokens": d.get("avg_total_tokens", 0),
    "archive": d.get("archive", 0),
    "recover": d.get("recover", 0),
    "output_dir": out,
}
with open(summary, "a", newline="", encoding="utf-8") as f:
    csv.DictWriter(f, fieldnames=list(row)).writerow(row)
PY
}

wait_for_port() {
  local port="$1"
  for _ in {1..60}; do
    curl -fsS "http://127.0.0.1:${port}/mcp" >/dev/null 2>&1 && return 0
    sleep 2
  done
  return 1
}

run_point() {
  local axis="$1" w="$2" b="$3" port="$4"
  local run_id="${RUN_PREFIX}_${axis}_W${w}_B${b}"
  local mcp_log="runs/${run_id}_mcp.log"

  echo "========== POINT axis=${axis} W=${w} B=${b} port=${port} =========="

  if [[ -f "runs/${run_id}_react/judge_summary.json" && -f "runs/${run_id}_vista/judge_summary.json" ]]; then
    echo "[skip] already judged ${run_id}"
    append_summary "$axis" "$w" "$b" "react" "runs/${run_id}_react"
    append_summary "$axis" "$w" "$b" "vista" "runs/${run_id}_vista"
    return 0
  fi

  echo "[mcp] starting dense server on ${port}, log=${mcp_log}"
  .venv/bin/python searcher/mcp_server_api_dense.py \
    --index-path 'indexes/qwen3-embedding-8b/corpus.shard*.pkl' \
    --port "$port" --k 5 --snippet-max-tokens 512 \
    > "$mcp_log" 2>&1 &
  local mcp_pid=$!
  echo "MCP_PID=${mcp_pid} RUN_ID=${run_id}"
  trap 'kill "$mcp_pid" 2>/dev/null || true' RETURN

  wait_for_port "$port" || {
    echo "MCP failed to listen on ${port}; last log lines:" >&2
    tail -80 "$mcp_log" >&2 || true
    exit 1
  }
  echo "[mcp] ready on ${port}"

  for mode in react vista; do
    local out="runs/${run_id}_${mode}"
    if [[ ! -f "${out}/judge_summary.json" ]]; then
      echo "[run] mode=${mode} W=${w} B=${b} out=${out}"
      .venv/bin/python search_agent/gemini_vista_client.py \
        --mode "$mode" \
        --model "$MODEL" \
        --mcp-url "http://127.0.0.1:${port}/mcp" \
        --query "$QUERIES" \
        --output-dir "$out" \
        --max-context-size "$w" \
        --max-total-tokens "$b" \
        --max-tokens 10000 \
        --max-steps 0 \
        --archive-placeholder-style "$ARCHIVE_PLACEHOLDER_STYLE" \
        --num-threads "$THREADS" \
        --verbose
      .venv/bin/python scripts_evaluation/judge_api.py \
        --input_dir "$out" \
        --label "${run_id}_${mode}"
      echo "[judge] mode=${mode} done"
    fi
    append_summary "$axis" "$w" "$b" "$mode" "$out"
  done

  kill "$mcp_pid" 2>/dev/null || true
  trap - RETURN
}

write_summary_header

# Baseline point exists from the original 40% vs 63.33% run. This script logs
# new points under RUN_PREFIX; keep the old run as the canonical baseline row.
echo "canonical_baseline,12288,163840,react,30,0.4,0.23333333333333334,273055.13333333336,0,0,runs/wb12288_b163840_subset30_20260625_210817_react" >> "$SUMMARY"
echo "canonical_baseline,12288,163840,vista,30,0.6333333333333333,0.5666666666666667,636779.0333333333,544,7,runs/wb12288_b163840_subset30_20260625_210817_vista" >> "$SUMMARY"

port="$BASE_PORT"

# Direction 1: fixed trajectory budget, vary context window.
for w in 6144 8192 16384 32768; do
  run_point "window" "$w" 163840 "$port"
  port=$((port + 1))
done

# Direction 2: fixed context window, vary total trajectory budget.
for b in 40960 81920 327680; do
  run_point "budget" 12288 "$b" "$port"
  port=$((port + 1))
done

echo "DONE summary=${SUMMARY}"
