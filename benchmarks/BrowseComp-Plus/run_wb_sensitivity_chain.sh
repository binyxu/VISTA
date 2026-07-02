#!/usr/bin/env bash
# Minimal chain runner: each point is exactly one invocation of the existing
# run_wb_sweep.sh with only environment variables changed.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BROWSECOMP_ROOT="${BROWSECOMP_ROOT:-${SCRIPT_DIR}}"
cd "$BROWSECOMP_ROOT"

RUN_PREFIX="${RUN_PREFIX:-wb_sensitivity_chain_$(date +%Y%m%d_%H%M%S)}"
MODEL="${MODEL:-deepseek-v4-pro-qcloud}"
THREADS="${THREADS:-5}"
QUERIES="${QUERIES:-topics-qrels/queries_subset30.tsv}"
BASE_PORT="${BASE_PORT:-8090}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-3600}"
TOOL_TIMEOUT_SECONDS="${TOOL_TIMEOUT_SECONDS:-180}"
TOOL_RETRIES="${TOOL_RETRIES:-2}"
QUERY_RETRIES="${QUERY_RETRIES:-1}"
SUMMARY="runs/${RUN_PREFIX}_summary.csv"

mkdir -p runs

if [[ ! -f "$SUMMARY" ]]; then
  echo "axis,W,B,mode,n,accuracy,completion,avg_total_tokens,archive,recover,output_dir" > "$SUMMARY"
  echo "canonical_baseline,12288,163840,react,30,0.4,0.23333333333333334,273055.13333333336,0,0,runs/wb12288_b163840_subset30_20260625_210817_react" >> "$SUMMARY"
  echo "canonical_baseline,12288,163840,vista,30,0.6333333333333333,0.5666666666666667,636779.0333333333,544,7,runs/wb12288_b163840_subset30_20260625_210817_vista" >> "$SUMMARY"
fi

append_summary() {
  local axis="$1" w="$2" b="$3" mode="$4" out="$5"
  .venv/bin/python - "$axis" "$w" "$b" "$mode" "$out" "$SUMMARY" <<'PY'
import csv, json, sys
axis, w, b, mode, out, summary = sys.argv[1:7]
with open(f"{out}/judge_summary.json", encoding="utf-8") as f:
    d = json.load(f)
row = {
    "axis": axis, "W": w, "B": b, "mode": mode,
    "n": d.get("n", 0), "accuracy": d.get("accuracy", 0),
    "completion": d.get("completion", 0),
    "avg_total_tokens": d.get("avg_total_tokens", 0),
    "archive": d.get("archive", 0), "recover": d.get("recover", 0),
    "output_dir": out,
}
try:
    with open(summary, newline="", encoding="utf-8") as f:
        for old in csv.DictReader(f):
            if old.get("output_dir") == out and old.get("mode") == mode:
                sys.exit(0)
except FileNotFoundError:
    pass
with open(summary, "a", newline="", encoding="utf-8") as f:
    csv.DictWriter(f, fieldnames=list(row)).writerow(row)
PY
}

run_point() {
  local axis="$1" w="$2" b="$3" port="$4"
  local run_id="${RUN_PREFIX}_${axis}_W${w}_B${b}"
  local log="runs/${run_id}.driver.log"
  echo "========== ${run_id} =========="
  echo "log=${log}"

  if [[ -f "runs/${run_id}_react/judge_summary.json" && -f "runs/${run_id}_vista/judge_summary.json" ]]; then
    echo "[skip] already judged ${run_id}"
  else
    env W="$w" B="$b" MODEL="$MODEL" PORT="$port" QUERIES="$QUERIES" THREADS="$THREADS" RUN_ID="$run_id" TIMEOUT_SECONDS="$TIMEOUT_SECONDS" TOOL_TIMEOUT_SECONDS="$TOOL_TIMEOUT_SECONDS" TOOL_RETRIES="$TOOL_RETRIES" QUERY_RETRIES="$QUERY_RETRIES" bash run_wb_sweep.sh > "$log" 2>&1
    rc=$?
    echo "[point exit] ${run_id} rc=${rc}"
    if [[ "$rc" != "0" ]]; then
      echo "[error] ${run_id} failed; last log lines:"
      tail -120 "$log" || true
      return "$rc"
    fi
  fi

  append_summary "$axis" "$w" "$b" "react" "runs/${run_id}_react"
  append_summary "$axis" "$w" "$b" "vista" "runs/${run_id}_vista"
}

port="$BASE_PORT"

run_point window 6144 163840 "$port"; port=$((port + 1))
run_point window 8192 163840 "$port"; port=$((port + 1))
run_point window 16384 163840 "$port"; port=$((port + 1))
run_point window 32768 163840 "$port"; port=$((port + 1))

run_point budget 12288 40960 "$port"; port=$((port + 1))
run_point budget 12288 81920 "$port"; port=$((port + 1))
run_point budget 12288 327680 "$port"; port=$((port + 1))

echo "DONE summary=${SUMMARY}"
