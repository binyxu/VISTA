#!/usr/bin/env bash
# Sensitivity sweep that reuses the existing run_wb_sweep.sh unchanged for each
# point. The canonical W=12288/B=163840 point is recorded from the prior run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BROWSECOMP_ROOT="${BROWSECOMP_ROOT:-${SCRIPT_DIR}}"
cd "$BROWSECOMP_ROOT"

RUN_PREFIX="${RUN_PREFIX:-wb_sensitivity_subset30_$(date +%Y%m%d_%H%M%S)}"
MODEL="${MODEL:-deepseek-v4-pro-qcloud}"
THREADS="${THREADS:-5}"
BASE_PORT="${BASE_PORT:-8090}"
QUERIES="${QUERIES:-topics-qrels/queries_subset30.tsv}"
SUMMARY="runs/${RUN_PREFIX}_summary.csv"

mkdir -p runs

write_header() {
  if [[ ! -f "$SUMMARY" ]]; then
    echo "axis,W,B,mode,n,accuracy,completion,avg_total_tokens,archive,recover,output_dir" > "$SUMMARY"
    echo "canonical_baseline,12288,163840,react,30,0.4,0.23333333333333334,273055.13333333336,0,0,runs/wb12288_b163840_subset30_20260625_210817_react" >> "$SUMMARY"
    echo "canonical_baseline,12288,163840,vista,30,0.6333333333333333,0.5666666666666667,636779.0333333333,544,7,runs/wb12288_b163840_subset30_20260625_210817_vista" >> "$SUMMARY"
  fi
}

append_one() {
  local axis="$1" w="$2" b="$3" mode="$4" out="$5"
  .venv/bin/python - "$axis" "$w" "$b" "$mode" "$out" "$SUMMARY" <<'PY'
import csv, json, sys
axis, w, b, mode, out, summary = sys.argv[1:7]
with open(f"{out}/judge_summary.json", encoding="utf-8") as f:
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

run_point() {
  local axis="$1" w="$2" b="$3" port="$4"
  local run_id="${RUN_PREFIX}_${axis}_W${w}_B${b}"
  echo "========== ${run_id} =========="

  if [[ -f "runs/${run_id}_react/judge_summary.json" && -f "runs/${run_id}_vista/judge_summary.json" ]]; then
    echo "[skip] already judged ${run_id}"
  else
    W="$w" B="$b" MODEL="$MODEL" PORT="$port" QUERIES="$QUERIES" THREADS="$THREADS" RUN_ID="$run_id" bash run_wb_sweep.sh
  fi

  append_one "$axis" "$w" "$b" "react" "runs/${run_id}_react"
  append_one "$axis" "$w" "$b" "vista" "runs/${run_id}_vista"
}

write_header

port="$BASE_PORT"

# Fixed trajectory budget, vary context window.
for w in 6144 8192 16384 32768; do
  run_point "window" "$w" 163840 "$port"
  port=$((port + 1))
done

# Fixed context window, vary trajectory budget.
for b in 40960 81920 327680; do
  run_point "budget" 12288 "$b" "$port"
  port=$((port + 1))
done

echo "DONE summary=${SUMMARY}"
