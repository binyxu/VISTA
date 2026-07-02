#!/usr/bin/env bash
# GAIA official-style baselines for Table 1, mirroring BrowseComp-Plus
# run_all_baselines.sh while keeping the GAIA query template, attachments, and
# quasi-exact scorer. The default schedule is vertical/by-task: for one GAIA
# question, run every condition, refresh cumulative scores, then move to the
# next question. Defaults intentionally skip ReAct and VISTA full because those
# are normally produced by run_gaia_official_compare.sh; override CONDS to
# include them.
set -uo pipefail
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
PORT="${PORT:-8113}"
THREADS="${THREADS:-5}"
MAX_STEPS="${MAX_STEPS:-0}"
MAX_RESULT_CONTEXT_FRACTION="${MAX_RESULT_CONTEXT_FRACTION:-0.80}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-900}"
WALL_TIMEOUT_SECONDS="${WALL_TIMEOUT_SECONDS:-$((TIMEOUT_SECONDS + 30))}"
RUN_ID="${RUN_ID:-gaia_official_allbl33_seed42_$(date +%Y%m%d_%H%M%S)}"
DATA_DIR="${DATA_DIR:-data/official/random33_seed42}"
ATTACHMENTS_DIR="${ATTACHMENTS_DIR:-data/attachments}"
TASK_IDS="${TASK_IDS:-}"
LIMIT="${LIMIT:-0}"
SAMPLE_SIZE="${SAMPLE_SIZE:-33}"
SEED="${SEED:-42}"
CONDS="${CONDS:-clear mask skeleton slim active_compress fold auto_archive no_dashboard no_recovery}"
PARALLEL_CONDS="${PARALLEL_CONDS:-1}"
COND_PARALLELISM="${COND_PARALLELISM:-3}"

cond_args() {
  case "$1" in
    react)        echo "--mode react --react-cm truncate" ;;
    clear)        echo "--mode react --react-cm clear" ;;
    mask)         echo "--mode react --react-cm mask" ;;
    skeleton)     echo "--mode react --react-cm skeleton" ;;
    slim)         echo "--mode react --react-cm summary" ;;
    active_compress) echo "--mode react --react-cm active_compress" ;;
    fold)         echo "--mode react --react-cm fold" ;;
    vista_full)   echo "--mode vista --vista-ablate none" ;;
    no_dashboard) echo "--mode vista --vista-ablate no_dashboard" ;;
    no_recovery)  echo "--mode vista --vista-ablate no_recovery" ;;
    auto_archive) echo "--mode vista --vista-ablate auto_archive" ;;
    *) echo "" ;;
  esac
}

if [[ -z "${OPENAI_BASE_URL}" || -z "${OPENAI_API_KEY}" ]]; then
  echo "Set OPENAI_BASE_URL/OPENAI_API_KEY or LOCA_OPENAI_BASE_URL/LOCA_OPENAI_API_KEY." >&2
  exit 1
fi

mkdir -p runs

BUILD_ARGS=(--input data/gaia_2023_all_validation.jsonl --out-dir "$DATA_DIR" --limit "$LIMIT" --sample-size "$SAMPLE_SIZE" --seed "$SEED")
if [[ -n "$TASK_IDS" ]]; then
  BUILD_ARGS+=(--task-ids "$TASK_IDS")
fi
"$PYTHON" scripts/build_official_dataset.py "${BUILD_ARGS[@]}"

mkdir -p "$ATTACHMENTS_DIR"

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
cleanup() { echo "Stopping MCP $MCP_PID"; kill "$MCP_PID" 2>/dev/null || true; }
trap cleanup EXIT

for _ in {1..60}; do
  lsof -iTCP:"$PORT" -sTCP:LISTEN -n -P >/dev/null 2>&1 && break
  sleep 1
done
lsof -iTCP:"$PORT" -sTCP:LISTEN -n -P >/dev/null 2>&1 || {
  echo "MCP server failed to listen; see runs/${RUN_ID}_mcp.log" >&2
  exit 1
}

echo "RUN_ID=$RUN_ID MODEL=$MODEL W=$W B=$B DATA_DIR=$DATA_DIR THREADS=$THREADS TIMEOUT_SECONDS=$TIMEOUT_SECONDS WALL_TIMEOUT_SECONDS=$WALL_TIMEOUT_SECONDS MAX_STEPS=$MAX_STEPS PARALLEL_CONDS=$PARALLEL_CONDS COND_PARALLELISM=$COND_PARALLELISM CONDS='$CONDS'"
SUMMARY="runs/${RUN_ID}_summary.csv"
PROGRESS="runs/${RUN_ID}_progress.log"
TMP_QUERY_DIR="runs/${RUN_ID}_single_queries"
mkdir -p "$TMP_QUERY_DIR"

refresh_summary() {
  echo "condition,n,accuracy,correct,avg_budget_used,output_dir" > "$SUMMARY"
  for SC in $CONDS; do
    local OUT_DIR="runs/${RUN_ID}_${SC}"
    [[ -f "${OUT_DIR}/official_score.json" ]] || continue
    "$PYTHON" - "$SC" "$OUT_DIR" "$SUMMARY" <<'PY' || true
import csv
import json
import sys
from pathlib import Path

condition, out_dir, summary = sys.argv[1:4]
score_path = Path(out_dir) / "official_score.json"
d = json.loads(score_path.read_text(encoding="utf-8"))
rows = d.get("rows", [])
budgets = [r.get("budget_used") for r in rows if isinstance(r.get("budget_used"), (int, float))]
avg_budget = sum(budgets) / len(budgets) if budgets else 0
correct = sum(1 for r in rows if r.get("correct"))
with open(summary, "a", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow([
        condition,
        d.get("n", 0),
        d.get("accuracy", 0),
        correct,
        round(avg_budget),
        out_dir,
    ])
PY
  done
}

TOTAL_Q=$(wc -l < "${DATA_DIR}/queries.tsv" | tr -d ' ')
QI=0
while IFS=$'\t' read -r QID QTEXT; do
  QI=$((QI + 1))
  SINGLE_QUERY="${TMP_QUERY_DIR}/${QI}.tsv"
  EXPECTED_QUERY="${TMP_QUERY_DIR}/expected_upto_${QI}.tsv"
  printf "%s\t%s\n" "$QID" "$QTEXT" > "$SINGLE_QUERY"
  head -n "$QI" "${DATA_DIR}/queries.tsv" > "$EXPECTED_QUERY"
  echo "========== TASK ${QI}/${TOTAL_Q} qid=${QID} ==========" | tee -a "$PROGRESS"

  PIDS=""
  PENDING_PIDS=""
  PENDING_COUNT=0
  STARTED_CONDS=""
  for C in $CONDS; do
    CA="$(cond_args "$C")"
    if [[ -z "$CA" ]]; then
      echo "WARN: unknown condition '$C'; skipping" >&2
      continue
    fi
    OUT_DIR="runs/${RUN_ID}_${C}"
    echo "########## TASK ${QI}/${TOTAL_Q} qid=${QID} condition=${C} ($CA) ##########" | tee -a "$PROGRESS"
    (
      "$PYTHON" scripts/run_with_timeout.py "$WALL_TIMEOUT_SECONDS" \
      "$PYTHON" "${BROWSECOMP_ROOT}/search_agent/gemini_vista_client.py" \
        --model "$MODEL" \
        --mcp-url "http://127.0.0.1:${PORT}/mcp" \
        --query "$SINGLE_QUERY" \
        --output-dir "$OUT_DIR" \
        --max-context-size "$W" \
        --max-total-tokens "$B" \
        --max-result-context-fraction "$MAX_RESULT_CONTEXT_FRACTION" \
        --max-tokens 10000 \
        --max-steps "$MAX_STEPS" \
        --timeout-seconds "$TIMEOUT_SECONDS" \
        --num-threads "$THREADS" \
        --query-template GAIA_TEMPLATE \
        $CA \
        --verbose
    ) > "runs/${RUN_ID}_${C}.log" 2>&1 &
    PID=$!
    PIDS="$PIDS $PID"
    PENDING_PIDS="$PENDING_PIDS $PID"
    PENDING_COUNT=$((PENDING_COUNT + 1))
    STARTED_CONDS="$STARTED_CONDS $C"
    if [[ "$PARALLEL_CONDS" != "1" ]]; then
      wait "$PID" || echo "WARN: $C qid=$QID non-zero exit"
      "$PYTHON" scripts/score_gaia_official.py \
        --input-dir "$OUT_DIR" \
        --ground-truth "${DATA_DIR}/ground_truth.jsonl" \
        --queries "$EXPECTED_QUERY" \
        --out "${OUT_DIR}/official_score.json" > "${OUT_DIR}/score.log" 2>&1 || echo "WARN: score $C failed"
      PENDING_PIDS=""
      PENDING_COUNT=0
    elif [[ "$PENDING_COUNT" -ge "$COND_PARALLELISM" ]]; then
      for CHILD_PID in $PENDING_PIDS; do
        wait "$CHILD_PID" || echo "WARN: child pid=$CHILD_PID qid=$QID non-zero exit"
      done
      PENDING_PIDS=""
      PENDING_COUNT=0
    fi
  done

  if [[ "$PARALLEL_CONDS" == "1" ]]; then
    for PID in $PENDING_PIDS; do
      wait "$PID" || echo "WARN: child pid=$PID qid=$QID non-zero exit"
    done
    for C in $STARTED_CONDS; do
      OUT_DIR="runs/${RUN_ID}_${C}"
      "$PYTHON" scripts/score_gaia_official.py \
        --input-dir "$OUT_DIR" \
        --ground-truth "${DATA_DIR}/ground_truth.jsonl" \
        --queries "$EXPECTED_QUERY" \
        --out "${OUT_DIR}/official_score.json" > "${OUT_DIR}/score.log" 2>&1 || echo "WARN: score $C failed"
    done
  fi

  refresh_summary
  echo "CUMULATIVE_AFTER_TASK ${QI}/${TOTAL_Q} qid=${QID}" | tee -a "$PROGRESS"
  cat "$SUMMARY" | tee -a "$PROGRESS"
done < "${DATA_DIR}/queries.tsv"

echo "ALLBL_DONE RUN_ID=$RUN_ID SUMMARY=$SUMMARY"
