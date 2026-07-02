#!/usr/bin/env bash
# Budget sweep with per-method memoization from the canonical W=12288,B=163840 run.
# For each target B, copy qids whose canonical trace budget is below B and did not
# hit budget; rerun only qids whose answer could change under the target budget.
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
RUN_ID="${RUN_ID:-wb_budget_memoized_$(date +%Y%m%d_%H%M%S)}"
ARCHIVE_PLACEHOLDER_STYLE="${ARCHIVE_PLACEHOLDER_STYLE:-metadata}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-3600}"
TOOL_TIMEOUT_SECONDS="${TOOL_TIMEOUT_SECONDS:-180}"
TOOL_RETRIES="${TOOL_RETRIES:-2}"
QUERY_RETRIES="${QUERY_RETRIES:-1}"
REF_REACT="${REF_REACT:-runs/wb12288_b163840_subset30_20260625_210817_react}"
REF_VISTA="${REF_VISTA:-runs/wb12288_b163840_subset30_20260625_210817_vista}"

mkdir -p runs

echo "MEMOIZED_BUDGET RUN_ID=$RUN_ID W=$W B=$B MODEL=$MODEL PORT=$PORT"
echo "REF_REACT=$REF_REACT"
echo "REF_VISTA=$REF_VISTA"

.venv/bin/python - "$B" "$QUERIES" "$RUN_ID" "$REF_REACT" "$REF_VISTA" <<'PY'
import json, shutil, sys
from pathlib import Path

B = int(sys.argv[1])
queries_path = Path(sys.argv[2])
run_id = sys.argv[3]
refs = {
    "react": Path(sys.argv[4]),
    "vista": Path(sys.argv[5]),
}

queries = []
for line in queries_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    qid, qtext = line.split("\t", 1)
    queries.append((qid, qtext))

def budget_used(record):
    meta = record.get("metadata") or {}
    usage = record.get("usage") or {}
    for key in ("budget_used", "new_tokens", "total_tokens"):
        val = meta.get(key)
        if isinstance(val, (int, float)):
            return int(val)
    val = usage.get("total_tokens")
    return int(val) if isinstance(val, (int, float)) else 0

for mode, ref_dir in refs.items():
    out_dir = Path("runs") / f"{run_id}_{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_by_qid = {}
    for p in ref_dir.glob("run_*.json"):
        with p.open(encoding="utf-8") as f:
            record = json.load(f)
        qid = str(record.get("query_id"))
        if qid:
            ref_by_qid[qid] = (p, record)

    todo = []
    copied = []
    rerun = []
    for qid, qtext in queries:
        # Existing terminal result in output dir wins for resume.
        exists = False
        for p in out_dir.glob("run_*.json"):
            try:
                old = json.load(p.open(encoding="utf-8"))
            except Exception:
                continue
            if str(old.get("query_id")) == qid and old.get("status") not in {"mcp_transport_error", "query_error"}:
                exists = True
                break
        if exists:
            continue

        if qid not in ref_by_qid:
            todo.append((qid, qtext))
            rerun.append((qid, "missing_reference"))
            continue

        src, record = ref_by_qid[qid]
        used = budget_used(record)
        status = record.get("status")
        if status != "budget_exhausted" and used < B:
            copied_record = json.loads(json.dumps(record))
            copied_record["memoized_from"] = str(src)
            copied_record["memoized_reason"] = f"reference_budget_used={used} < target_B={B} and status={status}"
            meta = copied_record.setdefault("metadata", {})
            if isinstance(meta, dict):
                meta["max_total_tokens"] = B
                meta["output_dir"] = str(out_dir.resolve())
                meta["memoized_from"] = str(src)
            dst = out_dir / src.name
            with dst.open("w", encoding="utf-8") as f:
                json.dump(copied_record, f, indent=2, ensure_ascii=False)
            copied.append((qid, status, used))
        else:
            todo.append((qid, qtext))
            rerun.append((qid, status, used))

    todo_path = Path("runs") / f"{run_id}_{mode}_todo.tsv"
    with todo_path.open("w", encoding="utf-8") as f:
        for qid, qtext in todo:
            f.write(f"{qid}\t{qtext}\n")
    print(f"[{mode}] copied={len(copied)} rerun={len(todo)} todo={todo_path}")
    print(f"[{mode}] rerun_qids={[x[0] for x in rerun]}")
PY

need_mcp=0
for MODE in react vista; do
  TODO="runs/${RUN_ID}_${MODE}_todo.tsv"
  if [[ -s "$TODO" && ! -f "runs/${RUN_ID}_${MODE}/judge_summary.json" ]]; then
    need_mcp=1
  fi
done

MCP_PID=""
cleanup() {
  if [[ -n "${MCP_PID}" ]]; then
    echo "Stopping MCP server $MCP_PID"
    kill "$MCP_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ "$need_mcp" == "1" ]]; then
  if lsof -iTCP:"$PORT" -sTCP:LISTEN -n -P >/dev/null 2>&1; then
    echo "Port $PORT already in use; pick another PORT." >&2
    exit 1
  fi
  .venv/bin/python searcher/mcp_server_api_dense.py \
    --index-path 'indexes/qwen3-embedding-8b/corpus.shard*.pkl' \
    --port "$PORT" --k 5 --snippet-max-tokens 512 \
    > "runs/${RUN_ID}_mcp.log" 2>&1 &
  MCP_PID=$!
  echo "MCP_PID=$MCP_PID"
  for _ in {1..60}; do
    lsof -iTCP:"$PORT" -sTCP:LISTEN -n -P >/dev/null 2>&1 && { echo "MCP listening on $PORT"; break; }
    sleep 2
  done
  lsof -iTCP:"$PORT" -sTCP:LISTEN -n -P >/dev/null 2>&1 || { echo "MCP failed to listen" >&2; exit 1; }
fi

for MODE in react vista; do
  OUT="runs/${RUN_ID}_${MODE}"
  TODO="runs/${RUN_ID}_${MODE}_todo.tsv"
  if [[ -f "$OUT/judge_summary.json" ]]; then
    echo "[skip] ${RUN_ID}_${MODE} already judged"
    continue
  fi
  if [[ -s "$TODO" ]]; then
    .venv/bin/python search_agent/gemini_vista_client.py \
      --mode "$MODE" \
      --model "$MODEL" \
      --mcp-url "http://127.0.0.1:${PORT}/mcp" \
      --query "$TODO" \
      --output-dir "$OUT" \
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
  else
    echo "[memoized] ${RUN_ID}_${MODE}: no qids need rerun"
  fi
  .venv/bin/python scripts_evaluation/judge_api.py \
    --input_dir "$OUT" \
    --label "${RUN_ID}_${MODE}"
done

echo "DONE MEMOIZED_BUDGET RUN_ID=$RUN_ID"
