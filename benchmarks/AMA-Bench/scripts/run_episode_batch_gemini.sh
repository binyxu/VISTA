#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

if [ -f "../LOCA-bench/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "../LOCA-bench/.env"
  set +a
fi

METHOD="${METHOD:-self_managed_agentic}"
if [ "$METHOD" = "ama_agent" ]; then
  METHOD_CONFIG="${METHOD_CONFIG:-configs/method_configs/ama_agent_noembed.yaml}"
  OUTPUT_DIR="${OUTPUT_DIR:-../../results/ama_episode_batch_ama_agent_gemini}"
else
  METHOD_CONFIG="${METHOD_CONFIG:-configs/method_configs/self_managed_agentic.yaml}"
  OUTPUT_DIR="${OUTPUT_DIR:-../../results/ama_episode_batch_sms_gemini}"
fi

LLM_CONFIG="${LLM_CONFIG:-configs/loca_proxy_gemini3_flash.yaml}"
JUDGE_CONFIG="${JUDGE_CONFIG:-$LLM_CONFIG}"
TEST_FILE="${TEST_FILE:-dataset/test/open_end_qa_set.jsonl}"
RUN_ID="${RUN_ID:-episode_batch_${METHOD}_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs/${RUN_ID}}"
EPISODE_IDS="${EPISODE_IDS:?Set EPISODE_IDS to a comma-separated list, e.g. EPISODE_IDS=89,82,85}"
EVALUATE="${EVALUATE:-False}"
JUDGE_MAX_CONCURRENCY="${JUDGE_MAX_CONCURRENCY:-4}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
: > "$LOG_DIR/failed_ids.txt"
: > "$LOG_DIR/succeeded_ids.txt"

IFS=',' read -r -a IDS <<< "$EPISODE_IDS"

echo "Run ID: $RUN_ID"
echo "Method: $METHOD"
echo "Method config: $METHOD_CONFIG"
echo "Output: $OUTPUT_DIR"
echo "Logs: $LOG_DIR"
echo "Evaluate: $EVALUATE"
echo "Episode count: ${#IDS[@]}"

for raw_id in "${IDS[@]}"; do
  eid="$(echo "$raw_id" | xargs)"
  [ -n "$eid" ] || continue
  echo
  echo "======================================================================"
  echo "Episode $eid"
  echo "======================================================================"
  if python src/run.py \
    --llm-server api \
    --llm-config "$LLM_CONFIG" \
    --subset openend \
    --method "$METHOD" \
    --method-config "$METHOD_CONFIG" \
    --test-file "$TEST_FILE" \
    --episode-ids "$eid" \
    --output-dir "$OUTPUT_DIR" \
    --max-concurrency-episodes 1 \
    --max-concurrency-questions-per-episode 1 \
    --judge-config "$JUDGE_CONFIG" \
    --judge-server api \
    --judge-max-concurrency "$JUDGE_MAX_CONCURRENCY" \
    --evaluate "$EVALUATE" \
    2>&1 | tee "$LOG_DIR/episode_${eid}.log"; then
    echo "$eid" >> "$LOG_DIR/succeeded_ids.txt"
  else
    echo "$eid" >> "$LOG_DIR/failed_ids.txt"
    echo "Episode $eid failed; continuing."
  fi
done

echo
echo "Done."
echo "Succeeded: $(tr '\n' ',' < "$LOG_DIR/succeeded_ids.txt" | sed 's/,$//')"
echo "Failed: $(tr '\n' ',' < "$LOG_DIR/failed_ids.txt" | sed 's/,$//')"
