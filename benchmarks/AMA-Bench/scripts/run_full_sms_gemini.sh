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

RUN_ID="${RUN_ID:-sms_full_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs/${RUN_ID}}"
mkdir -p "$LOG_DIR"

LLM_CONFIG="${LLM_CONFIG:-configs/loca_proxy_gemini3_flash.yaml}"
JUDGE_CONFIG="${JUDGE_CONFIG:-$LLM_CONFIG}"
METHOD_CONFIG="${METHOD_CONFIG:-configs/method_configs/self_managed_agentic.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-../../results/ama_full_sms_gemini}"
TEST_FILE="${TEST_FILE:-dataset/test/open_end_qa_set.jsonl}"

MAX_CONCURRENCY_EPISODES="${MAX_CONCURRENCY_EPISODES:-1}"
MAX_CONCURRENCY_QUESTIONS_PER_EPISODE="${MAX_CONCURRENCY_QUESTIONS_PER_EPISODE:-1}"
JUDGE_MAX_CONCURRENCY="${JUDGE_MAX_CONCURRENCY:-4}"
EVALUATE="${EVALUATE:-True}"

ARGS=(
  --llm-server api
  --llm-config "$LLM_CONFIG"
  --subset openend
  --method self_managed_agentic
  --method-config "$METHOD_CONFIG"
  --test-file "$TEST_FILE"
  --output-dir "$OUTPUT_DIR"
  --max-concurrency-episodes "$MAX_CONCURRENCY_EPISODES"
  --max-concurrency-questions-per-episode "$MAX_CONCURRENCY_QUESTIONS_PER_EPISODE"
  --judge-config "$JUDGE_CONFIG"
  --judge-server api
  --judge-max-concurrency "$JUDGE_MAX_CONCURRENCY"
  --evaluate "$EVALUATE"
)

if [ -n "${SAMPLES:-}" ]; then
  ARGS+=(--samples "$SAMPLES")
fi
if [ -n "${DOMAINS:-}" ]; then
  ARGS+=(--domains "$DOMAINS")
fi
if [ -n "${EPISODE_IDS:-}" ]; then
  ARGS+=(--episode-ids "$EPISODE_IDS")
fi
if [ -n "${SKIP_EXISTING_ANSWERS:-}" ]; then
  ARGS+=(--skip-existing-answers "$SKIP_EXISTING_ANSWERS")
fi

echo "Run ID: $RUN_ID"
echo "Logs: $LOG_DIR"
echo "Output: $OUTPUT_DIR"
echo "Method: self_managed_agentic"
python src/run.py "${ARGS[@]}" 2>&1 | tee "$LOG_DIR/run.log"
