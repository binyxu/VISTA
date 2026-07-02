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

ANSWERS_FILE="${ANSWERS_FILE:?Set ANSWERS_FILE to an answers_*.jsonl file}"
TEST_FILE="${TEST_FILE:-dataset/test/open_end_qa_set.jsonl}"
JUDGE_CONFIG="${JUDGE_CONFIG:-configs/loca_proxy_gemini3_flash_minimal.yaml}"
JUDGE_SERVER="${JUDGE_SERVER:-api}"
MAX_WORKERS="${MAX_WORKERS:-1}"

answers_dir="$(dirname "$ANSWERS_FILE")"
answers_base="$(basename "$ANSWERS_FILE" .jsonl)"
OUTPUT_FILE="${OUTPUT_FILE:-${answers_dir}/results_from_${answers_base}.json}"
USAGE_LOG="${USAGE_LOG:-${answers_dir}/usage_judge_from_${answers_base}.jsonl}"

PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" python src/evaluate.py \
  --answers-file "$ANSWERS_FILE" \
  --test-file "$TEST_FILE" \
  --judge-config "$JUDGE_CONFIG" \
  --judge-server "$JUDGE_SERVER" \
  --output-file "$OUTPUT_FILE" \
  --usage-log "$USAGE_LOG" \
  --max-workers "$MAX_WORKERS"
