#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LOCA_DIR="${LOCA_DIR:-../LOCA-bench}"
if [ -f "$LOCA_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$LOCA_DIR/.env"
  set +a
fi

export LOCA_OPENAI_BASE_URL="${LOCA_OPENAI_BASE_URL:-https://api.example.com/v1}"

LLM_SERVER="${LLM_SERVER:-api}"
LLM_CONFIG="${LLM_CONFIG:-configs/loca_proxy_gpt52.yaml}"
JUDGE_SERVER="${JUDGE_SERVER:-api}"
JUDGE_CONFIG="${JUDGE_CONFIG:-configs/loca_proxy_gpt52.yaml}"
SUBSET="${SUBSET:-openend}"
METHOD="${METHOD:-self_managed}"
METHOD_CONFIG="${METHOD_CONFIG:-configs/method_configs/self_managed.yaml}"
TEST_DIR="${TEST_DIR:-dataset/test}"
OUTPUT_DIR="${OUTPUT_DIR:-../../results/ama_bench_loca_proxy_self_managed}"
SAMPLES="${SAMPLES:-1}"
EVALUATE="${EVALUATE:-False}"
MAX_CONCURRENCY_EPISODES="${MAX_CONCURRENCY_EPISODES:-1}"
MAX_CONCURRENCY_QUESTIONS_PER_EPISODE="${MAX_CONCURRENCY_QUESTIONS_PER_EPISODE:-1}"
JUDGE_MAX_CONCURRENCY="${JUDGE_MAX_CONCURRENCY:-1}"

ARGS=(
  --llm-server "$LLM_SERVER"
  --llm-config "$LLM_CONFIG"
  --judge-server "$JUDGE_SERVER"
  --judge-config "$JUDGE_CONFIG"
  --subset "$SUBSET"
  --method "$METHOD"
  --method-config "$METHOD_CONFIG"
  --test-dir "$TEST_DIR"
  --output-dir "$OUTPUT_DIR"
  --samples "$SAMPLES"
  --evaluate "$EVALUATE"
  --max-concurrency-episodes "$MAX_CONCURRENCY_EPISODES"
  --max-concurrency-questions-per-episode "$MAX_CONCURRENCY_QUESTIONS_PER_EPISODE"
  --judge-max-concurrency "$JUDGE_MAX_CONCURRENCY"
)

python src/run.py "${ARGS[@]}"
