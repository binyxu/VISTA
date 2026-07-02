#!/usr/bin/env bash
set -euo pipefail

export LOCA_OPENAI_BASE_URL="${LOCA_OPENAI_BASE_URL:-https://api.example.com/v1}"
export LOCA_OPENAI_API_KEY="${LOCA_OPENAI_API_KEY:?Set LOCA_OPENAI_API_KEY}"

CONFIG_FILE="${CONFIG_FILE:-task-configs/final_128k_set_config.json}"
MODELS="${MODELS:-gpt-5.2 glm-5 deepseek-v4-pro claude-sonnet-4-6}"
MAX_CONTEXT_SIZE="${MAX_CONTEXT_SIZE:-128000}"
# Paper aligns each model to its native max context window (not a shared cap).
# gpt-5.2 native window is 400K; override per model via *_CONTEXT_SIZE env vars.
GPT_CONTEXT_SIZE="${GPT_CONTEXT_SIZE:-260000}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
MAX_WORKERS="${MAX_WORKERS:-5}"
MAX_TOOL_USES="${MAX_TOOL_USES:-9999}"
MAX_RETRIES="${MAX_RETRIES:-50}"
# Align with paper's GPT-5.2-Medium; Venus honors top-level reasoning_effort.
REASONING_EFFORT="${REASONING_EFFORT:-medium}"

context_size_for_model() {
    case "$1" in
        gpt-5*) echo "$GPT_CONTEXT_SIZE" ;;
        glm-*) echo "$GLM_CONTEXT_SIZE" ;;
        deepseek-*) echo "$DEEPSEEK_CONTEXT_SIZE" ;;
        claude-*) echo "$CLAUDE_CONTEXT_SIZE" ;;
        *) echo "$MAX_CONTEXT_SIZE" ;;
    esac
}

for MODEL in $MODELS; do
    echo "Running ReAct baseline with model: $MODEL"
    MODEL_CONTEXT_SIZE="$(context_size_for_model "$MODEL")"
    python -m loca.cli.main run \
        --config-file "$CONFIG_FILE" \
        --strategy react \
        --model "$MODEL" \
        --max-context-size "$MODEL_CONTEXT_SIZE" \
        --max-tokens "$MAX_TOKENS" \
        --max-workers "$MAX_WORKERS" \
        --max-tool-uses "$MAX_TOOL_USES" \
        --max-retries "$MAX_RETRIES" \
        --reasoning-effort "$REASONING_EFFORT"
done
