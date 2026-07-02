#!/usr/bin/env bash
set -euo pipefail

export LOCA_OPENAI_BASE_URL="${LOCA_OPENAI_BASE_URL:-https://api.example.com/v1}"
export LOCA_OPENAI_API_KEY="${LOCA_OPENAI_API_KEY:?Set LOCA_OPENAI_API_KEY}"

BASE_CONFIG="${BASE_CONFIG:-task-configs/final_128k_set_config.json}"
STATE0_CONFIG="${STATE0_CONFIG:-task-configs/final_128k_state0_config.generated.json}"
MODELS="${MODELS:-gpt-5.2 glm-5 deepseek-v4-pro claude-sonnet-4-6}"
MAX_CONTEXT_SIZE="${MAX_CONTEXT_SIZE:-128000}"
# Paper aligns each model to its native max context window (not a shared cap).
# gpt-5.2 native window is 400K; override per model via *_CONTEXT_SIZE env vars.
GPT_CONTEXT_SIZE="${GPT_CONTEXT_SIZE:-260000}"
MAX_WORKERS="${MAX_WORKERS:-5}"
MAX_RETRIES="${MAX_RETRIES:-200}"
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

python - <<'PY'
import json
import os
from pathlib import Path

base_path = Path(os.environ.get("BASE_CONFIG", "task-configs/final_128k_set_config.json"))
state0_path = Path(os.environ.get("STATE0_CONFIG", "task-configs/final_128k_state0_config.generated.json"))

data = json.loads(base_path.read_text())
configs = data["configurations"]

seen = set()
state0_configs = []
for cfg in configs:
    key = cfg.get("name") or cfg.get("env_class")
    if key in seen:
        continue
    seen.add(key)
    state0_configs.append(cfg)

out = dict(data)
out["configurations"] = state0_configs
state0_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(f"Wrote {state0_path} with {len(state0_configs)} state0 configs")
PY

for MODEL in $MODELS; do
    echo "Running state0 ReAct smoke with model: $MODEL"
    MODEL_CONTEXT_SIZE="$(context_size_for_model "$MODEL")"
    python -m loca.cli.main run \
        -c "$STATE0_CONFIG" \
        -s react \
        --model "$MODEL" \
        --max-context-size "$MODEL_CONTEXT_SIZE" \
        --max-workers "$MAX_WORKERS" \
        --max-retries "$MAX_RETRIES" \
        --reasoning-effort "$REASONING_EFFORT"
done
