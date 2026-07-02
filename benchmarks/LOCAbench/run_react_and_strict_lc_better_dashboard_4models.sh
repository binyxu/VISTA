#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
COMPACT_SCRIPT="${COMPACT_SCRIPT:-./scripts/compact_all_outputs.sh}"

# Skynet-YU openai-compatible gateway (replaces the deprecated pub_api.access service).
# Inside Tencent network: api.example.com (http is fine, avoids DNS/cert hiccups).
# Outside Tencent network: set LOCA_OPENAI_BASE_URL=https://momi.qq.com/v1

# export LOCA_OPENAI_BASE_URL="${LOCA_OPENAI_BASE_URL:-https://api.example.com/v1}"
# export LOCA_OPENAI_API_KEY="${LOCA_OPENAI_API_KEY:?Set LOCA_OPENAI_API_KEY}"

export LOCA_OPENAI_BASE_URL="${LOCA_OPENAI_BASE_URL:?Set LOCA_OPENAI_BASE_URL}"
export LOCA_OPENAI_API_KEY="${LOCA_OPENAI_API_KEY:?Set LOCA_OPENAI_API_KEY}"

export LOCA_TASK_TIMEOUT_SECONDS="${LOCA_TASK_TIMEOUT_SECONDS:-3600}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

MODELS="${MODELS:-glm-5 deepseek-v4-pro}"
RUN_TIMEOUT="${RUN_TIMEOUT:-8h}"
CONFIG_FILE="${CONFIG_FILE:-task-configs/final_128k_set_config.json}"
STRICT_CONFIG="${STRICT_CONFIG:-task-configs/final_128k_set_config_strict_lc_better_dashboard.generated.json}"
MAX_CONTEXT_SIZE="${MAX_CONTEXT_SIZE:-128000}"
# Paper aligns each model to its native max context window (not a shared cap).
# gpt-5.2 native window is 400K; override per model via *_CONTEXT_SIZE env vars.
GPT_CONTEXT_SIZE="${GPT_CONTEXT_SIZE:-260000}"
DEEPSEEK_CONTEXT_SIZE="${DEEPSEEK_CONTEXT_SIZE:-94000}"
GLM_CONTEXT_SIZE="${GLM_CONTEXT_SIZE:-94000}"
STRICT_MAX_WORKERS="${STRICT_MAX_WORKERS:-2}"
MAX_RETRIES="${MAX_RETRIES:-200}"
MAX_TOOL_USES="${MAX_TOOL_USES:-9999}"
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
compact_outputs() {
    if [[ -x "$COMPACT_SCRIPT" ]]; then
        "$COMPACT_SCRIPT" || true
    else
        bash "$COMPACT_SCRIPT" || true
    fi
}

run_with_limit() {
    local label="$1"
    shift
    echo "===== START $label ====="
    set +e
    python - "$RUN_TIMEOUT" "$@" <<'PY'
import os
import signal
import subprocess
import sys


def parse_timeout(value: str) -> float:
    s = value.strip().lower()
    if s.endswith("ms"):
        return max(0.001, float(s[:-2]) / 1000.0)
    multiplier = 1.0
    if s.endswith("s"):
        s = s[:-1]
    elif s.endswith("m"):
        multiplier = 60.0
        s = s[:-1]
    elif s.endswith("h"):
        multiplier = 3600.0
        s = s[:-1]
    return float(s) * multiplier


timeout_seconds = parse_timeout(sys.argv[1])
cmd = sys.argv[2:]
proc = subprocess.Popen(cmd, start_new_session=True)
def terminate_child(exit_code: int) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
    sys.exit(exit_code)


try:
    sys.exit(proc.wait(timeout=timeout_seconds))
except KeyboardInterrupt:
    terminate_child(130)
except subprocess.TimeoutExpired:
    terminate_child(124)
PY
    local status=$?
    set -e
    if [[ $status -eq 130 ]]; then
        echo "===== INTERRUPTED $label ====="
        exit 130
    elif [[ $status -eq 124 ]]; then
        echo "===== TIMEOUT $label after $RUN_TIMEOUT ====="
    elif [[ $status -ne 0 ]]; then
        echo "===== FAILED $label exit=$status ====="
    else
        echo "===== DONE $label ====="
    fi
    compact_outputs
    return 0
}

cd "$ROOT"

# Reduce to state0 only: keep the first configuration per env (seed 42 == state0).
STATE0_CONFIG="${STATE0_CONFIG:-task-configs/final_128k_set_config_state0.generated.json}"
SRC_CONFIG="$CONFIG_FILE" OUT_CONFIG="$STATE0_CONFIG" python - <<'PY'
import json, os
from pathlib import Path
src = Path(os.environ["SRC_CONFIG"]); out = Path(os.environ["OUT_CONFIG"])
data = json.loads(src.read_text())
seen, state0 = set(), []
for cfg in data["configurations"]:
    key = cfg.get("name") or cfg.get("env_class")
    if key in seen:
        continue
    seen.add(key); state0.append(cfg)
data["configurations"] = state0
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
print(f"Wrote {out} with {len(state0)} state0 configs (from {src.name})")
PY
CONFIG_FILE="$STATE0_CONFIG"

cp "$CONFIG_FILE" "$STRICT_CONFIG"

for MODEL in $MODELS; do
    MODEL_CONTEXT_SIZE="$(context_size_for_model "$MODEL")"
    run_with_limit "strict_lc_better_dashboard/$MODEL" \
        env SM_STRICT_LONG_CONTEXT=1 SM_BETTER_DASHBOARD=1 \
        python -m loca.cli.main run \
            --config-file "$STRICT_CONFIG" \
            --strategy self_managed \
            --model "$MODEL" \
            --max-context-size "$MODEL_CONTEXT_SIZE" \
            --max-workers "$STRICT_MAX_WORKERS" \
            --max-tool-uses "$MAX_TOOL_USES" \
            --max-retries "$MAX_RETRIES" \
            --reasoning-effort "$REASONING_EFFORT"
done
