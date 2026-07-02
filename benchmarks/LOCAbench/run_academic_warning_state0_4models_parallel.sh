#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
COMPACT_SCRIPT="${COMPACT_SCRIPT:-./scripts/compact_all_outputs.sh}"

export LOCA_OPENAI_BASE_URL="${LOCA_OPENAI_BASE_URL:-https://api.example.com/v1}"
export LOCA_OPENAI_API_KEY="${LOCA_OPENAI_API_KEY:?Set LOCA_OPENAI_API_KEY}"
export LOCA_TASK_TIMEOUT_SECONDS="${LOCA_TASK_TIMEOUT_SECONDS:-1800}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export SM_STRICT_LONG_CONTEXT="${SM_STRICT_LONG_CONTEXT:-1}"
export SM_BETTER_DASHBOARD="${SM_BETTER_DASHBOARD:-1}"

MODELS="${MODELS:-gpt-5.2 glm-5 deepseek-v4-pro claude-sonnet-4-6}"
RUN_TIMEOUT="${RUN_TIMEOUT:-8h}"
BASE_CONFIG="${BASE_CONFIG:-task-configs/final_128k_set_config.json}"
ACADEMIC_CONFIG="${ACADEMIC_CONFIG:-task-configs/academic_warning_state0.generated.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/academic_warning_state0_strict_lc_better_dashboard_$(date +%Y%m%d_%H%M%S)}"
MAX_CONTEXT_SIZE="${MAX_CONTEXT_SIZE:-128000}"
# Paper aligns each model to its native max context window (not a shared cap).
# gpt-5.2 native window is 400K; override per model via *_CONTEXT_SIZE env vars.
GPT_CONTEXT_SIZE="${GPT_CONTEXT_SIZE:-260000}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
MAX_WORKERS="${MAX_WORKERS:-1}"
MAX_RETRIES="${MAX_RETRIES:-200}"
MAX_TOOL_USES="${MAX_TOOL_USES:-9999}"
STRATEGY="${STRATEGY:-self_managed}"
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
    return 0
}

on_interrupt() {
    echo "Interrupted; terminating parallel runs..."
    jobs -pr | xargs -r kill -TERM 2>/dev/null || true
    sleep 2
    jobs -pr | xargs -r kill -KILL 2>/dev/null || true
    exit 130
}
trap on_interrupt INT TERM

cd "$ROOT"
mkdir -p "$(dirname "$ACADEMIC_CONFIG")" "$OUTPUT_ROOT"

BASE_CONFIG="$BASE_CONFIG" ACADEMIC_CONFIG="$ACADEMIC_CONFIG" python - <<'PY'
import json
import os
from pathlib import Path

base_path = Path(os.environ["BASE_CONFIG"])
out_path = Path(os.environ["ACADEMIC_CONFIG"])
data = json.loads(base_path.read_text())
academic = next(
    cfg for cfg in data["configurations"]
    if cfg.get("name") == "AcademicWarningS2LEnv"
    or cfg.get("env_class", "").endswith("AcademicWarningS2LEnv")
)
out = dict(data)
out["configurations"] = [academic]
out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(f"Wrote {out_path} with AcademicWarningS2LEnv state0")
PY

pids=()
labels=()

for MODEL in $MODELS; do
    SAFE_MODEL="${MODEL//\//-}"
    SAFE_MODEL="${SAFE_MODEL//:/-}"
    LABEL="academic_warning_state0/$STRATEGY/$MODEL"
    labels+=("$LABEL")
    MODEL_CONTEXT_SIZE="$(context_size_for_model "$MODEL")"
    run_with_limit "$LABEL" \
        python -m loca.cli.main run \
            --config-file "$ACADEMIC_CONFIG" \
            --strategy "$STRATEGY" \
            --model "$MODEL" \
            --max-context-size "$MODEL_CONTEXT_SIZE" \
            --max-tokens "$MAX_TOKENS" \
            --max-workers "$MAX_WORKERS" \
            --max-tool-uses "$MAX_TOOL_USES" \
            --max-retries "$MAX_RETRIES" \
            --reasoning-effort "$REASONING_EFFORT" \
            --output-dir "$OUTPUT_ROOT/$SAFE_MODEL" &
    pids+=("$!")
done

for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || true
    echo "===== COMPACT after ${labels[$i]} ====="
    compact_outputs
done

echo "All academic warning state0 runs finished. Output root: $ROOT/$OUTPUT_ROOT"
