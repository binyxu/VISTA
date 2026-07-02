export LOCA_OPENAI_BASE_URL="${LOCA_OPENAI_BASE_URL:?Set LOCA_OPENAI_BASE_URL}"
export LOCA_OPENAI_API_KEY="${LOCA_OPENAI_API_KEY:?Set LOCA_OPENAI_API_KEY}"
export LOCA_TASK_TIMEOUT_SECONDS="${LOCA_TASK_TIMEOUT_SECONDS:-1800}"

export SM_STRICT_LONG_CONTEXT=1
export SM_BETTER_DASHBOARD=1

CONTEXT_K="${CONTEXT_K:-128}"
MAX_CONTEXT_SIZE="${MAX_CONTEXT_SIZE:-128000}"
BASE_CONFIG="task-configs/final_${CONTEXT_K}k_set_config.json"
STRICT_LC_BETTER_DASHBOARD_CONFIG="task-configs/final_${CONTEXT_K}k_set_config_strict_lc_better_dashboard.generated.json"

BASE_CONFIG="$BASE_CONFIG" STRICT_LC_BETTER_DASHBOARD_CONFIG="$STRICT_LC_BETTER_DASHBOARD_CONFIG" python - <<'PY'
import json
import os
from pathlib import Path

base_path = Path(os.environ["BASE_CONFIG"])
out_path = Path(os.environ["STRICT_LC_BETTER_DASHBOARD_CONFIG"])

data = json.loads(base_path.read_text())
out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
print(f"Wrote {out_path} with {len(data.get('configurations', []))} configs")
PY

loca run -c "$STRICT_LC_BETTER_DASHBOARD_CONFIG" -s self_managed \
    --model gemini-3-flash \
    --max-context-size "$MAX_CONTEXT_SIZE" \
    --max-workers "${MAX_WORKERS:-8}" \
    --max-retries 200
