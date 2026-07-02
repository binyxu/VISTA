export LOCA_OPENAI_BASE_URL="${LOCA_OPENAI_BASE_URL:?Set LOCA_OPENAI_BASE_URL}"
export LOCA_OPENAI_API_KEY="${LOCA_OPENAI_API_KEY:?Set LOCA_OPENAI_API_KEY}"
export LOCA_TASK_TIMEOUT_SECONDS="${LOCA_TASK_TIMEOUT_SECONDS:-1800}"

export SM_STRICT_LONG_CONTEXT=1
export SM_DISABLE_RECOVER=1

BASE_CONFIG="task-configs/final_128k_set_config.json"
STRICT_LC_NO_RECOVER_CONFIG="task-configs/final_128k_set_config_strict_lc_no_recover.generated.json"

python - <<'PY'
import json
from pathlib import Path

base_path = Path("task-configs/final_128k_set_config.json")
out_path = Path("task-configs/final_128k_set_config_strict_lc_no_recover.generated.json")

data = json.loads(base_path.read_text())
out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
print(f"Wrote {out_path} with {len(data.get('configurations', []))} configs")
PY

loca run -c "$STRICT_LC_NO_RECOVER_CONFIG" -s self_managed \
    --model gemini-3-flash \
    --max-context-size "${MAX_CONTEXT_SIZE:-128000}" \
    --max-workers "${MAX_WORKERS:-8}" \
    --max-retries 200
