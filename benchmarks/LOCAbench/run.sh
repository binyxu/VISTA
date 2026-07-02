export LOCA_OPENAI_BASE_URL="${LOCA_OPENAI_BASE_URL:?Set LOCA_OPENAI_BASE_URL}"
export LOCA_OPENAI_API_KEY="${LOCA_OPENAI_API_KEY:?Set LOCA_OPENAI_API_KEY}"
export LOCA_TASK_TIMEOUT_SECONDS="${LOCA_TASK_TIMEOUT_SECONDS:-1800}"

BASE_CONFIG="task-configs/final_128k_set_config.json"
STATE0_CONFIG="task-configs/final_128k_state0_config.generated.json"

python - <<'PY'
import json
from pathlib import Path

base_path = Path("task-configs/final_128k_set_config.json")
state0_path = Path("task-configs/final_128k_state0_config.generated.json")

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

loca run -c "$STATE0_CONFIG" -s self_managed \
    --model gemini-3-flash \
    --max-workers 5 \
    --max-retries 200
