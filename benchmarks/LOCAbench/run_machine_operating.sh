#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export LOCA_OPENAI_BASE_URL="${LOCA_OPENAI_BASE_URL:-https://api.example.com/v1}"
export LOCA_OPENAI_API_KEY="${LOCA_OPENAI_API_KEY:?Set LOCA_OPENAI_API_KEY}"

export BASE_CONFIG="${BASE_CONFIG:-task-configs/final_128k_set_config.json}"
export SINGLE_CONFIG="${SINGLE_CONFIG:-task-configs/final_128k_filter_low_selling_products_5states.generated.json}"
STRATEGY="${STRATEGY:-self_managed}"
MODEL="${MODEL:-gemini-3-flash}"
MAX_WORKERS="${MAX_WORKERS:-5}"
MAX_RETRIES="${MAX_RETRIES:-200}"
export LOCA_TASK_TIMEOUT_SECONDS="${FILTER_LOW_SELLING_TIMEOUT_SECONDS:-1800}"

python - <<'PY'
import json
import os
from pathlib import Path

base_path = Path(os.environ.get("BASE_CONFIG", "task-configs/final_128k_set_config.json"))
single_path = Path(os.environ.get("SINGLE_CONFIG", "task-configs/final_128k_filter_low_selling_products_5states.generated.json"))

data = json.loads(base_path.read_text())
configs = data["configurations"]

target = "FilterLowSellingProductsS2LEnv"
selected = []

for cfg in configs:
    name = cfg.get("name", "")
    env_class = cfg.get("env_class", "")
    if name == target or env_class.endswith(f".{target}"):
        selected.append(cfg)

if not selected:
    raise SystemExit(f"Missing {target} in {base_path}")

selected.sort(key=lambda cfg: cfg.get("state_id", cfg.get("state", 0)))
out = dict(data)
out["configurations"] = selected
single_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(f"Wrote {single_path} with {target} ({len(selected)} state configs)")
PY

loca run -c "$SINGLE_CONFIG" -s "$STRATEGY" \
    --model "$MODEL" \
    --max-workers "$MAX_WORKERS" \
    --max-retries "$MAX_RETRIES"
