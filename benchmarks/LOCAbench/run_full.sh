export LOCA_OPENAI_BASE_URL="${LOCA_OPENAI_BASE_URL:?Set LOCA_OPENAI_BASE_URL}"
export LOCA_OPENAI_API_KEY="${LOCA_OPENAI_API_KEY:?Set LOCA_OPENAI_API_KEY}"
export LOCA_TASK_TIMEOUT_SECONDS=1800

BASE_CONFIG="task-configs/final_128k_set_config.json"

loca run -c "$BASE_CONFIG" -s self_managed \
    --model gemini-3-flash \
    --max-workers 8 \
    --max-retries 200
