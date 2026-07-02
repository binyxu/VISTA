#!/bin/bash
set -e

CONFIG="${1:-configs/qwen3-32B.yaml}"
[ ! -f "$CONFIG" ] && echo "Config not found: $CONFIG" && exit 1

VLLM_LOG="${VLLM_LOG:-logs/vllm_server.log}"
mkdir -p "$(dirname "$VLLM_LOG")"

# Parse all config at once
read -r MODEL VLLM_HOST VLLM_PORT GPUS MAX_LEN TP_SIZE GPU_MEM_UTIL < <(python -c "
import yaml
c = yaml.safe_load(open('$CONFIG'))
vl = c.get('vllm_launch', {})
print(c['model'], c.get('vllm_host', 'localhost'), c.get('vllm_port', 8000),
      vl.get('gpus', '0'), vl.get('max_model_len', 16384), vl.get('tensor_parallel_size', 1),
      vl.get('gpu_memory_utilization', 0.9))
")

# Check if already running
curl -sf "http://${VLLM_HOST}:${VLLM_PORT}/health" >/dev/null && \
    echo "vLLM already running at http://${VLLM_HOST}:${VLLM_PORT}" && exit 0

# Launch
echo "Launching vLLM: $MODEL on GPU $GPUS at http://${VLLM_HOST}:${VLLM_PORT}"
echo "vLLM log: $VLLM_LOG"
CUDA_VISIBLE_DEVICES=$GPUS \
VLLM_USE_DEEP_GEMM=0 \
VLLM_DEEP_GEMM_WARMUP=skip \
nohup python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --host "$VLLM_HOST" --port "$VLLM_PORT" \
    --max-model-len "$MAX_LEN" --tensor-parallel-size "$TP_SIZE" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    > "$VLLM_LOG" 2>&1 &

echo "PID: $! | Waiting for startup..."
for i in {1..120}; do
    curl -sf "http://${VLLM_HOST}:${VLLM_PORT}/health" >/dev/null && echo "Ready!" && exit 0
    sleep 2
done

echo "Timeout. Check $VLLM_LOG" && exit 1
