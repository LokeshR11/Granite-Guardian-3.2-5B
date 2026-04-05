#!/bin/bash
set -e

echo "=================================================="
echo "   Granite Guardian 3.2 (5B)"
echo "=================================================="

# -------------------------------
# ENV DEBUG
# -------------------------------
echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
echo "TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"
echo "VLLM_USE_V1=$VLLM_USE_V1"

# -------------------------------
# CREATE WRITABLE DIRS
# -------------------------------
mkdir -p /tmp/huggingface \
         /tmp/vllm \
         /tmp/triton \
         /tmp/numba \
         /tmp/torch \
         /tmp/cache \
         /tmp/outlines \
         /tmp/models \
         /tmp/shm

chmod -R 777 /tmp || true

echo "[INFO] Writable dirs ready"

# -------------------------------
# GPU CHECK
# -------------------------------
echo "===== GPU STATUS ====="
nvidia-smi || echo "No GPU detected"

python3 - <<EOF
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2))
EOF


MODEL_PATH="/tmp/models/granite-guardian-3.2-5b"

echo "===== MODEL CHECK ====="

if [ ! -d "$MODEL_PATH" ]; then
  echo "ERROR: Model not found at $MODEL_PATH"
  exit 1
fi

chmod -R 777 $MODEL_PATH || true

echo "Model found at $MODEL_PATH"
echo "Model size:"
du -sh $MODEL_PATH || true

echo "===== MODEL FILES ====="
ls -lh $MODEL_PATH | head -20

# -------------------------------
# START VLLM SERVER
# -------------------------------
echo "===== STARTING VLLM SERVER ====="

export VLLM_LOGGING_LEVEL=DEBUG

exec python3 -u -m vllm.entrypoints.openai.api_server \
  --model $MODEL_PATH \
  --host 0.0.0.0 \
  --port 8080 \
  --dtype float16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 1024 \
  --max-num-seqs 1 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --enforce-eager \
  --disable-custom-all-reduce \
  --served-model-name granite-guardian-3.2-5b
