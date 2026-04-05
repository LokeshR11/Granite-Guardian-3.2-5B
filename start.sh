#!/bin/bash
set -e

echo "=============================================="
echo "   Granite Guardian 3.2 (5B)    "
echo "=============================================="

# --------------------------------------------------
# Environment Debug
# --------------------------------------------------
echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
echo "TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"
echo "VLLM_USE_V1=$VLLM_USE_V1"

# --------------------------------------------------
# GPU CHECK
# --------------------------------------------------
echo "===== GPU STATUS ====="
nvidia-smi || echo "⚠️ No GPU detected"

python3 - <<EOF
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2))
EOF

# --------------------------------------------------
# MODEL PATH CHECK
# --------------------------------------------------
MODEL_PATH="/models/granite-guardian-3.2-5b"

echo "===== MODEL CHECK ====="

if [ ! -d "$MODEL_PATH" ]; then
  echo "❌ ERROR: Model not found at $MODEL_PATH"
  exit 1
fi

echo "✅ Model found at $MODEL_PATH"
echo "Model size:"
du -sh $MODEL_PATH || true

echo "===== MODEL FILES (TOP 20) ====="
ls -lh $MODEL_PATH | head -20

# --------------------------------------------------
# START VLLM SERVER
# --------------------------------------------------
echo "===== STARTING VLLM SERVER ====="

exec python3 -u -m vllm.entrypoints.openai.api_server \
  --model $MODEL_PATH \
  --host 0.0.0.0 \
  --port 8080 \
  --dtype float16 \
  --gpu-memory-utilization 0.70 \
  --max-model-len 1024 \
  --max-num-seqs 2 \
  --tensor-parallel-size 1 \
  --task classify \
  --trust-remote-code \
  --enforce-eager
