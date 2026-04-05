FROM vllm/vllm-openai:v0.8.2


ENV VLLM_USE_V1=0
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV VLLM_DISABLE_COMPILE_CACHE=1

# ===============================
# Memory Optimization
# ===============================
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV CUDA_VISIBLE_DEVICES=0

ENV HF_HOME=/tmp/huggingface
ENV VLLM_CONFIG_ROOT=/tmp/vllm
ENV TRITON_CACHE_DIR=/tmp/triton
ENV NUMBA_CACHE_DIR=/tmp/numba
ENV TORCH_HOME=/tmp/torch


RUN pip install --no-cache-dir \
    transformers==4.49.0 \
    huggingface_hub

# ===============================
# DOWNLOAD MODEL 
# ===============================
RUN python3 - <<EOF
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ibm-granite/granite-guardian-3.2-5b",
    local_dir="/models/granite-guardian-3.2-5b",
    local_dir_use_symlinks=False
)
print("Model downloaded successfully")
EOF


RUN ls -lh /models/granite-guardian-3.2-5b && \
    du -sh /models/granite-guardian-3.2-5b


ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV HF_HUB_OFFLINE=1

# ===============================
# App
# ===============================
WORKDIR /app
COPY start.sh .
RUN chmod +x /app/start.sh

EXPOSE 8080

ENTRYPOINT []
CMD ["/app/start.sh"]
