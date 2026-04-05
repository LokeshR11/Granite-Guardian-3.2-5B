FROM vllm/vllm-openai:v0.8.2

# ── Engine fix (Tesla T4 sm_75 compatibility) ─────────────────────────────────
ENV VLLM_USE_V1=0
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV VLLM_DISABLE_COMPILE_CACHE=1

# ── Memory optimization ───────────────────────────────────────────────────────
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV CUDA_VISIBLE_DEVICES=0

# ── Writable paths (KServe runs as non-root) ──────────────────────────────────
ENV HF_HOME=/tmp/huggingface
ENV VLLM_CONFIG_ROOT=/tmp/vllm
ENV TRITON_CACHE_DIR=/tmp/triton
ENV NUMBA_CACHE_DIR=/tmp/numba
ENV TORCH_HOME=/tmp/torch
ENV XDG_CACHE_HOME=/tmp/cache
ENV VLLM_CACHE_ROOT=/tmp/vllm
ENV OUTLINES_CACHE_DIR=/tmp/outlines

# ── Dependencies (pinned) ─────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    transformers==4.49.0 \
    huggingface_hub

# ── Download model  ─────────────────
RUN python3 - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="ibm-granite/granite-guardian-3.2-5b",
    local_dir="/models/granite-guardian-3.2-5b",
    local_dir_use_symlinks=False
)
print("Model downloaded successfully")
EOF

# ── Verify download ───────────────────────────────────────────────────────────
RUN ls -lh /models/granite-guardian-3.2-5b && \
    du -sh /models/granite-guardian-3.2-5b


ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV HF_HUB_OFFLINE=1

# ── App ───────────────────────────────────────────────────────────────────────
WORKDIR /app
COPY start.sh .
RUN chmod +x /app/start.sh

EXPOSE 8080

# ── Override vllm base image default entrypoint ───────────────────────────────
ENTRYPOINT []
CMD ["/app/start.sh"]
