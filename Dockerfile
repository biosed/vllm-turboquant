# vLLM TurboQuant — overlay on official vLLM image
# All turboquant changes are Python/Triton, no CUDA compilation needed
FROM --platform=linux/amd64 vllm/vllm-openai:latest

# SSH server for RunPod / remote access
RUN apt-get update && \
    apt-get install -y openssh-server && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /var/run/sshd && \
    echo 'root:root' | chpasswd && \
    sed -i 's/#PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config

ARG SITE=/usr/local/lib/python3.12/dist-packages/vllm

# Patch config + CLI (adds turboquant25/35 dtype and --enable-turboquant flag)
COPY vllm/config/cache.py ${SITE}/config/cache.py
COPY vllm/engine/arg_utils.py ${SITE}/engine/arg_utils.py

# Patch attention backends and ops
COPY vllm/v1/attention/backends/triton_attn.py ${SITE}/v1/attention/backends/triton_attn.py
COPY vllm/v1/attention/selector.py ${SITE}/v1/attention/selector.py
COPY vllm/v1/attention/ops/triton_prefill_attention.py ${SITE}/v1/attention/ops/triton_prefill_attention.py
COPY vllm/v1/attention/ops/triton_turboquant_decode.py ${SITE}/v1/attention/ops/triton_turboquant_decode.py
COPY vllm/v1/attention/ops/triton_turboquant_kv_update.py ${SITE}/v1/attention/ops/triton_turboquant_kv_update.py
COPY vllm/v1/attention/ops/turboquant_kv_cache.py ${SITE}/v1/attention/ops/turboquant_kv_cache.py
COPY vllm/v1/attention/ops/turboquant_metadata.py ${SITE}/v1/attention/ops/turboquant_metadata.py

# Patch worker, kv cache interface, platform, model executor
COPY vllm/v1/worker/gpu_worker.py ${SITE}/v1/worker/gpu_worker.py
COPY vllm/v1/worker/utils.py ${SITE}/v1/worker/utils.py
COPY vllm/v1/kv_cache_interface.py ${SITE}/v1/kv_cache_interface.py
COPY vllm/platforms/cuda.py ${SITE}/platforms/cuda.py
COPY vllm/utils/torch_utils.py ${SITE}/utils/torch_utils.py
COPY vllm/model_executor/layers/attention/attention.py ${SITE}/model_executor/layers/attention/attention.py

# Include benchmark tooling
COPY benchmarks/generate_turboquant_metadata.py /vllm-workspace/benchmarks/

EXPOSE 22 8000

# Start SSH on boot, then exec vllm serve
ENTRYPOINT ["/bin/bash", "-c", "/usr/sbin/sshd && exec vllm serve \"$@\"", "--"]
CMD ["lukealonso/MiniMax-M2.5-NVFP4", \
     "--download-dir", "/workspace/huggingface/hub", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--served-model-name", "MiniMax-M2.5-NVFP4", \
     "--trust-remote-code", \
     "--tensor-parallel-size", "1", \
     "--kv-cache-dtype", "turboquant25", \
     "--enable-turboquant", \
     "--gpu-memory-utilization", "0.95", \
     "--max-model-len", "190000", \
     "--max-num-batched-tokens", "16384", \
     "--max-num-seqs", "64", \
     "--tool-call-parser", "minimax_m2", \
     "--reasoning-parser", "minimax_m2_append_think"]
