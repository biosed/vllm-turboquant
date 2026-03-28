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

# Patch in turboquant attention backends and worker changes
COPY vllm/v1/attention/backends/triton_attn.py \
     /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/triton_attn.py
COPY vllm/v1/attention/ops/triton_prefill_attention.py \
     /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_prefill_attention.py
COPY vllm/v1/attention/ops/triton_turboquant_decode.py \
     /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_turboquant_decode.py
COPY vllm/v1/attention/ops/turboquant_metadata.py \
     /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/turboquant_metadata.py
COPY vllm/v1/worker/gpu_worker.py \
     /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py
COPY vllm/v1/worker/utils.py \
     /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/utils.py

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
     "--gpu-memory-utilization", "0.95", \
     "--max-model-len", "190000", \
     "--max-num-batched-tokens", "16384", \
     "--max-num-seqs", "64", \
     "--tool-call-parser", "minimax_m2", \
     "--reasoning-parser", "minimax_m2_append_think"]
