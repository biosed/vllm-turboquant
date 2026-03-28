# vLLM TurboQuant — overlay on official vLLM image
# All turboquant changes are Python/Triton, no CUDA compilation needed
FROM vllm/vllm-openai:latest

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

# Startup script: launch SSH + vLLM
COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 22 8000

ENTRYPOINT ["/start.sh"]
