# vLLM TurboQuant — source install at boot on GPU node
FROM --platform=linux/amd64 nvidia/cuda:12.8.1-devel-ubuntu24.04

RUN apt-get update && apt-get install -y \
    python3.12 python3.12-dev python3.12-venv python3-pip \
    git openssh-server curl && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    mkdir -p /var/run/sshd && \
    echo 'root:root' | chpasswd && \
    sed -i 's/#PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy turboquant source
COPY . /opt/vllm-turboquant
WORKDIR /opt/vllm-turboquant

# Pre-install to bake into image (uses precompiled wheels, no CUDA compile)
RUN uv venv --python 3.12 /opt/venv && \
    . /opt/venv/bin/activate && \
    VLLM_USE_PRECOMPILED=1 uv pip install -e ".[all]"

ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

EXPOSE 22 8000

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
