# vLLM TurboQuant — CUDA 12.8 for B200 (SM100) support
# SM121 checks removed for B200+ compatibility
FROM --platform=linux/amd64 nvidia/cuda:12.8.1-devel-ubuntu24.04

# RunPod-compatible: SSH + basic tools
RUN apt-get update && apt-get install -y \
    python3.12 python3.12-dev python3.12-venv python3-pip \
    openssh-server git curl wget && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    mkdir -p /var/run/sshd && \
    echo 'root:root' | chpasswd && \
    sed -i 's/#PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config

# Copy turboquant source (with SM121 patches applied)
COPY . /opt/vllm-turboquant
WORKDIR /opt/vllm-turboquant

# Create venv and install vLLM from turboquant source
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV="/opt/venv"

RUN VLLM_USE_PRECOMPILED=1 pip install -e ".[all]" && \
    pip install regex && \
    pip cache purge

# Include default MiniMax turboquant metadata
COPY minimax_turboquant_kv.json /opt/minimax_turboquant_kv.json

EXPOSE 22 8000

ENV CUDA_DEVICE_ORDER=PCI_BUS_ID \
    SAFETENSORS_FAST_GPU=1 \
    VLLM_NVFP4_GEMM_BACKEND=cutlass \
    VLLM_USE_FLASHINFER_MOE_FP4=0 \
    OMP_NUM_THREADS=8 \
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    HF_HOME=/workspace/huggingface \
    HUGGINGFACE_HUB_CACHE=/workspace/huggingface/hub

ENTRYPOINT ["/bin/bash", "-c", "/usr/sbin/sshd && exec vllm serve \"$@\"", "--"]
CMD ["lukealonso/MiniMax-M2.5-NVFP4", \
     "--download-dir", "/workspace/huggingface/hub", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--served-model-name", "MiniMax-M2.5-NVFP4", \
     "--trust-remote-code", \
     "--tensor-parallel-size", "1", \
     "--kv-cache-dtype", "turboquant25", \
     "--enable-turboquant", \
     "--turboquant-metadata-path", "/opt/minimax_turboquant_kv.json", \
     "--gpu-memory-utilization", "0.95", \
     "--max-model-len", "32768", \
     "--max-num-batched-tokens", "16384", \
     "--max-num-seqs", "64", \
     "--enable-auto-tool-choice", \
     "--tool-call-parser", "minimax_m2"]
