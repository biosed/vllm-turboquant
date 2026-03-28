# MiniMax-M2.5-NVFP4 on B200 — official vLLM 0.15.1+ for SM100 support
FROM vllm/vllm-openai:v0.18.0

# RunPod-compatible: SSH
RUN apt-get update && apt-get install -y openssh-server && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /var/run/sshd && \
    echo 'root:root' | chpasswd && \
    sed -i 's/#PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config

EXPOSE 22 8000

ENV CUDA_DEVICE_ORDER=PCI_BUS_ID \
    SAFETENSORS_FAST_GPU=1 \
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
     "--gpu-memory-utilization", "0.95", \
     "--max-model-len", "196608", \
     "--max-num-batched-tokens", "16384", \
     "--max-num-seqs", "64", \
     "--enable-auto-tool-choice", \
     "--tool-call-parser", "minimax_m2"]
