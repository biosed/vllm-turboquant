# vLLM TurboQuant — build from source with precompiled C extensions
# SM121 checks removed for B200+ compatibility
FROM --platform=linux/amd64 runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Copy turboquant source (with SM121 patches applied)
COPY . /opt/vllm-turboquant
WORKDIR /opt/vllm-turboquant

# Install vLLM from turboquant source using precompiled C/CUDA extensions
RUN VLLM_USE_PRECOMPILED=1 pip install -e ".[all]" && \
    pip install regex && \
    pip cache purge

# Include default MiniMax turboquant metadata
# Users can override with --turboquant-metadata-path at runtime
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
     "--max-num-seqs", "64"]
