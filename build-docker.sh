#!/usr/bin/env bash
set -euo pipefail

# Build vLLM TurboQuant Docker image from source
# Targeting Blackwell (compute capability 12.0)

IMAGE_NAME="${IMAGE_NAME:-vllm-turboquant}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
MAX_JOBS="${MAX_JOBS:-$(nproc)}"
NVCC_THREADS="${NVCC_THREADS:-8}"

# Blackwell = 12.0; include 9.0 (Hopper) and 10.0 (Ada) for flexibility
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0 10.0 12.0}"

echo "Building ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  CUDA arch list: ${TORCH_CUDA_ARCH_LIST}"
echo "  MAX_JOBS: ${MAX_JOBS}"
echo "  NVCC_THREADS: ${NVCC_THREADS}"

DOCKER_BUILDKIT=1 docker build \
    -f docker/Dockerfile \
    --target vllm-openai \
    --build-arg max_jobs="${MAX_JOBS}" \
    --build-arg nvcc_threads="${NVCC_THREADS}" \
    --build-arg torch_cuda_arch_list="${TORCH_CUDA_ARCH_LIST}" \
    --build-arg GIT_REPO_CHECK=0 \
    --build-arg RUN_WHEEL_CHECK=false \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    .

echo ""
echo "Build complete: ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "Run with:"
echo "  docker run --gpus all -p 8000:8000 ${IMAGE_NAME}:${IMAGE_TAG} \\"
echo "    --model <model-name> \\"
echo "    --quantization turboquant"
