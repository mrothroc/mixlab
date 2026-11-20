# Layer 2: Add GPU architectures incrementally to the base image.
# Ninja reuses existing .o files — only compiles new architecture kernels.
#
# Usage: set ARCHS build arg to the FULL list including previous ones.
#
# Examples:
#   docker build -f docker/addarch.Dockerfile --build-arg ARCHS="80;86" \
#       --build-arg BASE_IMAGE=mixlab-cuda-base -t mixlab-cuda .
#
#   docker build -f docker/addarch.Dockerfile --build-arg ARCHS="80;86;89" \
#       --build-arg BASE_IMAGE=mixlab-cuda -t mixlab-cuda .
#
# Chain: base(80) -> addarch(80;86) -> addarch(80;86;89) -> etc.
#
# Memory: each architecture adds ~2GB peak RAM during compilation.
# Use a machine with at least 4GB per concurrent architecture at -j4.

ARG BASE_IMAGE=mixlab-cuda-base
FROM ${BASE_IMAGE}

ARG ARCHS="80;86"
ARG MLX_VERSION=v0.32.4
ARG MLX_COMMIT=36969c094354086e15d2a327a9c50feb20b9cf11

COPY docker/test_mlx_jit_cccl_probe.cpp /tmp/test_mlx_jit_cccl_probe.cpp
RUN set -x \
    && test "${MIXLAB_MLX_BUILD_VERSION}" = "${MLX_VERSION}" \
    && test "${MIXLAB_MLX_CUDA_WORKER_FIX}" = "1" \
    && test "${MIXLAB_MLX_CONV_TRANSPOSE_GRAD_FIX}" = "1" \
    && test "${MIXLAB_MLX_CUDA_ALLOCATOR_FIX}" = "2" \
    && test "${MIXLAB_MLX_CUDA_CONV_PLAN_FIX}" = "1" \
    && test "${MIXLAB_MLX_CUDA_JIT_CCCL_FIX}" = "1" \
    && test "${MIXLAB_MLX_BUILD_COMMIT}" = "${MLX_COMMIT}" \
    && test "$(git -C /opt/mlx rev-parse HEAD)" = "${MLX_COMMIT}" \
    && cd /opt/mlx/build \
    && cmake .. -DMLX_BUILD_CUDA=ON -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_GGUF=OFF \
       -DMLX_CUDA_ARCHITECTURES="${ARCHS}" -DCMAKE_BUILD_TYPE=Release -G Ninja \
    && grep -Eq '^NCCL_LIBRARIES:FILEPATH=.*/libnccl' CMakeCache.txt \
    && ninja -j4 \
    && ninja install \
    && mkdir -p /tmp/mlx-jit-cccl && cd /tmp/mlx-jit-cccl \
    && ar x /usr/local/lib/libmlx.a dirs.cpp.o \
    && g++ -std=c++17 /tmp/test_mlx_jit_cccl_probe.cpp dirs.cpp.o -I/usr/local/cuda/include \
       -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 -lnvrtc -o probe \
    && ./probe /usr/local/include/cccl /usr/local/cuda/include

WORKDIR /app
