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
ARG MLX_VERSION=v0.32.0
ARG MLX_COMMIT=7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247

RUN test "${MIXLAB_MLX_BUILD_VERSION}" = "${MLX_VERSION}" \
    && test "${MIXLAB_MLX_BUILD_COMMIT}" = "${MLX_COMMIT}" \
    && test "$(git -C /opt/mlx rev-parse HEAD)" = "${MLX_COMMIT}" \
    && cd /opt/mlx/build \
    && cmake .. -DMLX_BUILD_CUDA=ON -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_GGUF=OFF \
       -DMLX_CUDA_ARCHITECTURES="${ARCHS}" -DCMAKE_BUILD_TYPE=Release -G Ninja \
    && grep -Eq '^NCCL_LIBRARIES:FILEPATH=.*/libnccl' CMakeCache.txt \
    && ninja -j4 \
    && ninja install

WORKDIR /app
