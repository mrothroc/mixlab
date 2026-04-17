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

RUN cd /opt/mlx/build \
    && cmake .. -DMLX_BUILD_CUDA=ON -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_GGUF=OFF \
       -DMLX_CUDA_ARCHITECTURES="${ARCHS}" -DCMAKE_BUILD_TYPE=Release -G Ninja \
    && ninja -j4 \
    && ninja install

WORKDIR /app
