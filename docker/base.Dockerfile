# Layer 1: CUDA + Go + MLX compiled for sm_80 (A100)
# Keeps build artifacts for incremental architecture additions via addarch.Dockerfile.
#
# Build:  docker build -f docker/base.Dockerfile -t mixlab-cuda-base .
# ~30 min (compiles MLX from source with CUDA)

FROM --platform=linux/amd64 nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y \
    wget gcc g++ ninja-build git \
    libopenblas-dev liblapack-dev liblapacke-dev \
    python3 python3-dev python3-pip \
    libcudnn9-dev-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

# CMake 3.25+ required by MLX
RUN wget -q https://github.com/Kitware/CMake/releases/download/v3.29.3/cmake-3.29.3-linux-x86_64.tar.gz \
    && tar -C /usr/local --strip-components=1 -xzf cmake-3.29.3-linux-x86_64.tar.gz \
    && rm cmake-3.29.3-linux-x86_64.tar.gz

# Go
RUN wget -q https://go.dev/dl/go1.24.4.linux-amd64.tar.gz \
    && tar -C /usr/local -xzf go1.24.4.linux-amd64.tar.gz \
    && rm go1.24.4.linux-amd64.tar.gz
ENV PATH="/usr/local/go/bin:${PATH}"

# Pin MLX to a known release to avoid API breakage from HEAD.
ARG MLX_VERSION=v0.25.2
RUN git clone --branch ${MLX_VERSION} --depth 1 https://github.com/ml-explore/mlx.git /opt/mlx

# Build MLX with sm_80 ONLY — minimal first tier.
# KEEP the build directory for incremental arch additions.
RUN cd /opt/mlx \
    && mkdir -p build && cd build \
    && cmake .. -DMLX_BUILD_CUDA=ON -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_GGUF=OFF \
       -DMLX_CUDA_ARCHITECTURES="80" -DCMAKE_BUILD_TYPE=Release -G Ninja \
    && ninja -j4 \
    && ninja install

WORKDIR /app
