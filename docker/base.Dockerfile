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

# The NVIDIA devel image provides matching held NCCL runtime/development
# packages. Keep that CUDA-matched version and fail if a future base drops it.
RUN test -f /usr/include/nccl.h \
    && ldconfig -p | grep -q 'libnccl\.so'

# CMake 3.25+ required by MLX
RUN wget -q https://github.com/Kitware/CMake/releases/download/v3.29.3/cmake-3.29.3-linux-x86_64.tar.gz \
    && tar -C /usr/local --strip-components=1 -xzf cmake-3.29.3-linux-x86_64.tar.gz \
    && rm cmake-3.29.3-linux-x86_64.tar.gz

# Go
RUN wget -q https://go.dev/dl/go1.24.4.linux-amd64.tar.gz \
    && tar -C /usr/local -xzf go1.24.4.linux-amd64.tar.gz \
    && rm go1.24.4.linux-amd64.tar.gz
ENV PATH="/usr/local/go/bin:${PATH}"

# Pin MLX to an immutable release commit. The tag check catches accidental
# disagreement between the human-readable release and the commit used by CI.
ARG MLX_VERSION=v0.32.0
ARG MLX_COMMIT=7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247
RUN git clone --branch ${MLX_VERSION} --depth 1 https://github.com/ml-explore/mlx.git /opt/mlx \
    && test "$(git -C /opt/mlx rev-parse HEAD)" = "${MLX_COMMIT}"

ENV MIXLAB_MLX_BUILD_VERSION=${MLX_VERSION}
ENV MIXLAB_MLX_BUILD_COMMIT=${MLX_COMMIT}

# Build MLX with sm_80 ONLY — minimal first tier.
# KEEP the build directory for incremental arch additions.
RUN cd /opt/mlx \
    && mkdir -p build && cd build \
    && cmake .. -DMLX_BUILD_CUDA=ON -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_GGUF=OFF \
       -DMLX_CUDA_ARCHITECTURES="80" -DCMAKE_BUILD_TYPE=Release -G Ninja \
    && grep -Eq '^NCCL_LIBRARIES:FILEPATH=.*/libnccl' CMakeCache.txt \
    && ninja -j4 \
    && ninja install

WORKDIR /app
