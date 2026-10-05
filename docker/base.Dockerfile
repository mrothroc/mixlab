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

# CMake 3.25+ required by MLX. The checksum is Kitware's published SHA-256.
RUN wget -q https://github.com/Kitware/CMake/releases/download/v3.29.3/cmake-3.29.3-linux-x86_64.tar.gz \
    && echo "90b543a30220401db0e08347af067545be158ce89ffb09b7df1516cda8617329  cmake-3.29.3-linux-x86_64.tar.gz" | sha256sum -c - \
    && tar -C /usr/local --strip-components=1 -xzf cmake-3.29.3-linux-x86_64.tar.gz \
    && rm cmake-3.29.3-linux-x86_64.tar.gz

# Go. GO_VERSION must equal the toolchain line in go.mod (TestGoToolchainHasOneSource
# enforces it); GO_SHA256 is the archive checksum published at go.dev/dl.
ARG GO_VERSION=1.27.1
ARG GO_SHA256=63d339f0da5ab53635a56f2490a7984dfe12dfcff22ad749f63edaf590168445
RUN wget -q "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" -O /tmp/go.tar.gz \
    && echo "${GO_SHA256}  /tmp/go.tar.gz" | sha256sum -c - \
    && tar -C /usr/local -xzf /tmp/go.tar.gz \
    && rm /tmp/go.tar.gz
ENV PATH="/usr/local/go/bin:${PATH}"

# Pin MLX to an immutable release commit. The tag check catches accidental
# disagreement between the human-readable release and the commit used by CI.
ARG MLX_VERSION=v0.32.0
ARG MLX_COMMIT=7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247
RUN git clone --branch ${MLX_VERSION} --depth 1 https://github.com/ml-explore/mlx.git /opt/mlx \
    && test "$(git -C /opt/mlx rev-parse HEAD)" = "${MLX_COMMIT}"

ENV MIXLAB_MLX_BUILD_VERSION=${MLX_VERSION}
ENV MIXLAB_MLX_BUILD_COMMIT=${MLX_COMMIT}

# Local dependency fix, not a version upgrade. Test the pinned worker body with
# stubbed CUDA event delivery before/after patching, without requiring a GPU.
COPY docker/patches/mlx-cuda-worker-wait.patch /tmp/mlx-cuda-worker-wait.patch
COPY docker/test_mlx_worker.py /tmp/test_mlx_worker.py
RUN python3 /tmp/test_mlx_worker.py --source /opt/mlx/mlx/backend/cuda/worker.cpp --expect-spin \
    && git -C /opt/mlx apply --check /tmp/mlx-cuda-worker-wait.patch \
    && git -C /opt/mlx apply /tmp/mlx-cuda-worker-wait.patch \
    && python3 /tmp/test_mlx_worker.py --source /opt/mlx/mlx/backend/cuda/worker.cpp
ENV MIXLAB_MLX_CUDA_WORKER_FIX=1

# Exact transposed-convolution weight VJP via ordinary-convolution patches.
# Avoid the CUDA fallback's enormous input-dilated unfold in decoder training.
COPY docker/patches/mlx-conv-transpose-weight-grad.patch /tmp/mlx-conv-transpose-weight-grad.patch
RUN git -C /opt/mlx apply --check /tmp/mlx-conv-transpose-weight-grad.patch \
    && git -C /opt/mlx apply /tmp/mlx-conv-transpose-weight-grad.patch
ENV MIXLAB_MLX_CONV_TRANSPOSE_GRAD_FIX=1

# Bound the logical cache and reclaim completed pool frees before retrying OOM.
# Exercise the pinned methods without CUDA before/after applying the patch.
COPY docker/patches/mlx-cuda-allocator-reclaim.patch /tmp/mlx-cuda-allocator-reclaim.patch
COPY docker/test_mlx_allocator.py /tmp/test_mlx_allocator.py
RUN python3 /tmp/test_mlx_allocator.py --source /opt/mlx/mlx/backend/cuda/allocator.cpp --expect-broken \
    && git -C /opt/mlx apply --check /tmp/mlx-cuda-allocator-reclaim.patch \
    && git -C /opt/mlx apply /tmp/mlx-cuda-allocator-reclaim.patch \
    && python3 /tmp/test_mlx_allocator.py --source /opt/mlx/mlx/backend/cuda/allocator.cpp
ENV MIXLAB_MLX_CUDA_ALLOCATOR_FIX=1

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
