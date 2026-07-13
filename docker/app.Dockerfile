# mixlab CLI image for NVIDIA GPUs.
# No Python, no RunPod handler — just the Go binary + example configs.
#
# Build: docker build -f docker/app.Dockerfile -t mixlab .
# Run:   docker run --gpus all mixlab -mode smoke
#        docker run --gpus all -v $(pwd)/data:/data mixlab \
#            -mode arch -config /examples/plain_3L.json -train '/data/*.bin'

# Set BASE_IMAGE to your MLX CUDA base image.
ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfmt-dev \
    && rm -rf /var/lib/apt/lists/*
COPY go.mod ./
COPY . .
RUN go mod download

RUN MIXLAB_REQUIRE_CUDA_KERNELS=1 bash gpu/cuda_kernels/generate_registry.sh \
    && grep -q 'mamba3_selective_scan_fwd' gpu/cuda_kernels/registry_generated.h \
    && grep -q 'ttt_mlp_causal_conv' gpu/cuda_kernels/registry_generated.h

RUN CGO_ENABLED=1 go build -tags mlx -o /mixlab ./cmd/mixlab \
    && echo "Build OK: $(file /mixlab)"

# --- Runtime image ---
# MLX JIT-compiles CUDA kernels at runtime, so use the same MLX CUDA base that
# built the binary. A plain CUDA image can have different CUDA/MLX libraries and
# report the MLX GPU backend as unavailable at runtime.
FROM ${BASE_IMAGE} AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 liblapack3 \
    libcudnn9-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

# Binary
COPY --from=builder /mixlab /usr/local/bin/mixlab
# Allow libcuda.so.1 to be missing at build time — it's the NVIDIA driver lib,
# bind-mounted by NVIDIA Container Toolkit at runtime on the GPU host. All
# other "not found" entries are real build-time errors.
RUN ldd /usr/local/bin/mixlab \
    && ! ldd /usr/local/bin/mixlab | grep 'not found' | grep -qv 'libcuda\.so\.1'

# Example configs
COPY examples/ /examples/

ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_VISIBLE_DEVICES=all

RUN mixlab -mode smoke 2>&1 || echo "Smoke test skipped (no GPU in build)"

WORKDIR /data
ENTRYPOINT ["mixlab"]
CMD ["-help"]
