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
COPY go.mod ./
COPY . .
RUN go mod download

RUN CGO_ENABLED=1 go build -tags mlx -o /mixlab ./cmd/mixlab \
    && echo "Build OK: $(file /mixlab)"

# --- Runtime image ---
# MLX JIT-compiles CUDA kernels at runtime, so it needs the full devel image
# (CUDA headers + nvrtc compiler). The runtime-only image does not work.
FROM --platform=linux/amd64 nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 liblapack3 \
    libcudnn9-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

# Binary
COPY --from=builder /mixlab /usr/local/bin/mixlab

# Example configs
COPY examples/ /examples/

ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_VISIBLE_DEVICES=all

RUN mixlab -mode smoke 2>&1 || echo "Smoke test skipped (no GPU in build)"

WORKDIR /data
ENTRYPOINT ["mixlab"]
CMD ["-help"]
