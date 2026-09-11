# mixlab CLI image for NVIDIA GPUs.
# Includes the embedded prepare runtime, but no RunPod handler.
#
# Build: docker build -f docker/app.Dockerfile -t mixlab .
# Run:   docker run --gpus all mixlab -mode smoke
#        docker run --gpus all -v $(pwd)/data:/data mixlab \
#            -mode arch -config /examples/plain_3L.json -train '/data/*.bin'

# Set BASE_IMAGE to your MLX CUDA base image.
ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS builder

# Refuse an old cached MLX base: rebuilding only the app cannot fix libmlx.
RUN test "${MIXLAB_MLX_CUDA_WORKER_FIX}" = "1" \
    || { echo "Rebuild the MLX CUDA base and architecture tiers with the worker fix" >&2; exit 1; }

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfmt-dev \
    && rm -rf /var/lib/apt/lists/*
COPY go.mod ./
COPY . .
RUN go mod download

# Assert that every kernel listed in cuda_kernels.list reached the generated
# registry. The names are derived from the list rather than hard-coded, so
# kernels added later are covered automatically; the previous form named two
# kernels by hand and silently stopped covering everything added after them.
RUN MIXLAB_REQUIRE_CUDA_KERNELS=1 bash gpu/cuda_kernels/generate_registry.sh \
    && while read -r kernel_path; do \
         case "${kernel_path}" in ''|\#*) continue ;; esac; \
         kernel_name="$(basename "${kernel_path}" .cu)"; \
         grep -q "\"${kernel_name}\"" gpu/cuda_kernels/registry_generated.h \
           || { echo "CUDA kernel missing from generated registry: ${kernel_name}" >&2; exit 1; }; \
       done < gpu/cuda_kernels/cuda_kernels.list

RUN CGO_ENABLED=1 go build -tags mlx -o /mixlab ./cmd/mixlab \
    && CGO_ENABLED=0 go build -o /mixlab-prepare-check ./cmd/mixlab \
    && echo "Build OK: $(file /mixlab)"

# --- Runtime image ---
# MLX JIT-compiles CUDA kernels at runtime, so use the same MLX CUDA base that
# built the binary. A plain CUDA image can have different CUDA/MLX libraries and
# report the MLX GPU backend as unavailable at runtime.
FROM ${BASE_IMAGE} AS runtime

ARG MIXLAB_VERSION=dev
ARG VCS_REF=unknown
LABEL org.opencontainers.image.version="${MIXLAB_VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.source="https://github.com/mrothroc/mixlab"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 liblapack3 \
    libcudnn9-cuda-12 python3-venv \
    && rm -rf /var/lib/apt/lists/*

# One dependency contract for local/CI prepare and both runtime images. A venv
# avoids changing distro packages and remains readable by arbitrary --user UIDs.
COPY requirements-prepare.txt /opt/mixlab/requirements-prepare.txt
RUN python3 -m venv /opt/mixlab/venv
ENV PATH="/opt/mixlab/venv/bin:${PATH}"
RUN python3 -m pip install --no-cache-dir -r /opt/mixlab/requirements-prepare.txt

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

# Build hosts have no NVIDIA driver. Exercise the same embedded prepare path in
# a non-MLX binary, against the actual runtime Python environment, without a
# source checkout or a writable home directory. Fail the build on any error.
FROM runtime AS prepare-check
COPY --from=builder /mixlab-prepare-check /tmp/mixlab-prepare-check
COPY docker/prepare_smoke.py /tmp/prepare_smoke.py
USER 10001:10001
WORKDIR /tmp
RUN python3 /tmp/prepare_smoke.py /tmp/mixlab-prepare-check --stamp /tmp/prepare-check.passed

FROM runtime AS final
COPY --from=prepare-check /tmp/prepare-check.passed /opt/mixlab/prepare-check.passed
