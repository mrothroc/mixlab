#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cuda="${CUDA_HOME:-/usr/local/cuda}"
mlx="${MLX_PREFIX:-/usr/local}"
"${CXX:-c++}" -O2 -std=c++20 -I"$mlx/include" -I"$cuda/include" \
  "$root/test_mlx_conv_precision.cpp" -L"$mlx/lib" -L"$cuda/lib64" \
  -L"$cuda/lib64/stubs" -Wl,-rpath,"$mlx/lib" -Wl,-rpath,"$cuda/lib64" \
  -lmlx -lcudart -lcublas -lcublasLt -lcudnn -lcufft -lcuda -lnvrtc \
  -lnccl -lopenblas -llapack -lpthread -ldl -o "$tmp/probe"
for batch in 1 16; do
  env -u NVIDIA_TF32_OVERRIDE MLX_ENABLE_TF32=0 "$tmp/probe" "$batch"
  env -u NVIDIA_TF32_OVERRIDE MLX_ENABLE_TF32=1 "$tmp/probe" "$batch"
done
