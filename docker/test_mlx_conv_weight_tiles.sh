#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mlx="${MLX_PREFIX:-/usr/local}"
cuda="${CUDA_HOME:-/usr/local/cuda}"
"${CXX:-c++}" -O2 -std=c++20 -I"$mlx/include" "$root/test_mlx_conv_weight_tiles.cpp" \
  -L"$mlx/lib" -Wl,-rpath,"$mlx/lib" -lmlx \
  -L"$cuda/lib64" -L"$cuda/lib64/stubs" -Wl,-rpath,"$cuda/lib64" \
  -lcudart -lcublas -lcublasLt -lcudnn -lcufft -lcusolver -lcuda -lnvrtc -lnccl \
  -lopenblas -llapack -lpthread -ldl -o "$tmp/probe"
env -u NVIDIA_TF32_OVERRIDE MLX_ENABLE_TF32=0 "$tmp/probe"
