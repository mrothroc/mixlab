#!/usr/bin/env bash
# Explicit CUDA pressure test; requires an otherwise idle GPU with >=8 GiB free.
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cuda="${CUDA_HOME:-/usr/local/cuda}"
mlx="${MLX_PREFIX:-/usr/local}"
"${CXX:-c++}" -O2 -std=c++20 -I"$mlx/include" -I"$cuda/include" \
  "$root/test_mlx_allocator_cuda.cpp" -L"$mlx/lib" -L"$cuda/lib64" \
  -L"$cuda/lib64/stubs" -Wl,-rpath,"$mlx/lib" -Wl,-rpath,"$cuda/lib64" \
  -lmlx -lcudart -lcublas -lcublasLt -lcudnn -lcufft -lcuda -lnvrtc \
  -lnccl -lopenblas -llapack -lpthread -ldl -o "$tmp/probe"
"$tmp/probe" "$@"
