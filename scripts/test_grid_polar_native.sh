#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mlx="${MLX_PREFIX:-/usr/local}"
if [[ "$(uname)" == Darwin ]]; then mlx="${MLX_PREFIX:-/opt/homebrew}"; fi
flags=(-L"$mlx/lib" -Wl,-rpath,"$mlx/lib" -lmlx)
if [[ "$(uname)" != Darwin ]]; then
  cuda="${CUDA_HOME:-/usr/local/cuda}"
  flags+=(-L"$cuda/lib64" -L"$cuda/lib64/stubs" -Wl,-rpath,"$cuda/lib64"
    -lcudart -lcublas -lcublasLt -lcudnn -lcufft -lcusolver -lcuda -lnvrtc -lnccl
    -lopenblas -llapack -lpthread -ldl)
fi
"${CXX:-c++}" -O2 -std=c++20 -I"$mlx/include" -I"$root/gpu" \
  "$root/gpu/testdata/grid_polar_native.cpp" "$root/gpu/grid_polar.cpp" "${flags[@]}" -o "$tmp/probe"
env -u NVIDIA_TF32_OVERRIDE MLX_ENABLE_TF32=0 "$tmp/probe"
