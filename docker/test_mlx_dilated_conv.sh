#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mlx="${MLX_PREFIX:-/usr/local}"
flags=(-L"$mlx/lib" -Wl,-rpath,"$mlx/lib" -lmlx)
if [[ "$(uname)" == Darwin ]]; then
  mlx="${MLX_PREFIX:-/opt/homebrew}"
  flags=(-L"$mlx/lib" -Wl,-rpath,"$mlx/lib" -lmlx)
else
  cuda="${CUDA_HOME:-/usr/local/cuda}"
  flags+=(-L"$cuda/lib64" -L"$cuda/lib64/stubs" -Wl,-rpath,"$cuda/lib64"
    -lcudart -lcublas -lcublasLt -lcudnn -lcufft -lcuda -lnvrtc -lnccl
    -lopenblas -llapack -lpthread -ldl)
fi
"${CXX:-c++}" -O2 -std=c++20 -I"$mlx/include" "$root/test_mlx_dilated_conv.cpp" "${flags[@]}" -o "$tmp/probe"
env -u NVIDIA_TF32_OVERRIDE MLX_ENABLE_TF32=0 "$tmp/probe" "$@"
env -u NVIDIA_TF32_OVERRIDE MLX_ENABLE_TF32=1 "$tmp/probe" "$@"
