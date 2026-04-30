#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CU_FILE="${ROOT_DIR}/gpu/gated_delta_kernels.cu"
PTX_FILE="${ROOT_DIR}/gpu/gated_delta_kernels.ptx"
HEADER_FILE="${ROOT_DIR}/gpu/gated_delta_kernels_ptx.h"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found; leaving existing ${HEADER_FILE}" >&2
  exit 0
fi

nvcc -ptx -arch=compute_80 "${CU_FILE}" -o "${PTX_FILE}"

python3 - "${PTX_FILE}" "${HEADER_FILE}" <<'PY'
import pathlib
import sys

ptx_path = pathlib.Path(sys.argv[1])
header_path = pathlib.Path(sys.argv[2])
data = ptx_path.read_bytes()

lines = [
    "#ifndef MLX_GATED_DELTA_KERNELS_PTX_H",
    "#define MLX_GATED_DELTA_KERNELS_PTX_H",
    "",
    "static const unsigned char kGatedDeltaKernelsPtx[] = {",
]

for i in range(0, len(data), 12):
    chunk = data[i:i+12]
    lines.append("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",")

lines.extend([
    "};",
    f"static const unsigned int kGatedDeltaKernelsPtxLen = {len(data)};",
    "",
    "#endif",
    "",
])

header_path.write_text("\n".join(lines))
PY
