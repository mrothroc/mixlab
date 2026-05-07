#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIST_FILE="${ROOT_DIR}/gpu/cuda_kernels/cuda_kernels.list"
HEADER_FILE="${ROOT_DIR}/gpu/cuda_kernels/registry_generated.h"
BUILD_DIR="${ROOT_DIR}/gpu/cuda_kernels/.build"
ARCHES=(80 86 89 90)

write_empty_header() {
  cat >"${HEADER_FILE}" <<'EOF'
#ifndef MLX_IR_CUDA_KERNELS_REGISTRY_GENERATED_H
#define MLX_IR_CUDA_KERNELS_REGISTRY_GENERATED_H

namespace mlx_ir::cuda_kernels {

struct EmbeddedKernelImage {
  const char* kernel_name;
  const unsigned char* blob;
  unsigned int blob_len;
  const unsigned char* source;
  unsigned int source_len;
};

static constexpr EmbeddedKernelImage kEmbeddedCudaKernelImages[] = {};
static constexpr unsigned int kEmbeddedCudaKernelImageCount = 0;

} // namespace mlx_ir::cuda_kernels

#endif
EOF
}

if [[ ! -f "${LIST_FILE}" ]]; then
  write_empty_header
  exit 0
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found; emitting empty CUDA kernel registry" >&2
  write_empty_header
  exit 0
fi

mkdir -p "${BUILD_DIR}"
mapfile -t KERNEL_PATHS < <(grep -v '^[[:space:]]*#' "${LIST_FILE}" | sed '/^[[:space:]]*$/d')

if [[ "${#KERNEL_PATHS[@]}" -eq 0 ]]; then
  write_empty_header
  exit 0
fi

JSON_FILE="${BUILD_DIR}/kernel_images.json"
printf '[]' > "${JSON_FILE}"

for kernel_rel in "${KERNEL_PATHS[@]}"; do
  kernel_src="${ROOT_DIR}/${kernel_rel}"
  if [[ ! -f "${kernel_src}" ]]; then
    echo "missing CUDA kernel source: ${kernel_rel}" >&2
    exit 1
  fi
  kernel_name="$(basename "${kernel_src}" .cu)"
  fatbin_out="${BUILD_DIR}/${kernel_name}.fatbin"
  nvcc_args=(-fatbin -std=c++17)
  for arch in "${ARCHES[@]}"; do
    nvcc_args+=(-gencode "arch=compute_${arch},code=sm_${arch}")
  done
  nvcc "${nvcc_args[@]}" "${kernel_src}" -o "${fatbin_out}"
  python3 - "${JSON_FILE}" "${kernel_name}" "${fatbin_out}" "${kernel_src}" <<'PY'
import json
import pathlib
import re
import sys

json_path = pathlib.Path(sys.argv[1])
kernel_name = sys.argv[2]
fatbin_path = pathlib.Path(sys.argv[3])
kernel_src_path = pathlib.Path(sys.argv[4])

include_re = re.compile(r'^\s*#\s*include\s+"([^"]+)"\s*$')

def inline_local_includes(path, seen):
    path = path.resolve()
    if path in seen:
        return ""
    seen.add(path)
    out = []
    for line in path.read_text().splitlines():
        match = include_re.match(line)
        if match:
            candidate = path.parent / match.group(1)
            if candidate.exists():
                out.append(inline_local_includes(candidate, seen))
                continue
        out.append(line)
    return "\n".join(out) + "\n"

data = json.loads(json_path.read_text())
data.append({
    "kernel_name": kernel_name,
    "bytes": list(fatbin_path.read_bytes()),
    "source": list(inline_local_includes(kernel_src_path, set()).encode()),
})
json_path.write_text(json.dumps(data))
PY
done

python3 - "${JSON_FILE}" "${HEADER_FILE}" <<'PY'
import json
import pathlib
import re
import sys

json_path = pathlib.Path(sys.argv[1])
header_path = pathlib.Path(sys.argv[2])
entries = json.loads(json_path.read_text())

def ident(name: str) -> str:
    return re.sub(r'[^0-9A-Za-z_]', '_', name)

lines = [
    "#ifndef MLX_IR_CUDA_KERNELS_REGISTRY_GENERATED_H",
    "#define MLX_IR_CUDA_KERNELS_REGISTRY_GENERATED_H",
    "",
    "namespace mlx_ir::cuda_kernels {",
    "",
    "struct EmbeddedKernelImage {",
    "  const char* kernel_name;",
    "  const unsigned char* blob;",
    "  unsigned int blob_len;",
    "  const unsigned char* source;",
    "  unsigned int source_len;",
    "};",
    "",
]

registry_names = []
for entry in entries:
    arr_name = f'kFatbin_{ident(entry["kernel_name"])}'
    src_name = f'kSource_{ident(entry["kernel_name"])}'
    registry_names.append((arr_name, src_name, entry))
    lines.append(f"static const unsigned char {arr_name}[] = {{")
    blob = entry["bytes"]
    for i in range(0, len(blob), 12):
        chunk = blob[i:i+12]
        lines.append("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",")
    lines.append("};")
    lines.append("")
    lines.append(f"static const unsigned char {src_name}[] = {{")
    source = entry["source"]
    for i in range(0, len(source), 12):
        chunk = source[i:i+12]
        lines.append("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",")
    lines.append("};")
    lines.append("")

lines.append("static constexpr EmbeddedKernelImage kEmbeddedCudaKernelImages[] = {")
for arr_name, src_name, entry in registry_names:
    lines.append(
        f'  {{"{entry["kernel_name"]}", {arr_name}, {len(entry["bytes"])}, '
        f'{src_name}, {len(entry["source"])}}},')
lines.append("};")
lines.append(f"static constexpr unsigned int kEmbeddedCudaKernelImageCount = {len(registry_names)};")
lines.append("")
lines.append("} // namespace mlx_ir::cuda_kernels")
lines.append("")
lines.append("#endif")

header_path.write_text("\n".join(lines) + "\n")
PY
