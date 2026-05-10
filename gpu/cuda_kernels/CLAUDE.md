# gpu/cuda_kernels/ — agent orientation

Custom CUDA kernels embedded in the binary at build time. Full pipeline doc: [`README.md`](README.md).

## Quick reference
- `.cu` files in this directory + `cuda_kernels.list` → `generate_registry.sh` runs `nvcc -fatbin` for `sm_80/86/89/90` → embeds bytes in `registry_generated.h` → linked into binary.
- `nvcc` absent at build time (e.g., GitHub CI) → empty registry → primitives detect missing kernel → MLX-composed fallback.
- Cloud Build (production image build) has `nvcc` → real kernels embedded.
- `libcuda.so.1` is runtime-provided by NVIDIA Container Toolkit; the Dockerfile ldd check tolerates it being missing at build time.
- Adding kernels for newer GPUs: extend `ARCHES=(...)` in `generate_registry.sh`.
