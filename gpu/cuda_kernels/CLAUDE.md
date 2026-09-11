# gpu/cuda_kernels/ — agent orientation

Custom CUDA kernels embedded in the binary at build time. Full pipeline doc: [`README.md`](README.md).

## Quick reference
- `.cu` files in this directory + `cuda_kernels.list` → `generate_registry.sh` runs `nvcc -fatbin` for the architectures in its `ARCHES` array → embeds bytes in `registry_generated.h` → linked into binary. The generator also emits `ARCHES` into that header as `kEmbeddedCudaKernelArchitectures`, so the runtime can report what it was built for; read `ARCHES` rather than trusting a list copied into prose.
- `nvcc` absent at build time (e.g., GitHub CI) → empty registry → primitives detect missing kernel → MLX-composed fallback.
- **Registry populated but unloadable on the running GPU** → a different path: `-gencode code=sm_XX` embeds SASS only, no PTX, so a fatbin cannot JIT forward onto an architecture outside `ARCHES`. `cuda_kernel_dispatch.cpp` recompiles each kernel from embedded source via NVRTC. Results stay correct — it is the same source — but the precompiled path is inert and every process pays compilation on first use. Because nothing looks wrong, one banner names the running GPU with its `sm_` version and the built-for list. Seen in production on an sm_120 Blackwell.
- Cloud Build (production image build) has `nvcc` → real kernels embedded.
- `libcuda.so.1` is runtime-provided by NVIDIA Container Toolkit; the Dockerfile ldd check tolerates it being missing at build time.
- Adding kernels for newer GPUs: extend `ARCHES=(...)` in `generate_registry.sh`.
