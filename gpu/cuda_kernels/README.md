## Build-time CUDA kernels

All custom CUDA kernels for mixlab live in this directory.

Pattern:
- Add a `.cu` file here. Its basename is the kernel symbol used at runtime.
- Keep one registered launch symbol per `.cu` file. Put helpers shared by several
  kernels in a local `.cuh` file; the registry generator inlines those includes
  into the embedded source fallback.
- Add the relative path to `cuda_kernels.list`.
- `generate_registry.sh` compiles each listed kernel to a multi-arch fatbin with SASS for `sm_80`, `sm_86`, `sm_89`, and `sm_90`, then emits `registry_generated.h`.
- Runtime code launches kernels via `mlx_ir::launch_precompiled_cuda_kernel(...)`, which hands the embedded fatbin to MLX/CUDA. The driver selects the matching image for the active GPU.

`ARCHES` in `generate_registry.sh` is the single source of truth: the generator
also emits it into `registry_generated.h` as `kEmbeddedCudaKernelArchitectures`,
so the runtime can report what the image was built for.

Platform behavior:
- macOS / Metal-only: if `nvcc` is unavailable, the generator emits an empty registry header and the build still succeeds.
- Linux / CUDA builders: fatbins are generated and baked into the binary — with no runtime NVRTC step **as long as the GPU's architecture is in `ARCHES`**.

## When the GPU is not in `ARCHES`

`-gencode arch=compute_XX,code=sm_XX` embeds SASS only, not PTX, so a fatbin
cannot JIT forward onto a newer architecture. On a GPU outside `ARCHES` every
precompiled load fails and `cuda_kernel_dispatch.cpp` recompiles each kernel
from embedded source via NVRTC.

**Results stay correct** — it is the same kernel source — but the precompiled
path is entirely inert and every process pays compilation on first use of each
kernel. Because nothing else looks wrong, the dispatcher prints one loud banner
on the first failure naming the running GPU, its `sm_` version, and the
architectures the image was built for:

```
================================================================================
[cuda_kernel_dispatch] WARNING: PRECOMPILED CUDA KERNELS UNUSABLE ON THIS GPU
  this GPU:      NVIDIA RTX PRO 6000 Blackwell Workstation Edition (sm_120)
  kernels built: sm_80, sm_86, sm_89, sm_90
...
```

The fix is to add the architecture to `ARCHES` and rebuild the image. Adding a
`code=compute_XX` PTX entry would let future GPUs JIT forward instead of failing
outright, at the cost of a larger fatbin.

To add the next kernel:
1. Add `gpu/cuda_kernels/my_kernel.cu` with `extern "C" __global__ void my_kernel(...)`.
2. Add `gpu/cuda_kernels/my_kernel.cu` to `cuda_kernels.list`.
3. Launch it from a primitive through `launch_precompiled_cuda_kernel("my_kernel", ...)`.
