## Build-time CUDA kernels

All custom CUDA kernels for mixlab live in this directory.

Pattern:
- Add a `.cu` file here. Its basename is the kernel symbol used at runtime.
- Add the relative path to `cuda_kernels.list`.
- `generate_registry.sh` compiles each listed kernel to a multi-arch fatbin with SASS for `sm_80`, `sm_86`, `sm_89`, and `sm_90`, then emits `registry_generated.h`.
- Runtime code launches kernels via `mlx_ir::launch_precompiled_cuda_kernel(...)`, which hands the embedded fatbin to MLX/CUDA. The driver selects the matching image for the active GPU.

Platform behavior:
- macOS / Metal-only: if `nvcc` is unavailable, the generator emits an empty registry header and the build still succeeds.
- Linux / CUDA builders: fatbins are generated and baked into the binary. No runtime NVRTC step.

To add the next kernel:
1. Add `gpu/cuda_kernels/my_kernel.cu` with `extern "C" __global__ void my_kernel(...)`.
2. Add `gpu/cuda_kernels/my_kernel.cu` to `cuda_kernels.list`.
3. Launch it from a primitive through `launch_precompiled_cuda_kernel("my_kernel", ...)`.
