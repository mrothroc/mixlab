# gpu/ — IR dispatcher, MLX bridge, custom primitives

This package executes the IR via MLX (Metal on macOS, CUDA on Linux). Forward + backward dispatch, custom primitives, CUDA kernels.

## Key files
- `ir.cpp` / `ir.h` — IR interpreter; one `case OP_*` per IR op. The biggest file in the package.
- `ir_trainer.cpp` / `ir_trainer.h` — `IRTrainer::submit_step` (the train-step state machine), optimizer apply, special paths for canonical Mamba-3
- `mlx_bridge.{cpp,h}` — cgo bridge between Go and MLX C++; manages tensor handles, evals, gradients
- `gated_delta_cuda_primitive.{cpp,h}` — CUDA primitive for OP_GATED_DELTA_SCAN (reference pattern)
- `mamba3_cuda_primitive.{cpp,h}` — CUDA primitives for OP_MAMBA3_SELECTIVE_SCAN forward + backward
- `gated_delta_metal_primitive.{cpp,h}` — Metal primitive for the same op (M1/M2)
- `cuda_graph_limits.go` — CUDA graph batching policy (per-op-type caps); see `train/cuda_graph_limits.go` for the wiring
- [`cuda_kernels/`](cuda_kernels/README.md) — `.cu` source + build pipeline for embedded fatbins

## MLX primitive pattern
Custom GPU compute is wrapped in an `mx::Primitive` subclass with `eval_gpu()` + `vjp()`. Examples: `SolveStrictlyLowerCUDAPrimitive` (gated_delta), `Mamba3SelectiveScanCUDAForwardPrimitive` + `Mamba3SelectiveScanCUDABackwardPrimitive` (mamba3 scan). MLX autograd handles the surrounding host-side MLX ops natively.

**Anti-pattern: do NOT wrap a long sequence of host-side MLX ops in `mx::custom_vjp`.** That forces MLX to trace and compile all of them as a single fused custom op on first call — 10+ minute CPU hangs at scale. Past incident: the canonical Mamba-3 fused block (commit `e1899bf`) wrapped 20+ MLX ops in `mx::custom_vjp` and had to be reverted (`3509a87`). Use `mx::Primitive` subclasses for the GPU-specific compute and let MLX autograd handle the host-side composition.

## Mamba-3 specifics
The canonical Mamba-3 path has 7 layers of memory/compile pressure mitigation. See [`../docs/canonical_mamba3.md`](../docs/canonical_mamba3.md) for the full architecture, env-var reference, and which file each layer lives in.

## Runtime gotchas
- `libcuda.so.1` is the NVIDIA driver lib, runtime-provided (NVIDIA Container Toolkit). The Dockerfile's ldd check excludes it — see `docker/app.Dockerfile`.
- Cloud Build has nvcc → CUDA kernels are precompiled and embedded. GitHub CI doesn't → kernels stub out, fallback paths run.
