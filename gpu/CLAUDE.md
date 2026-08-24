# gpu/ — IR dispatcher, MLX bridge, custom primitives

This package executes the IR via MLX (Metal on macOS, CUDA on Linux). Forward + backward dispatch, custom primitives, CUDA kernels.

## Key files
- `ir.cpp` / `ir.h` — IR interpreter; one `case OP_*` per IR op. The biggest file in the package.
- `ir_trainer.cpp` / `ir_trainer.h` — `IRTrainer::submit_step` (the train-step state machine), optimizer apply, special paths for canonical Mamba-3
- `mlx_bridge.{cpp,h}` — cgo bridge between Go and MLX C++; manages tensor handles, evals, gradients
- `gated_delta_cuda_primitive.{cpp,h}` — CUDA primitive for OP_GATED_DELTA_SCAN (reference pattern)
- `mamba3_cuda_primitive.{cpp,h}` — CUDA primitives for OP_MAMBA3_SELECTIVE_SCAN forward + backward
- `mamba3_metal_primitive.{cpp,h}` — Metal forward + CUDA-style parallel-window VJP for OP_MAMBA3_SELECTIVE_SCAN; short sequences retain sequential reverse replay
- `s4d_sobolev_cuda_primitive.{cpp,h}` — CUDA forward/backward primitive for the learned real Sobolev filter applied to S4D's complex FFT spectrum
- `gated_delta_chunk_metal_primitive.{cpp,h}` — exact chunk-parallel Metal forward and analytical VJP; the preferred Gated DeltaNet path for `d_k <= 32`, `d_v <= 32`, `scan_chunk_size <= 64`
- `gated_delta_metal_primitive.{cpp,h}` — Metal triangular solve, plus the recurrent Gated DeltaNet scan covering the rest of `d_k <= 64`, `d_v <= 256`. The backward checkpoints matrix state every `W <= min(scan_chunk_size, 8)` tokens and recomputes each window; see [`../docs/performance.md`](../docs/performance.md#gated-deltanet-long-sequences)
- `s4d_kernel_metal_primitive.{cpp,h}` — Metal forward/backward for bidirectional S4D kernel synthesis; writes the compact two-direction kernel without materializing the `[D,state_size/2,T]` power tensors
- `cuda_graph_limits.go` — CUDA graph batching policy (per-op-type caps); see `train/cuda_graph_limits.go` for the wiring
- [`cuda_kernels/`](cuda_kernels/README.md) — `.cu` source + build pipeline for embedded fatbins

## MLX primitive pattern
Custom GPU compute is wrapped in an `mx::Primitive` subclass with `eval_gpu()` + `vjp()`. Examples: `SolveStrictlyLowerCUDAPrimitive` (gated_delta), `Mamba3SelectiveScanCUDAForwardPrimitive` + `Mamba3SelectiveScanCUDABackwardPrimitive` (mamba3 scan). MLX autograd handles the surrounding host-side MLX ops natively.

`mx::custom_vjp` is legitimate for **gluing two primitives together** — a forward
`mx::Primitive` and a separate backward `mx::Primitive` — when the backward is an
analytical kernel rather than something MLX can differentiate. The S4D kernel and
Gated DeltaNet scan primitives both do this. What matters is that each side is real
GPU compute, not a traced op graph.

**Anti-pattern: do NOT wrap a long sequence of host-side MLX ops in `mx::custom_vjp`.** That forces MLX to trace and compile all of them as a single fused custom op on first call — 10+ minute CPU hangs at scale. Past incident: the canonical Mamba-3 fused block (commit `e1899bf`) wrapped 20+ MLX ops in `mx::custom_vjp` and had to be reverted (`3509a87`). Use `mx::Primitive` subclasses for the GPU-specific compute and let MLX autograd handle the host-side composition.

## Mamba-3 specifics
The canonical Mamba-3 path has 7 layers of memory/compile pressure mitigation. On Metal, trainer setup also prewarms the embedded Mamba-3 library while weights initialize/upload, and backward uses parallel window summaries once a sequence spans at least four replay windows. See [`../docs/canonical_mamba3.md`](../docs/canonical_mamba3.md) for the full architecture, env-var reference, and which file each layer lives in.

## Verifying a primitive

Every primitive needs an env flag that forces the fallback, and a differential test
that runs the same workload both ways and compares **forward and every gradient**,
not just the loss. A primitive with a wrong analytical VJP still trains and still
reports a falling loss.

**Confirm the primitive was actually live in the fast run.** If it was unavailable —
unsupported shape, missing kernel, wrong backend — both sides silently take the
fallback and match trivially, which looks exactly like a pass. Each primitive logs
its selected path once; assert on that log, or check it by hand when running the
test locally. This is the single most repeated way a primitive test has been wrong.

Flag names use two shapes, both in active use, so grep for `DISABLE` rather than
guessing: `MIXLAB_DISABLE_<THING>` (e.g. `MIXLAB_DISABLE_GATED_DELTA_METAL_SCAN`)
and `MIXLAB_<COMPONENT>_DISABLE_<THING>` (e.g. `MIXLAB_S4D_DISABLE_METAL_KERNEL_PRIMITIVE`).
[`../docs/performance.md`](../docs/performance.md) documents the ones intended for
operators; the rest are debug-only.

MLX can change autodiff behavior in a **patch** release. 0.32.1 made `take_along_axis`
reject a VJP with respect to its indices, which broke every MoE and bf16 path until
those indices were wrapped in `mx::stop_gradient`. Run the `-tags mlx` suite after any
MLX upgrade; CI cannot see this because it builds with `CGO_ENABLED=0`.

## Determinism / keyed RNG
- `OP_DROPOUT` takes an optional `dropout_keys` input; when present it draws its
  mask from an explicit per-op MLX PRNG key instead of the global RNG, so
  training resumes bit-exactly (see `train/dropout_rng.go`, `arch` keys the op).
- `OP_RANDOM_NORMAL` still uses the **global** MLX RNG (no key) — not
  resume-reproducible. Tracked as GitHub issue #3; key it the same way if you
  touch that path.

## Runtime gotchas
- `libcuda.so.1` is the NVIDIA driver lib, runtime-provided (NVIDIA Container Toolkit). The Dockerfile's ldd check excludes it — see `docker/app.Dockerfile`.
- Cloud Build has nvcc → CUDA kernels are precompiled and embedded. GitHub CI doesn't → kernels stub out, fallback paths run.
- Op codes are a stable on-disk/ABI contract shared across `arch/ir.go`, `gpu/ir.h`, `gpu/mlx_types.go`, `gpu/lower.go`, and the `TestIRToGPUOpCodeAlignment` table in `gpu/lower_test.go`. A new op MUST touch all five.
