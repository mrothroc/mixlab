# Canonical Mamba-3 block

The `mamba3-canonical` block implements Lahoti et al. 2026 — *"Mamba-3: Improved Sequence Modeling using State Space Principles"* (arXiv 2603.15569).

## Paper-aligned features

| Improvement | Source | Implementation |
|---|---|---|
| Exponential-trapezoidal recurrence | §3.1 Prop. 1 Eq. 5–6 | `α_t = exp(Δt·A)`, `β_t = (1-λ)·Δt·α`, `γ_t = λ·Δt`; three-term update `h_t = α·h_{t-1} + β·B_{t-1}x_{t-1} + γ·B_t·x_t` |
| Complex/rotational state | §3.2 Prop. 2–4 Eq. 9, 11, 25 | Real 2×2 rotations via the RoPE trick; cumulative `phi_t = Σ Δ_s θ_s` applied to B/C |
| Grouped/MIMO B,C | §3.3 + Appx C | `n_groups` (rank R in paper); B,C indexed `[B,T,G,N]`; gradients accumulate per-group |

## Block architecture
```
x → RMSNorm → MatMul(W_X) → [optional Conv1d] →
    dt = MatMul(W_dt_low @ W_dt_high) + dt_bias    (low-rank)
    λ  = MatMul(W_λ_low @ W_λ_high)                 (low-rank)
    θ  = MatMul(W_θ_low @ W_θ_high)                 (low-rank)
    B  = RMSNorm(MatMul(W_B)) + B_bias              (per-group N)
    C  = RMSNorm(MatMul(W_C)) + C_bias              (per-group N)
    y  = Mamba3SelectiveScan(x_branch, dt, λ, θ, A_log, B, C)
    y  = RMSNorm(y)
    z  = SiLU(MatMul(x_norm, W_Z))
    out = MatMul(y * z, W_O)
    return x + out
```

Config knobs (all optional):
- `inner_dim` (default `D`)
- `state_size` (default 16; must be even for complex pairs)
- `n_groups` (default 4; must divide `inner_dim`)
- `dt_rank` (default `max(inner/16, 1)`)
- `conv_kernel` (default 4)
- `use_conv` (default `true`)
- `scan_chunk_size` (default 64; `0` = full-sequence scan)

## Production-readiness stack

Canonical Mamba-3 at competition scale (D=448, T=4096, 8 layers on H100) needed seven layers of memory/compile-pressure mitigation. Each layer addresses a distinct bottleneck.

| # | What | Where | Why |
|---|---|---|---|
| 1 | Time-axis chunked Hillis-Steele scan | `gpu/ir.cpp::affine_scan_chunked` | Full T-length scan materializes T-sized intermediates per pass; chunk to bound them |
| 2 | Channel-axis chunking | `gpu/ir.cpp::*_channel_chunked` | Even after time chunking, `[B,T,D,N]` intermediates are huge at D=448; chunk D too |
| 3 | Hand-written closed-form VJP | `gpu/ir.cpp::mamba3_selective_scan_canonical_phase6_vjp` | MLX autodiff through the parallel scan creates oversized compiled CUDA graphs |
| 4 | Mamba-3-aware CUDA graph caps | `gpu/cuda_graph_limits.go` | `MLX_MAX_OPS_PER_BUFFER=64`, `MLX_MAX_MB_PER_BUFFER=128`, `MLX_CUDA_GRAPH_CACHE_SIZE=1024` for any program with the scan or fused block op |
| 5 | Uncompiled trainer fallback | `gpu/ir_trainer.cpp::use_compiled_training_step` | Even with custom VJP, `mx::compile` over the full step graph fuses everything into one CUDA graph instance — too big at H100 scale |
| 6 | Native CUDA and Metal kernels | `gpu/mamba3_cuda_primitive.{cpp,h}`, `gpu/cuda_kernels/mamba3_selective_scan_*.cu`, and `gpu/mamba3_metal_primitive.{cpp,h}` | Backend-specific `mx::Primitive` subclasses run the recurrence and its closed-form VJP without materializing Hillis-Steele intermediates. Metal uses the same window-summary/carry/emit backward decomposition as CUDA for sequences with at least four replay windows. Unsupported shapes and explicitly disabled primitives retain the MLX-composed fallback. |
| 7 | Fused `OP_MAMBA3_CANONICAL_BLOCK` op | `arch/ir.go` + `gpu/ir.cpp` | Block emitted as a single IR op, not 25 separate ones. The IR-side handler does the entire forward + backward in one C++ function. **Do not wrap it in `mx::custom_vjp`** — that recreates the graph-fusion problem (see "Anti-pattern" below). |

## Anti-pattern: don't wrap host-side MLX ops in `mx::custom_vjp`

Wrapping a sequence of host-side MLX ops in `mx::custom_vjp` causes MLX to trace and compile all of them as a single fused custom op on first invocation — 10+ minute CPU hangs at scale. The CUDA-specific compute should be its own `mx::Primitive` subclass (with `eval_gpu` + `vjp`); the surrounding MLX ops should be plain (no wrapper) so MLX autograd handles them natively. This is the gated_delta pattern.

Past incident: commit `e1899bf` wrapped 20+ MLX ops in `mx::custom_vjp` for the fused canonical Mamba-3 block; reverted in `3509a87`.

## Environment variables

| Var | Default | Effect |
|---|---|---|
| `MIXLAB_MAMBA3_DISABLE_CUDA_PRIMITIVE` | unset | Disable the native selective-scan primitive for small debug fallback runs. Unsupported for fused canonical block training unless `MIXLAB_ALLOW_MAMBA3_MLX_SCAN_FALLBACK=1` is also set. |
| `MIXLAB_MAMBA3_DISABLE_METAL_PRIMITIVE` | unset | Disable the native Metal forward/VJP primitive and use the MLX-composed fallback. Intended for parity tests and debugging. |
| `MIXLAB_MAMBA3_DISABLE_METAL_PARALLEL_BACKWARD` | unset | Use the lower-memory sequential Metal backward instead of the parallel window-summary/carry/emit path. Short sequences with fewer than four replay windows select the sequential path automatically. |
| `MIXLAB_MAMBA3_DISABLE_METAL_PREWARM` | unset | Disable asynchronous Metal pipeline prewarming during trainer weight initialization and upload. |
| `MIXLAB_ALLOW_MAMBA3_MLX_SCAN_FALLBACK` | unset | Explicitly allow the debug MLX-composed scan fallback inside fused canonical block training. This can create invalid or oversized CUDA graphs at production scale. |
| `MIXLAB_MAMBA3_SCAN_FWD_CHUNK` | `64` | Time chunk length for the chunked forward CUDA scan. |
| `MIXLAB_MAMBA3_BWD_WINDOW` | `32` | Replay window length for native CUDA and Metal backward. Values up to `64` are supported; `32` bounds per-thread replay state. |
| `MIXLAB_MAMBA3_CHANNEL_CHUNK` | auto (~16ch at production) | Channels per chunk in the MLX-composed fallback; native CUDA and Metal primitives do not use it. |
| `MIXLAB_FORCE_COMPILED_STEP` | unset | Force `mx::compile` even for Mamba-3 programs |
| `MIXLAB_DISABLE_COMPILED_STEP` | unset | Force eager `value_and_grad` for any program |
| `MIXLAB_DISABLE_TRAINING_CHECKPOINT` | unset | Skip `mx::checkpoint` on the eager path |
| `MIXLAB_DISABLE_MAMBA3_LOW_MEMORY_UPDATES` | unset | Disable per-chunk gradient/optimizer fusion |
| `MIXLAB_FORCE_MAMBA3_COMPILED_UPDATE_STEP` | unset | Force compiled update path even if it OOMs |
| `MIXLAB_MAMBA3_HOST_TIMING_START` | `100` | First step that emits host-timing log |
| `MIXLAB_MAMBA3_HOST_TIMING_EVERY` | `100` | Cadence of host-timing log |
| `MIXLAB_MAMBA3_HOST_TIMING` | unset | Emit fused canonical Mamba3 host timing logs |
| `MIXLAB_DISABLE_MAMBA3_HOST_TIMING` | unset | Suppress host-timing log entirely |
| `MLX_MAX_OPS_PER_BUFFER` | auto via `gpu.TuneCUDAGraphLimits` | User override of CUDA graph batching cap |
| `MLX_MAX_MB_PER_BUFFER` | auto | Same, by total bytes |
| `MLX_CUDA_GRAPH_CACHE_SIZE` | auto (`1024`) | CUDA graph variant cache size for canonical Mamba3; explicit user values are preserved |

## Verification tests
- `train/gpu_trainer_mamba3_test.go::TestMamba3SelectiveScanGrad` — analytical CPU oracle vs MLX gradients (~1e-6 relative error) at G ∈ {1,2}, chunk ∈ {0,3}
- `TestMamba3SelectiveScanGradChannelChunked` — exercises channel-chunking path with `MIXLAB_MAMBA3_CHANNEL_CHUNK=2`
- `TestMamba3SelectiveScanGradSmallCudaChunks` — exercises small native CUDA forward chunks and backward windows
- `TestMamba3SelectiveScanMetalMatchesMLXFallback` — compares native Metal loss and all seven gradient families with the MLX-composed fallback
- `TestMamba3CanonicalBlockGradMatchesExpanded` — fused-op vs expanded-25-op equivalence (loss + all 20 weight gradients within 2e-5)
- `TestMamba3CanonicalSmokeLossDecreases` — 10-step smoke; loss must monotonically decrease

## Metal performance

The Metal primitive assigns one 32-lane SIMD group to each `(batch, channel)`.
State pairs remain in registers through the forward recurrence. Backward stores
bounded checkpoints, computes independent per-window affine summaries, scans the
small summary sequence for reverse carries, then emits token and weight gradients
from all windows in parallel. Shared grouped B/C gradients use atomic
accumulation. Sequences with fewer than four replay windows use the prior
sequential reverse replay to avoid temporary-buffer overhead.

On a 32-core M1 Max, the repository benchmark at `B=2`, `T=1024`, `D=128`,
`N=16`, `G=4` improved from approximately 24.1 ms/op on the MLX-composed
fallback to 7.6 ms/op on the sequential native Metal path and 5.8 ms/op with
parallel-window backward. A warmed two-block canonical training benchmark at
`D=128`, `T=512` improved from approximately 21.5 ms/step on the composed
fallback to 14.0 ms/step on sequential Metal and 13.1 ms/step on parallel
Metal.

At the production probe shape (`D=448`, `T=4096`, eight canonical blocks
interleaved with eight SwiGLU blocks), parallel backward reduced a warmed step
from approximately 562.9 ms to 498.3 ms. Peak MLX memory increased from
approximately 2980.9 MiB to 3033.5 MiB for window summaries and carries. These
process-level probes varied, with full-step improvements ranging from roughly
3% to 11%. These numbers are local M1 Max measurements and should be treated as
relative, not portable absolute performance.

Trainer setup asynchronously prewarms the embedded Metal library while weights
are initialized and uploaded, then waits before trainer creation. This moves
pipeline compilation off the first submitted step when setup work is long enough
to cover it. Run the local comparisons with:

```bash
go test -tags mlx ./train -run '^$' \
  -bench '^BenchmarkMamba3SelectiveScanForwardBackward$' -benchtime=100x
MIXLAB_MAMBA3_DISABLE_METAL_PARALLEL_BACKWARD=1 go test -tags mlx ./train -run '^$' \
  -bench '^BenchmarkMamba3SelectiveScanForwardBackward$' -benchtime=100x
MIXLAB_MAMBA3_DISABLE_METAL_PRIMITIVE=1 go test -tags mlx ./train -run '^$' \
  -bench '^BenchmarkMamba3SelectiveScanForwardBackward$' -benchtime=100x
go test -tags mlx ./train -run '^$' \
  -bench '^BenchmarkMamba3SelectiveScanForward$' -benchtime=100x
go test -tags mlx ./train -run '^$' \
  -bench '^BenchmarkMamba3CanonicalTrainingStep$' -benchtime=20x
```

## Naming history

The block was originally named `mamba3` for the simplified gated-linear-scan implementation that predated the actual Mamba-3 paper. After the paper landed and a true canonical implementation was added, the simplified block was renamed to `gated_linear_ssm` (with `mamba3` as a deprecated alias emitting a one-time stderr warning) and the paper-aligned block took the explicit name `mamba3-canonical`. A future release will reassign `mamba3` to canonical Mamba-3 once the deprecation period closes.
