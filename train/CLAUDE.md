# train/ — trainer, optimizer, training-step paths

Houses the `GPUTrainer` Go API plus the in-process training loop. The actual MLX dispatch lives in `gpu/`; this package is the orchestrator.

## Key files
- `gpu_trainer.go` — public `GPUTrainer` API (`TrainStepGPU`, `SubmitStepGPU`, `CollectLossGPU`, etc.)
- `weight_init.go` — RNG-seeded weight init by metadata flags (`InitOne`, `InitLogArange`, `InitDtBias`, etc.)
- `cuda_graph_limits.go` — pre-train hook: inspects the IR and sets `MLX_MAX_OPS_PER_BUFFER`, `MLX_MAX_MB_PER_BUFFER`, and `MLX_CUDA_GRAPH_CACHE_SIZE` env vars before MLX initializes
- `gpu_trainer_mamba3_test.go` — analytical CPU oracle + fused-vs-expanded equivalence tests for canonical Mamba-3
- `gpu_trainer_mamba3_bench_test.go` — benchmark at production-ish shape (D=128, T=1024, G=4)
- `swa_smoke_config_test.go` — config/data coverage checks (skips if `data/example/` is gitignored — common in CI)
- `swa_artifacts_test.go` — default Go tests for SWA/EMA CLI override validation, final/SWA safetensors naming, checkpoint artifact naming, and HF export selection
- `swa_smoke_test.go` — MLX-gated smoke test for live training behavior; run with `go test -tags mlx ./train -run TestSWAWindow128Smoke -count=1` on an MLX-capable machine with `data/example/` available

## Trainer step paths

`IRTrainer::submit_step` (in `gpu/ir_trainer.cpp`) picks one of three paths based on the IR program's op set:

| Path | When | Code |
|---|---|---|
| **Compiled** | Default (no Mamba-3) | `mx::compile(value_and_grad)`; cached per signature |
| **Uncompiled + checkpoint** | Program contains `OP_MAMBA3_SELECTIVE_SCAN` or `OP_MAMBA3_CANONICAL_BLOCK` | Eager `value_and_grad` with `mx::checkpoint` |
| **Low-memory Mamba-3** | Same trigger + low-memory updates active | Per-chunk gradient compute + opt update fusion |

The trainer logs a one-time stderr notice when it falls off the compiled path so you know which mode is active.

## Mamba-3 host-timing instrumentation
For any program with a fused `OP_MAMBA3_CANONICAL_BLOCK`, the trainer logs phase timings every N steps:
```
[mlx_ir] canonical Mamba3 host timing step=N path=<path> prep_us=... grad_us=... opt_us=... eval_us=... total_us=...
```
Enable with `MIXLAB_MAMBA3_HOST_TIMING=1`. Controlled by `MIXLAB_MAMBA3_HOST_TIMING_START` (default 100) and `MIXLAB_MAMBA3_HOST_TIMING_EVERY` (default 100). Disable with `MIXLAB_DISABLE_MAMBA3_HOST_TIMING=1`. See `docs/canonical_mamba3.md` for the full env-var reference.
