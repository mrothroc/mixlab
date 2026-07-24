# train/ — trainer, optimizer, objectives, resume, generation

Houses the `GPUTrainer` Go API plus the in-process training loop, objective
batch preparation, checkpoint/resume, and generation. The actual MLX dispatch
lives in `gpu/`; this package is the orchestrator. The largest package —
`runTrain` in `train.go` is one function, so extract cohesive tails to siblings
(e.g. `train_final_eval.go`, `train_batch.go`, `train_schedule.go`) to stay
under the 1000-line cap.

## Subsystems (entry files)
Each objective's batch prep dispatches from `objective.go::prepareObjectiveBatchWithSeqLen`.

- **Objectives / batching** — `objective.go` (causal/mlm/mntp/hybrid/block-diffusion/multihead/classification masking + label/valid-mask/position construction); `train_batch.go` (`trainBatch` ↔ loader `Batch`).
- **Classification** — `classification.go` (metrics, warm-start), `train_final_eval.go`; head IR is `arch/classification_ir.go`. Uses a `dropoutInactive` program-cache-key variant for dropout-free eval. Guide: [`../docs/config-training.md`](../docs/config-training.md).
- **Resume / checkpoints** — `resume_setup.go`, `resume_checkpoint.go`, `resume_manifest.go`, `gpu_trainer_resume_mlx.go`. Versioned `mixlab_resume_v1` bundle; manifest written last as the commit marker. Keyed dropout (`dropout_rng.go`) makes resumed dropout bit-reproducible.
- **Generation** — `generate.go`, `generate_sampling.go` (replay / batched / TTT-stateful paths + retry-on-incomplete). Grammar constraints: `generate_constraints.go`, `generate_gbnf*.go`, `generate_token_dfa.go` — see [`../docs/grammar-constrained-generation.md`](../docs/grammar-constrained-generation.md). GBNF parses **untrusted** input: keep the recursion-depth and incremental-production bounds.
- **Dataset wiring** — `dataset_manifest.go` (`configureDatasetForTraining`: validate manifest, attach runtime sequence-packing/framing/classification state). Formats live in [`../data/CLAUDE.md`](../data/CLAUDE.md).

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
