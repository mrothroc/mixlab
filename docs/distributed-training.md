# Distributed (DDP) training — current state and contributor guide

Later releases add a cluster manager, enrollment, LAN recruitment, and DiLoCo.
Those are not part of this release and are not documented here.

This guide covers explicitly launched fixed-world data-parallel training and
the lower-level contributor tests. Host recruitment and managed launch remain
R1.1 work. R1 assumes administrator-provisioned hosts and a trusted transport.

## Status

`training.distributed` now selects DDP through the ordinary `-mode arch`
entrypoint. There is no separate DDP mode or Mixlab hostfile flag: `mlx.launch`
(or an equivalent administrator-operated launcher) supplies MLX rendezvous.
`-mode smoke` only probes backend availability; it does not train.

The production path includes rank-disjoint data, accumulated optimizer attempts,
rank-zero control, and direct checkpoint resume. Release acceptance is tracked
separately in the [hardware record](distributed-r1-hardware-acceptance.md).
Earlier core-worker passes do not substitute for the executable gates.

The 2026-09-25 executable gates passed on M1/M4 Metal and two A40 CUDA GPUs,
including midpoint checkpoint restart. These are acceptance results for the
closeout worktree, not a claim that a new release has been published.

## Launch The Trainer

Start with [`examples/distributed_causal.json`](../examples/distributed_causal.json).
Use identical config and shard contents on every rank; local dataset directories
may differ. Each host needs a compatible native MLX build.

```bash
mlx.launch --hostfile hosts.json -- \
  env /path/to/mixlab -mode arch -config model.json -train 'data/train_*.bin' \
  -checkpoint-dir checkpoints -checkpoint-every 100 -safetensors final.safetensors
```

The `env` command is intentional: it prevents launcher versions which default
to Python from treating the native executable as a Python script. See the
[hardware procedure](distributed-r1-hardware-acceptance.md) for ring hostfiles
and the equivalent two-GPU NCCL environment. Some MLX launcher versions return
zero even when a child fails; inspect every rank's status and the expected
checkpoint/summary, not just the launcher's exit code.

`batch_tokens` is the **local microbatch** size. Global batch size is
`batch_tokens * gradient_accumulation_steps * world_size`. `steps`, warmup, and
checkpoint cadence count optimizer attempts, including globally skipped updates.
Learning rates are not automatically multiplied by world size. Lookahead is off.

Resume with the same launch and `-resume checkpoints` (or a specific distributed
resume manifest). Copy the complete checkpoint bundle to each host first.
Exact resume checks ordered host/rank identity, backend, model/program, optimizer,
batch/accumulation topology, and content-stable dataset identity. The sampler
cursor is restored directly; no earlier batches are replayed. A new launch gets
a new attempt ID, not a new membership. `-safetensors-load` is weights-only warm
start, not exact resume. Only rank zero publishes artifacts or opens telemetry.

Continuous token shards partition aligned shuffle chunks; sequence shards with
`one_record_per_row` partition whole records, reuse BOS/EOS/PAD framing, and
normalize by valid target count. Packed-segment and classification datasets are
not exposed. Shards must remain immutable. Short shard tails and fewer-than-world
remaining samples are dropped deterministically each epoch. Dataset identity
uses the canonical manifest, logical shard basenames, sizes, and content hashes;
paths and modification times are excluded.

## Backends

| Backend | Platform | Transport | Selected when |
|---------|----------|-----------|---------------|
| `ring`  | macOS / Metal | TCP over Ethernet (MLX ring) | `runtime.GOOS != linux` |
| `nccl`  | Linux / CUDA  | NCCL (single host, multi-GPU) | `runtime.GOOS == linux` |

`gpu.RequireDistributedBackend` (`gpu/runtime_capabilities.go`) accepts only
these two strings. `-mode smoke` (`train/smoke.go`) requires `ring` on macOS and
`nccl` on Linux as a readiness check.

## Architecture — where the code lives

The rank/world identity is **established by MLX from the launcher environment**,
not by any mixlab Go code. mixlab only reads it back.

- **Group runtime**: `gpu.NewGroupRuntime(ctx, view)` (strict) /
  `gpu.NewSingletonGroupRuntime` (non-strict Phase-1 fallback) —
  `gpu/group_runtime.go`. The C++ side calls
  `mx::distributed::init(strict, backend)` (`gpu/group_runtime.cpp`), which
  consumes the launcher-provided environment (ring hostfile, or the NCCL env
  contract). Rank/world are read back via `mlx_group_runtime_rank` /
  `mlx_group_runtime_world_size`.
- **Backend string** comes from `view.Membership.Backend` (`gpu/group_runtime.go`).
- **DDP trainer**: `initGPUTrainerWithDistributedContext`
  (`train/distributed_trainer.go`) takes a `DistributedTrainerContext`
  (`train/distributed_context.go`): `GroupRuntime`, `LocalView`,
  `GradientBucketBytes`, `AccumulationSteps`, `DatasetHash`, `ScheduledPhase`.
- **Production bootstrap**: `gpu.BootstrapGroupRuntime` strictly initializes MLX,
  reads authoritative rank/world, agrees ordered member descriptors, and verifies
  fresh/resumed membership before building the trainer or sampler. Size one is
  rejected. `train/distributed_train_mlx.go` orchestrates optimizer attempts.
- **JSON config**: `training.distributed` selects mode, backend, accumulation,
  and gradient bucketing. Prepared batches provide the loss normalizer
  (`gpu/group_runtime_mlx.go`: `mlxTrainerSetDistributedOptions`,
  `mlxTrainerSetNextLossNormalizer`). The one public knob is per-rank
  `batch_tokens`; the global batch is
  `batch_tokens × world_size × accumulation_steps`, computed in the trainer, not
  declared in config.

### R1 constraints

Public config validation (`arch/distributed.go`) intentionally exposes less than
the internal numerical test envelope:

- objective must be **causal** (MLM/MNTP remain internal parity tests);
- optimizer must be **adamw**;
- **no** `seq_len_schedule`;
- **no** distillation / data2vec / MTP / first-byte-mask / example-framing /
  attention-segment-mask auxiliary losses;
- no BatchNorm mutable buffers, canonical Mamba-3, MoE/custom blocks, recurrence,
  dynamic shapes, QAT, SWA, TTT validation updates, or `arch_race`.

## Run The Core Tests

Everything below is the **test-binary + `mlx.launch`** flow. `mlx.launch` is the
console script from the Python `mlx` package; it SSHes to each host, assigns
ranks, and sets the launcher environment. It does **not** move ring data.

### 1. Build the acceptance test binary (macOS or Linux)

```bash
CGO_ENABLED=1 go test -c -tags mlx -o /tmp/mixlab-ddp-hw.test ./train
```

The `train` MLX test package builds on **both** darwin and linux. (It didn't
until 2026-07: the shared helper `generateSyntheticBatch` was stranded in the
darwin-only `integration_test.go`, so the Linux/CUDA build failed to compile the
package. It now lives in `train/synthetic_batch_test.go`, build-tagged
`mlx && cgo && (darwin || linux)`. If you add a shared test helper, do **not**
gate it `darwin`-only.)

### 2. Check backend readiness

```bash
mixlab -mode smoke   # PASS: MLX distributed backend available: ring|nccl
```

### 3. Metal ring (macOS multi-host) and CUDA NCCL (Linux 2-GPU)

The exact commands, the ring hostfile JSON (`backend` / `envs[]` /
`hosts[].{ssh,ips,rdma}`), the NCCL environment contract, and full worked
evidence live in the release-gate doc:
[`distributed-r1-hardware-acceptance.md`](distributed-r1-hardware-acceptance.md).
Read it before running — it is the canonical procedure.

## Operational gotchas — read before repeating our discovery

These cost significant time to diagnose. They are host/environment issues, not
mixlab bugs (the R1 DDP code is validated by the loopback and hardware runs).

### macOS: the application firewall silently kills the ring

The macOS Application Firewall (ALF) auto-allows inbound connections **only** for
Apple-trusted code (notarized / Developer ID). It **blocks** inbound to
ad-hoc- or self-signed listeners — which the Go test binary and a Homebrew-built
`mixlab` both are. Symptom: ring connections establish, then die mid-handshake
(`ECONNRESET`/`EPIPE` on the peer, `ENOTCONN` on the listener), surfacing as
`[ring] Too many send/recv errors` and `context deadline exceeded` before step 1.

The previous acceptance run found self-signing and command-line app exceptions
insufficient on its particular hosts. Do not automatically disable host security.
Have an administrator allow the exact executable's inbound connections on every
Mac and verify the exception after rebuilding or replacing it. R1.1's signed
distribution and separate cluster-agent plan addresses the packaging problem.
Full historical detail: the "Metal TCP Ring" section of
[`distributed-r1-hardware-acceptance.md`](distributed-r1-hardware-acceptance.md).

### macOS: Little Snitch DPI closes connect-then-wait flows

If a host runs Little Snitch, its Deep Packet Inspection closes the ring's
connect-then-wait sockets ("Socket closed during DPI without data"). Fix:

```bash
sudo littlesnitch write-preference acceptUncheckedDPIName true
```

A stale/half-disabled Little Snitch network extension can keep closing flows even
after the filter is toggled off; a reboot flushes it.

### CUDA: the right image is in the private registry, not Docker Hub

Only the private Artifact Registry images tagged `mlx-0.32.0`
(`golf-mlx-cuda-base:mlx-0.32.0` = Go + MLX 0.32.0 + CUDA + NCCL + nvcc, and no
`mixlab` entrypoint — use this to build) carry the MLX commit current `main`
requires. The Docker Hub `mixlab-cuda:latest`, `mixlab:latest`, and
`mixlab:runpod` tags are **stale MLX** (an older `get_jit_module(core::Device)`
signature), so the CUDA build fails with a `cu::Device` vs `core::Device` type
error. See `docker/README.md` and `docker/base.Dockerfile` (`MLX_COMMIT`).

### CUDA: NCCL P2P hangs on PCIe-only multi-GPU hosts

On hosts where the two GPUs are linked only by a PCIe bridge (no NVLink), inside
a container, `ncclCommInitRank` **hangs** with P2P enabled — the group runtime
then fails its 45 s startup deadline. Set `NCCL_P2P_DISABLE=1`. Also expect the
**first optimizer step to spend ~50 s JIT-compiling** CUDA kernels (the prebuilt
libmlx targets sm_80); don't mistake it for a hang.

### Launcher availability

`mlx.launch` ships in the Python `mlx` package, which is **absent** from the
`golf-mlx-cuda-base` image (C++/Go toolchain only). When it's missing, launch the
ranks directly with the NCCL env contract (`NCCL_HOST_IP`, `NCCL_PORT`,
`MLX_WORLD_SIZE`, `MLX_RANK`, one `CUDA_VISIBLE_DEVICES` per rank) — the exact
form is in the CUDA NCCL section of the acceptance doc.

## Executable Regression Gate

```bash
CGO_ENABLED=1 go build -tags mlx -o /tmp/mixlab-ddp ./cmd/mixlab
MIXLAB_DDP_CLI=/tmp/mixlab-ddp go test -tags mlx ./train \
  -run TestDistributedCLITrainingAndResume -count=1 -v
```

On Metal this launches two local native processes and checks fresh training,
rank-zero validation/early stop, flat/record data, copied-path checkpoint resume,
weights and moments, telemetry counters, and incompatible-accumulation rejection.
It is a regression gate, not a replacement for two physical Macs.

## Reference

| Thing | Value / location |
|-------|------------------|
| Training mode (single-process) | `-mode arch` (`cmd/mixlab/main.go`) |
| Backend readiness probe | `-mode smoke` → `RequireDistributedBackend` (`train/smoke.go`) |
| DDP selection | `training.distributed`, ordinary `-mode arch` |
| Backends | `ring` (macOS), `nccl` (Linux) (`gpu/runtime_capabilities.go`) |
| Launcher | `mlx.launch` (Python `mlx` package) |
| MLX init | `mx::distributed::init(strict, backend)` (`gpu/group_runtime.cpp`) |
| Go runtime entry | `gpu.NewGroupRuntime` / `NewSingletonGroupRuntime` (`gpu/group_runtime.go`) |
| DDP trainer entry | `runDistributedTrain` -> `initGPUTrainerWithDistributedContext` |
| Distributed context | `DistributedTrainerContext` (`train/distributed_context.go`) |
| Launcher/MLX env vars | `MLX_RANK`, `MLX_WORLD_SIZE`, `NCCL_HOST_IP`, `NCCL_PORT`, `NCCL_P2P_DISABLE`, `CUDA_VISIBLE_DEVICES` |
| Per-rank batch (public) | `batch_tokens` (`docs/config-training.md`) |
| Release gate / worked runs | [`distributed-r1-hardware-acceptance.md`](distributed-r1-hardware-acceptance.md) |
