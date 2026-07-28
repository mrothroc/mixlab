# Distributed (DDP) training — current state and contributor guide

This is a **contributor/maintainer** document, not a user guide. It records how
data-parallel (DDP) training works in mixlab **today** and the operational
traps we already paid for, so contributors don't repeat the discovery. The
user-facing "train your model across N machines" CLI does **not** exist yet
(see [Status](#status)); when it lands, this doc's user-facing successor should
supersede the "How to run it today" section.

## Status

DDP is an **R1 walking-skeleton primitive**. It is **not exposed through the
`mixlab` CLI** — there is no `-mode`, `-distributed`, `-hostfile`, or `-backend`
flag, and `-mode arch` always constructs the trainer with a `nil` distributed
context (`train/gpu_trainer_mlx.go`). The DDP trainer is reachable **only from
Go test entrypoints launched under `mlx.launch`**. `-mode smoke` merely *probes*
that a backend is available; it does not train.

CLI wiring (a `-mode arch` path that builds a `DistributedTrainerContext` from a
hostfile/backend selection instead of `nil`) is planned for a later release.
Until then, the only runnable distributed job is the R1 hardware-acceptance
worker — see [`distributed-r1-hardware-acceptance.md`](distributed-r1-hardware-acceptance.md).

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
  **Only test files build a non-nil context.**
- **No JSON config fields** exist for `accumulation_steps`, gradient bucketing,
  or the loss normalizer — they are set programmatically
  (`gpu/group_runtime_mlx.go`: `mlxTrainerSetDistributedOptions`,
  `mlxTrainerSetNextLossNormalizer`). The one public knob is per-rank
  `batch_tokens`; the global batch is
  `batch_tokens × world_size × accumulation_steps`, computed in the trainer, not
  declared in config.

### R1 constraints

The DDP path validates (`train/distributed_trainer.go`) and currently rejects
anything outside the walking-skeleton envelope:

- objective must be **causal, mlm, or mntp**;
- optimizer must be **adamw**;
- **no** `seq_len_schedule`;
- **no** distillation / data2vec / MTP / first-byte-mask / example-framing /
  attention-segment-mask auxiliary losses.

## How to run it today

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

**Code-signing does not fix this** — a self-signed cert satisfies
`codesign --verify` but not the ALF trust decision, and Homebrew ad-hoc re-signs
binaries on install (arm64 relocation) so a Developer ID signature would not even
survive `brew install`. On a trusted, isolated training LAN, disable the ALF on
**every** host for the run:

```bash
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate off   # re-enable with 'on' after
```

`socketfilterfw --add/--unblockapp` is unreliable on current macOS for
non-Apple-signed binaries; the alternatives are the GUI "Allow" prompt (keeps the
firewall on) or disabling the ALF. Full detail: the "Metal TCP Ring" section of
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

## Where the CLI wiring will go

When DDP is exposed through the CLI, the extension point is `-mode arch`: build a
`DistributedTrainerContext` from a hostfile/backend selection and pass it to
`initGPUTrainerWithDistributedContext` instead of the current `nil`. The trainer,
group runtime, gradient bucketing, accumulation, and weighted loss reduction
already exist and are exercised by `train/distributed_*_mlx_test.go`; the missing
piece is CLI surface + config plumbing, within the R1 constraints above.

## Reference

| Thing | Value / location |
|-------|------------------|
| Training mode (single-process) | `-mode arch` (`cmd/mixlab/main.go`) |
| Backend readiness probe | `-mode smoke` → `RequireDistributedBackend` (`train/smoke.go`) |
| DDP CLI flags | none exist yet |
| Backends | `ring` (macOS), `nccl` (Linux) (`gpu/runtime_capabilities.go`) |
| Launcher | `mlx.launch` (Python `mlx` package) |
| MLX init | `mx::distributed::init(strict, backend)` (`gpu/group_runtime.cpp`) |
| Go runtime entry | `gpu.NewGroupRuntime` / `NewSingletonGroupRuntime` (`gpu/group_runtime.go`) |
| DDP trainer entry | `initGPUTrainerWithDistributedContext` (`train/distributed_trainer.go`) — test callers only |
| Distributed context | `DistributedTrainerContext` (`train/distributed_context.go`) |
| Launcher/MLX env vars | `MLX_RANK`, `MLX_WORLD_SIZE`, `NCCL_HOST_IP`, `NCCL_PORT`, `NCCL_P2P_DISABLE`, `CUDA_VISIBLE_DEVICES` |
| Per-rank batch (public) | `batch_tokens` (`docs/config-training.md`) |
| Release gate / worked runs | [`distributed-r1-hardware-acceptance.md`](distributed-r1-hardware-acceptance.md) |
