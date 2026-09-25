# R1 DDP Hardware Acceptance

This procedure is the manual **DDP-core hardware gate** for fixed-world R1.
It runs the same opt-in worker on two Metal hosts with the MLX TCP ring and on
two CUDA devices with MLX NCCL. Do not mark this gate complete from singleton
or loopback-ring results. The worker pass does not replace the separate
release-candidate `mixlab -mode arch` end-to-end gate.

The [2026-09-25 production executable record](#execution-record-2026-09-25-production-executable)
below completes that separate end-to-end gate on both hardware configurations.

## Build

Use one repository commit and one architecture-compatible test binary on every
rank:

```bash
CGO_ENABLED=1 go test -c -tags mlx -o /tmp/mixlab-ddp-hw.test ./train
```

The worker runs 16 causal microsteps with accumulation `K=2`. It fails on a
non-finite loss, a non-decreasing first-to-last loss, topology disagreement, or
missing distributed telemetry.

## Metal TCP Ring

Copy `/tmp/mixlab-ddp-hw.test` to the same absolute path on both Macs. Create a
hostfile using the hosts' Ethernet addresses:

```json
{
  "backend": "ring",
  "envs": [
    "MIXLAB_DDP_HARDWARE_ACCEPTANCE=1",
    "MIXLAB_DDP_HW_BACKEND=ring"
  ],
  "hosts": [
    {"ssh": "127.0.0.1", "ips": ["M1_ETHERNET_IP"], "rdma": []},
    {"ssh": "M4_SSH_ALIAS", "ips": ["M4_ETHERNET_IP"], "rdma": []}
  ]
}
```

Before launching, the macOS **application firewall must not block the ring's
listening sockets on either host**. The ALF auto-allows only Apple-trusted code
(notarized / Developer ID); it blocks inbound connections to ad-hoc- or
self-signed listeners — which the Go test binary and a Homebrew-built `mixlab`
both are. When it blocks, the ring connections establish and are then killed
mid-handshake (peer sees `ECONNRESET`/`EPIPE`, listener sees `ENOTCONN`),
surfacing as `[ring] Too many send/recv errors` and a `context deadline
exceeded` before step 1. Code-signing does not fix this: a self-signed cert
satisfies `codesign --verify` but not the ALF trust decision, and Homebrew
ad-hoc re-signs binaries on install (arm64 relocation) so a Developer ID
signature would not survive `brew install` anyway. On the trusted, isolated
training LAN, disable the ALF on both hosts for the run:

```bash
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate off
# re-enable afterwards:
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate on
```

The `socketfilterfw --add/--unblockapp` allow-list is unreliable on current
macOS for non-Apple-signed binaries; use the GUI "Allow" prompt or disable the
ALF. Separately, if the M1 runs Little Snitch, set
`sudo littlesnitch write-preference acceptUncheckedDPIName true` so its Deep
Packet Inspection does not close the ring's connect-then-wait flows.

Run from the M1 host:

```bash
mlx.launch --hostfile /tmp/mixlab-metal-ring.json --starting-port 29300 -- \
  env /tmp/mixlab-ddp-hw.test \
  -test.run '^TestDDPHardwareAcceptanceWorker$' -test.count=1 -test.v
```

This administrator-operated R1 primitive assumes a trusted, isolated LAN.

## CUDA NCCL

The R1 CUDA gate uses two visible GPUs in one CUDA host; it validates NCCL
multi-device execution, not multi-node CUDA networking. Build the same commit
on that host (the `train` MLX test package builds on Linux as well as macOS),
then run:

```bash
MIXLAB_DDP_HARDWARE_ACCEPTANCE=1 \
MIXLAB_DDP_HW_BACKEND=nccl \
NCCL_P2P_DISABLE=1 \
mlx.launch --backend nccl --hosts 127.0.0.1 --repeat-hosts 2 -- \
  /tmp/mixlab-ddp-hw.test \
  -test.run '^TestDDPHardwareAcceptanceWorker$' -test.count=1 -test.v
```

`NCCL_P2P_DISABLE=1` is required on hosts where the two GPUs are linked only by
a PCIe bridge (no NVLink) inside a container: with P2P enabled, `ncclCommInitRank`
hangs and the group runtime fails its startup deadline. Disabling P2P forces a
working transport; on NVLink hosts it can be omitted.

When `mlx.launch` (the Python `mlx` package) is not installed — e.g. on the
`golf-mlx-cuda-base` image, which ships only the C++/Go toolchain — launch the
two ranks directly with the NCCL backend's environment contract
(`NCCL_HOST_IP`, `NCCL_PORT`, `MLX_WORLD_SIZE`, `MLX_RANK`, one GPU per rank):

```bash
COMMON="NCCL_HOST_IP=127.0.0.1 NCCL_PORT=29500 MLX_WORLD_SIZE=2 \
  MIXLAB_DDP_HARDWARE_ACCEPTANCE=1 MIXLAB_DDP_HW_BACKEND=nccl NCCL_P2P_DISABLE=1"
T="/tmp/mixlab-ddp-hw.test -test.run ^TestDDPHardwareAcceptanceWorker$ -test.count=1 -test.v"
env $COMMON MLX_RANK=1 CUDA_VISIBLE_DEVICES=1 $T >/tmp/rank1.log 2>&1 &
env $COMMON MLX_RANK=0 CUDA_VISIBLE_DEVICES=0 $T >/tmp/rank0.log 2>&1
```

The first optimizer step spends ~50 s JIT-compiling CUDA kernels for the host's
compute capability (the prebuilt libmlx targets sm_80); the remaining steps run
in single-digit milliseconds. Do not mistake the one-time JIT pause for a hang.

## Required Evidence

Attach the following report to the Weft hardware-acceptance task.

```text
Git commit:
Mixlab/MLX versions:

Metal rank 0 hardware / OS:
Metal rank 1 hardware / OS:
Metal link and measured link rate:
Metal command:
Metal exit status:
Metal first/last loss by rank:
Metal telemetry excerpts:

CUDA GPU / driver:
CUDA MLX version:
CUDA command:
CUDA exit status:
CUDA first/last loss by rank:
CUDA telemetry excerpts:

Verdict:
```

Each telemetry excerpt must show distinct `compute_ms`, `wait_ms`,
`collective_ms`, `all_reduce_ms`, `effective_bandwidth_gb_per_sec`, and
`global_tokens_per_sec` fields, plus microstep/effective-token and bucket
diagnostics.

## Execution Record: 2026-07-27

- Commit under test: `c411246` plus the uncommitted R1 Phase 3 changes.
- Runtime: MLX `0.32.0`, staged identically on both hosts.
- Metal hosts: M4 Max, 64 GB (`Mac16,6`) and M1 Max, 64 GB
  (`MacBookPro18,2`).
- The exact worker passed with two local ring ranks. Rank 0 loss decreased from
  `4.531685` to `1.9450593`; rank 1 decreased from `4.502567` to `1.9095266`.
  Update microsteps reported nonzero collective, wait, all-reduce, bandwidth,
  global-throughput, accumulation, bucket, and effective-token telemetry.
- The required cross-host Metal run did not pass the hardware gate. TCP
  reachability succeeded in both directions, but the M4 application firewall
  listed `/private/tmp/mixlab-ddp-hw.test` as blocked. MLX ring initialization
  timed out before step 1. Administrator authentication was unavailable to
  unblock the temporary binary.
- CUDA/NCCL evidence was not run because no CUDA host is configured in this
  environment.

Verdict: worker behavior and loopback ring execution are verified. The
two-host Metal run subsequently passed (see the continued record below). The
two-GPU CUDA/NCCL run remains outstanding because no CUDA host is configured in
this environment; the DDP-core hardware gate is complete for Metal and pending
only the CUDA/NCCL leg.

## Execution Record: 2026-07-27 (continued) — cross-host Metal PASS

The two-host Metal ring gate **passed**.

- Hosts: rank 0 M1 Max, 64 GB (`MacBookPro18,2`, macOS 26.5.1) on `en7`
  `10.105.5.143`; rank 1 M4 Max, 64 GB (`Mac16,6`, macOS 26.5) on `en7`
  `10.105.5.175`. Both single-homed (Wi-Fi off).
- Link: `1000baseT` full-duplex gigabit Ethernet, measured all-reduce
  effective bandwidth ~0.0275 GB/s at this micro-batch size.
- Command: `mlx.launch --hostfile /tmp/mixlab-metal-ring.json
  --starting-port 29300 -- env /tmp/mixlab-ddp-hw.test -test.run
  '^TestDDPHardwareAcceptanceWorker$' -test.count=1 -test.v`.
- Exit status: `PASS` on both ranks (2.47 s / 2.31 s).
- First/last loss: rank 0 `4.531685 -> 1.945059`; rank 1
  `4.502566 -> 1.909526` (16 update steps each).
- Telemetry (first update microstep): `compute_ms=71.374`,
  `wait_ms=1947.411`, `collective_ms=1947.442`, `all_reduce_ms=21.915`,
  `effective_bandwidth_gb_per_sec=0.0275`, `global_tokens_per_sec=253.6`,
  `microsteps=2`, `effective_global_tokens=512`, `gradient_bytes=602880`,
  `bucket_count=1`, `world_size=2`, `accumulation_steps=2`.

Root cause of the earlier cross-host failures (isolated layer by layer, each
confirmed with plain-socket reproductions):

1. **macOS application firewall on both hosts — the release blocker.** The ALF
   blocks inbound connections to non-Apple-trusted listeners; the ring's
   connections established and were then killed mid-handshake (client
   `ECONNRESET`/`EPIPE`, listener `ENOTCONN`), surfacing as
   `[ring] Too many send/recv errors`. Reproduced deterministically: a
   Homebrew-python listener on the M4 was killed while an Apple-system-python
   listener on the same port passed. `socketfilterfw --add/--unblockapp` and a
   self-signed code-signing cert both failed to clear it (see the firewall note
   above for why). Disabling the ALF on both isolated-LAN hosts
   (`socketfilterfw --setglobalstate off`) resolved it and the gate passed.
2. **Little Snitch Deep Packet Inspection (M1).** The ring connects then waits
   for the first collective without sending data; LS DPI closed such flows
   ("Socket closed during DPI without data"). Fixed with
   `littlesnitch write-preference acceptUncheckedDPIName true`. (A stale
   half-disabled LS extension also caused erratic closures until the M1 was
   rebooted, which flushed it.)

Stock MLX 0.32.0 (`mx.distributed.all_sum` via `mlx.launch --backend ring`)
reproduced the same failures, confirming the blockers were host networking
policy, not the mixlab DDP layer.

## Execution Record: 2026-07-28 — CUDA NCCL PASS

The two-GPU CUDA/NCCL leg **passed** on a RunPod 2× A40 host.

- Host: 2× NVIDIA A40 (48 GB, sm_86), driver 580.159.03, CUDA 12.8, NCCL 2,
  MLX `0.32.0`. Image `golf-mlx-cuda-base:mlx-0.32.0` (Go 1.24.4 + MLX/CUDA/NCCL);
  test binary built on-host from commit `223d3a3`. GPUs linked by PCIe bridge
  (PXB), no NVLink.
- Launch: two ranks, one GPU each, NCCL backend with `NCCL_P2P_DISABLE=1`
  (see the CUDA NCCL section — with P2P enabled `ncclCommInitRank` hung).
- Exit status: `PASS` on both ranks (53.0 s / 54.0 s wall, ~52 s of which was
  the one-time first-step CUDA-kernel JIT).
- First/last loss: rank 0 `4.5316854 -> 1.9450707`; rank 1
  `4.5025673 -> 1.9095302` (16 update steps each).
- Telemetry (a representative update microstep, rank 0): `compute_ms≈16.4`,
  `wait_ms=3.08`, `collective_ms=3.09`, `all_reduce_ms=1.52`,
  `effective_bandwidth_gb_per_sec=0.397`, `global_tokens_per_sec≈26318`,
  `microsteps=16`, `effective_global_tokens=4096`, `gradient_bytes=602880`,
  `bucket_count=1`, `world_size=2`, `accumulation_steps=2`.

Building the `train` MLX test binary on Linux required moving the shared
`generateSyntheticBatch` helper out of the darwin-only integration suite into a
platform-neutral file (`train/synthetic_batch_test.go`, build-tagged
`mlx && cgo && (darwin || linux)`); previously the CUDA build could not compile
the package. This is fixed in the same change as this record.

Verdict: **both R1 DDP-core hardware-acceptance legs pass** — two-host Metal
TCP ring (2026-07-27) and two-GPU CUDA/NCCL (2026-07-28). This primitive gate
is complete; production-executable acceptance remains open.

## Execution Record: 2026-09-25 Production Executable

**PASS on both hardware legs**, using the actual `mixlab -mode arch` path,
not the internal hardware worker. This completes the remaining R1 hardware
gate; commit/tag/deployment remain a separate release ceremony.

Build provenance: base commit `38137aa0a852d5b41c6d2932c82dac38b9f22b96`
plus the R1 closeout worktree. The test source archive (excluding credentials
and `.git`) has SHA-256
`7a5b843ff0f7f8670e3a4686ea12f0e4d47577e240838b93c4600d1221091d0d`.
Later test/documentation-only edits do not change the tested executable.

| Build | Executable SHA-256 |
|---|---|
| macOS arm64 | `245c7d9d042a30d7d784b1f947d7ccdb5913d6af4a9e46882d1ebe2fce5ef1f1` |
| Linux amd64 CUDA | `dba1dee58750d648e58d7ee61fc156441c6d24463f57f1024a6013aa2e441e98` |

Recipe: `examples/distributed_causal.json`, D=64, V=64, T=8, local batch=128,
K=2, world=2, AdamW, dropout=0.1, eight optimizer attempts. Synthetic train and
validation shards each contain 8,193 tokens, `1 + i % 60`. Dataset content ID:
`9fc1ed337e1dae7acdc53b91c86323c39afd80caaebbd1651be11f9405b49d61`.
Each run commits eight updates, consumes sixteen local microbatches / 4,096
global valid targets, validates, and publishes checkpoints at attempts 4 and 8.
Restart resumes attempt 4 with the same membership and a new launch-attempt ID.

### Metal Executable

M1 Max and M4 Max, Ethernet addresses `10.105.5.143` and `10.105.5.173`.
MLX 0.32.1 on both hosts. Hostfile entries use SSH `127.0.0.1` and `m4`, respectively.
The native binary and identical synthetic data/config were staged at
`/tmp/mixlab-r1-closeout` on both machines.

```bash
mlx.launch --hostfile /tmp/mixlab-r1-closeout/metal-hosts.json \
  --starting-port 29450 -- env /tmp/mixlab-r1-closeout/mixlab \
  -mode arch -config /tmp/mixlab-r1-closeout/distributed_causal.json \
  -train '/tmp/mixlab-r1-closeout/train_*.bin' \
  -checkpoint-dir /tmp/mixlab-r1-closeout/metal-final-checkpoints \
  -checkpoint-every 4 -safetensors /tmp/mixlab-r1-closeout/metal-final-v2.safetensors \
  -telemetry-out /tmp/mixlab-r1-closeout/metal-final-v2.jsonl \
  -log-every 1 -val-every 2
```

Both ranks completed. First/last loss `4.357527 -> 3.891680`; validation loss
`3.5439`. Representative final-step telemetry: compute `18.33 ms`, wait
`21.98 ms`, collective `22.00 ms`, all-reduce `18.59 ms`, effective bandwidth
`0.032 GB/s`, global throughput `12,702 tokens/s`, gradient bytes `602,880`.
These are correctness-test measurements, not a speedup claim.

The restart command used port 29430, copied the complete midpoint bundle to the
M4, added `-resume .../metal-checkpoints/step_000004.distributed.resume.json`,
and selected separate `metal-resumed` outputs. Original run/restart maximum
weight difference was `1.49e-8`, optimizer-state difference `1.40e-9`.
The final fresh executable run also matched that restart within `6.71e-7`
for weights and `2.57e-9` for moments (tolerance `1e-6`).

Initial startup timed out with both application firewalls enabled. The owner
added and unblocked the exact staged executable on both hosts using
`socketfilterfw --add` / `--unblockapp`; the next run passed. **Neither firewall
was disabled.** This supersedes the earlier host-specific conclusion that the
command-line exception could not suffice. Recheck exceptions after replacing
the executable; R1.1's signed packaging remains separate work.

### CUDA Executable

Temporary RunPod host: two NVIDIA A40 GPUs (sm_86), driver `580.159.04`,
MLX `0.32.0`, image `michaelrothrock/mixlab:runpod`. The current source was built
inside the image; `libfmt-dev` was installed for the MLX development headers.
No real user training data or credentials were included in the source upload.

The bounded test driver launched two copies of `/tmp/mixlab-r1`, one per GPU,
with `MLX_WORLD_SIZE=2`, `MLX_RANK=0/1`, `CUDA_VISIBLE_DEVICES=0/1`,
`NCCL_HOST_IP=127.0.0.1`, `NCCL_PORT=29472`, and `NCCL_P2P_DISABLE=1`.
Both used the recipe above with backend `nccl`, identical `-mode arch` flags,
checkpoint interval 4, log interval 1, and validation interval 2. The driver
checked both process exit codes, killed peers on failure/timeout, then relaunched
from the attempt-4 manifest into a separate artifact directory.

Both fresh and resumed ranks exited zero. First/last loss
`4.357527 -> 3.891700`, validation loss `3.5439`. Final fresh step: compute
`28.10 ms`, wait `8.26 ms`, collective `8.28 ms`, all-reduce `1.24 ms`, effective
bandwidth `0.485 GB/s`, global throughput `14,083 tokens/s`.
Maximum final weight difference `8.94e-8`; optimizer-state difference `5.00e-8`.
JSONL assertions also verified lifetime optimizer-attempt, microstep, and
valid-token counters after resume. Non-root logs contained no training progress.

A separate negative rerun deliberately encountered an existing checkpoint:
rank zero refused to overwrite it and rank one reported publication failure;
both failed instead of continuing with an inconsistent checkpoint.

### Regression Evidence

- `go test ./... -count=1`: pass, using the repository Python environment.
- `make lint`: pass, zero issues.
- Focused MLX `gpu` / `train` distributed suite: pass, including the real CLI
  gate for flat and record datasets, rank-zero early stop, copied-path resume,
  K=4 global-batch parity, unequal normalization, one-rank NaN/global skip,
  initialization mismatch, and changed-member rejection.
- A one-rank strict MLX ring fails closed within the 30-second startup bound
  on this MLX version, before loading data. No single-process fallback occurs.
- Tests require native child `PASS` evidence, not only `mlx.launch`'s exit code;
  the installed launcher can return zero after a failed child.

Synthetic hardware artifacts were retained outside the repository under
`/tmp/mixlab-r1-closeout`; the CUDA archive is `cuda-ddp-evidence.tar.gz`.
The temporary RunPod pod was terminated after evidence retrieval; a follow-up
API query confirmed that no pods remained in the account.
