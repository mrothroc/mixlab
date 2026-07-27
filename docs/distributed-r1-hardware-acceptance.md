# R1 DDP Hardware Acceptance

This procedure is the manual release gate for fixed-world R1 DDP. It runs the
same opt-in worker on two Metal hosts with the MLX TCP ring and on two CUDA
devices with MLX NCCL. Do not mark the gate complete from singleton or
loopback-ring results.

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

Before launching, ensure macOS permits incoming connections to the test binary
on both hosts. A blocked application-firewall entry presents as one rank
connecting while its peer remains in `accept()` until the startup timeout.

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
on that host, then run:

```bash
MIXLAB_DDP_HARDWARE_ACCEPTANCE=1 \
MIXLAB_DDP_HW_BACKEND=nccl \
mlx.launch --backend nccl --hosts 127.0.0.1 --repeat-hosts 2 -- \
  /tmp/mixlab-ddp-hw.test \
  -test.run '^TestDDPHardwareAcceptanceWorker$' -test.count=1 -test.v
```

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
- The required cross-host Metal run did not pass the release gate. TCP
  reachability succeeded in both directions, but the M4 application firewall
  listed `/private/tmp/mixlab-ddp-hw.test` as blocked. MLX ring initialization
  timed out before step 1. Administrator authentication was unavailable to
  unblock the temporary binary.
- CUDA/NCCL evidence was not run because no CUDA host is configured in this
  environment.

Verdict: worker behavior and loopback ring execution are verified, but the
manual release gate remains incomplete until the two-host Metal run and the
two-GPU CUDA/NCCL run both pass.
