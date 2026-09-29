# When Distributed Training Helps

Distributed data parallelism (DDP) runs a full copy of the model on every
worker, splits each batch between them, and averages gradients after every
optimizer step. It adds machines' throughput only when that averaging is cheap
compared with the work done between averages.

## Short answer

| Setup | What to expect |
|-------|----------------|
| One Mac or one GPU | The default, and usually the fastest per token. |
| Several Macs on a LAN (managed cluster or `mlx.launch`) | Usually **slower** than the fastest Mac alone. Useful for a larger global batch or for the multi-machine workflow itself; faster only with heavy gradient accumulation, and never faster than the machines' combined single-machine speed. |
| Several GPUs in one CUDA machine (NCCL) | The interconnect is far faster than a LAN, so gradient averaging costs much less. Not measured here; measure your own config. |
| A model too large for one machine | DDP does not help: every worker holds the whole model and optimizer state. |

## Measured on two Macs

Two workers on an M1 Max and an M4 Max over wired gigabit Ethernet, using the
managed cluster's encrypted transport, with the same `batch_tokens` per worker.
Recorded 2026-09-29 with v0.119.0. Single-Mac figures are the steady-state
`tok/s` that `mixlab -mode arch` logs; two-Mac figures come from the difference
between 10-step and 40-step runs, which cancels startup cost. The M4 was a
shared machine under other load, and slower than the M1.

| Config | M1 alone | M4 alone | Both Macs | Both vs M1 alone |
|--------|----------|----------|-----------|------------------|
| 0.5M parameters, 1,024 tokens per worker | 78,000 tok/s | — | 9,900 tok/s | 8× slower |
| 19M parameters, 8,192 tokens per worker | 34,600 tok/s | 27,400 tok/s | 5,800 tok/s | 6× slower |
| 19M parameters, accumulation 8 | 34,600 tok/s | 27,400 tok/s | 27,600 tok/s | 1.25× slower |

## Why

Each optimizer step waits for the slowest worker's compute and then for the
gradient average. On this pair the average cost about **0.2 s plus the gradient
size at about 30 MB/s** (fp32 gradients: 4 bytes per parameter). Encryption and
relaying keep this well below gigabit's raw ~110 MB/s.

A useful estimate for two workers:

```
time per optimizer step ≈ k × t_micro(slowest worker) + t_sync
two-worker throughput   ≈ 2 × k × batch_tokens / time per step
one-machine throughput  = batch_tokens / t_micro(fastest machine)
```

where `k` is `gradient_accumulation_steps`. For the 19M-parameter model,
`t_sync` is about 2.5 s against 0.24–0.30 s of compute per microbatch. That pair
breaks even with the M1 alone at about 15 accumulation steps, and can never be
more than about 1.6× faster, because the slower M4 paces every step.

So distributed training on a LAN pays off when each optimizer step carries many
seconds of compute: larger models per microbatch, larger `batch_tokens`, or
more accumulation. Accumulation changes the optimization (a larger global batch
is `batch_tokens × gradient_accumulation_steps × workers`), so tune the
learning rate for it rather than treating it as free speed. Use wired Ethernet;
Wi-Fi is slower and less stable.

## Measure your own config

1. On your fastest machine, run the config without `training.distributed` and
   read the steady-state `tok/s` from the training log.
2. Submit the distributed config twice, for example with 10 and 40 steps, timing
   each `submit`. Throughput is
   `(40 − 10) × batch_tokens × gradient_accumulation_steps × workers / (T40 − T10)`.
3. Compare, then adjust `batch_tokens` and `gradient_accumulation_steps` and
   repeat.

To set up the machines, see [managed clusters](cluster-quickstart.md) or the
[unmanaged workflow](distributed-training.md).
