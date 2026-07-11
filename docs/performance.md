# Performance and profiling

mixlab is optimized for quick architecture iteration. Small configs train in
seconds on Apple Silicon, and the same configs can run on CUDA through Docker.

## Baseline expectation

On Apple M1 Max with Metal, a `d=1024`, `seq_len=1024` model runs at roughly
8.5 seconds per 100 training steps. Smaller models such as `d=128` finish in
seconds.

Use `mixlab -mode count -config model.json` before longer runs to estimate
parameters, FLOPs, model size, and IR op counts.

## MLX graph tuning

MLX can default to small graph batches on some GPUs. mixlab auto-tunes this
from the model's IR op count, with a larger floor for `gated_deltanet` because
its compact IR scan op expands into many MLX ops. This typically gives about a
10% speedup on GPUs like the A40.

If GPU utilization is still low, override manually:

```bash
MLX_MAX_OPS_PER_BUFFER=2000 MLX_MAX_MB_PER_BUFFER=4000 ./mixlab -mode arch ...
```

Higher values batch more kernels into each graph, reducing dispatch overhead at
the cost of more GPU memory.

## MLX memory bounds

mixlab sets conservative MLX memory limits at training startup so the MLX
buffer cache cannot silently grow into OS swap on unified-memory systems. The
default uses total RAM, reserves at least 25%, applies an 8 GiB floor capped
for small machines, and sets a smaller cache cap.

Override with:

```bash
MIXLAB_MLX_MEMORY_LIMIT_MB=32768 MIXLAB_MLX_CACHE_LIMIT_MB=8192 ./mixlab -mode arch ...
```

Disable the limits:

```bash
MIXLAB_DISABLE_MLX_MEMORY_LIMITS=1 ./mixlab -mode arch ...
```

Diagnostics:

```bash
MIXLAB_MLX_MEM_LOG_EVERY=100 MIXLAB_MLX_CLEAR_CACHE_EVERY=500 ./mixlab -mode arch ...
```

`MIXLAB_MLX_MEM_LOG_EVERY` prints the compact telemetry line at the requested
cadence, including MLX memory and best-effort GPU utilization when available.

## Step timing

Add `-timing` to see where each progress interval spends time:

```bash
./mixlab -mode arch -config model.json -train 'data/*.bin' -timing
```

Example output:

```text
[model] [timing] data=1.2ms gpu=142.5ms val=11.3ms log=0.2ms compile=train_hits=24 train_misses=1 sampler_hits=24 sampler_misses=1
```

Fields:

| Field | Meaning |
|-------|---------|
| `data` | Time waiting for the next batch. Should be near zero if prefetch keeps up. |
| `gpu` | Forward, backward, and optimizer time. |
| `val` | Validation loss time. Zero on non-validation steps. |
| `log` | Progress formatting time. |
| `compile` | MLX compiled-graph cache hits and misses when the backend exposes counters. |

If `data` is consistently high, the loader cannot keep up with the GPU. If
`gpu` dominates and `data` is near zero, the training pipeline is healthy.

For RTD/ELECTRA runs, Mixlab keeps replacement sampling vectorized on device
and compiles the generator sampler by default. After the first sampler miss,
steady-state RTD timing should show sampler cache hits. To force the older
eager sampler path for debugging:

```bash
MIXLAB_RTD_EAGER_GENERATOR_SAMPLER=1 ./mixlab -mode arch ...
```

To inspect MLX compiled-graph cache behavior, enable:

```bash
MIXLAB_MLX_COMPILE_LOG=1 ./mixlab -mode arch ...
```

`-timing` gives compact cumulative hit/miss counters at the log cadence.
`MIXLAB_MLX_COMPILE_LOG=1` prints one line per cache event for the main
compiled training step and the categorical sampler.

## Go profiling

mixlab uses standard Go profiling. No extra tooling is needed when profiling is
disabled.

```bash
# CPU profile
./mixlab -mode arch -config my_model.json -train 'data/*.bin' -cpuprofile cpu.prof
go tool pprof -http :8080 cpu.prof

# Memory profile
./mixlab -mode arch -config my_model.json -train 'data/*.bin' -memprofile mem.prof
go tool pprof mem.prof
```

Both flags are safe for real training runs. The output is a standard pprof
file that works with `go tool pprof`, Speedscope, and pprof-compatible viewers.

For live profiling, start the debug server:

```bash
./mixlab -mode arch -config my_model.json -train 'data/*.bin' -pprof-addr 127.0.0.1:6060
```

Then use standard pprof tooling for Go profiles:

```bash
go tool pprof http://127.0.0.1:6060/debug/pprof/profile?seconds=30
go tool pprof http://127.0.0.1:6060/debug/pprof/heap
```

The same server exposes Mixlab runtime telemetry as gauges rather than pprof
profiles:

```bash
curl http://127.0.0.1:6060/debug/mixlab/telemetry | jq .
curl http://127.0.0.1:6060/debug/vars | jq .
```

Telemetry includes current step, loss, learning rate, objective, sequence
length, steady-state tokens/sec, MLX active/cache/peak memory, host RSS, and
best-effort GPU utilization on macOS from `ioreg`. When a training graph
declares active scalar auxiliary losses, it also includes a
`component_losses` object, for example `invariance_loss`, `pll_margin_loss`,
`word_struct_loss`, or `moe_aux_loss`; absent/no-op objectives do not add keys.
Optimizer telemetry includes committed `optimizer_steps`, cumulative
`skipped_optimizer_steps`, and `optimizer_step_skipped` for the current batch.
Mixlab validates the complete persistent optimizer transaction after each
step. If the loss, any gradient, candidate weight, or optimizer moment is
non-finite, the candidate update is discarded atomically and the previous
weights and optimizer state remain active. GPU utilization is omitted when the
platform does not expose a no-sudo sampler.

To keep a time series for plotting, add:

```bash
./mixlab -mode arch -config my_model.json -train 'data/*.bin' -telemetry-out run.telemetry.jsonl
```

The debug server is opt-in and should normally bind to localhost. For remote
runs, prefer SSH tunneling:

```bash
ssh -L 6060:127.0.0.1:6060 user@host
```

Only bind `-pprof-addr 0.0.0.0:6060` on a trusted network, because pprof and
debug endpoints expose process internals.

## Remote GPU profiling

For RunPod or cloud jobs, generate a signed upload URL, pass the training
command in setup, and upload the profile after the run.

```bash
# Generate a signed URL
gcloud storage sign-url gs://your-bucket/profiles/cpu.prof \
    --http-verb=PUT --duration=1h \
    --impersonate-service-account=your-sa@project.iam.gserviceaccount.com

# Submit RunPod job with profiling
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT/run \
    -H 'Authorization: Bearer YOUR_API_KEY' \
    -d '{
  "input": {
    "setup": [
      "mixlab -mode arch -config /examples/plain_3L.json -train /data/*.bin -cpuprofile /tmp/cpu.prof"
    ],
    "mode": "smoke",
    "post": [
      "curl -X PUT -H '"'"'Content-Type: application/octet-stream'"'"' --data-binary @/tmp/cpu.prof '"'"'SIGNED_URL'"'"'"
    ]
  }
}'

# Download and view locally
gcloud storage cp gs://your-bucket/profiles/cpu.prof .
go tool pprof -http :8080 cpu.prof
```

## Troubleshooting MLX detection

The Makefile auto-detects MLX through:

```bash
python3 -c "import mlx; ..."
```

Run this to inspect the detected path:

```bash
make check-mlx
```

If detection fails because MLX is installed in a virtualenv or another Python:

```bash
make build MLX_PREFIX=$(python3.12 -c "import mlx, os; print(os.path.dirname(mlx.__file__))")

export MLX_PREFIX=/opt/homebrew/lib/python3.12/site-packages/mlx
make build
```
