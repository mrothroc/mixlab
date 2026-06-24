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

## Step timing

Add `-timing` to see where each progress interval spends time:

```bash
./mixlab -mode arch -config model.json -train 'data/*.bin' -timing
```

Example output:

```text
[model] [timing] data=1.2ms gpu=142.5ms val=11.3ms log=0.2ms
```

Fields:

| Field | Meaning |
|-------|---------|
| `data` | Time waiting for the next batch. Should be near zero if prefetch keeps up. |
| `gpu` | Forward, backward, and optimizer time. |
| `val` | Validation loss time. Zero on non-validation steps. |
| `log` | Progress formatting time. |

If `data` is consistently high, the loader cannot keep up with the GPU. If
`gpu` dominates and `data` is near zero, the training pipeline is healthy.

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
