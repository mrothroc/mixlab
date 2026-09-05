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

mixlab sets conservative MLX memory limits at training and causal-generation
startup. On Metal's unified-memory systems, the default uses total RAM,
reserves at least 25%, applies an 8 GiB floor capped for small machines, and
sets a smaller cache cap. On CUDA, the default is derived from the selected
GPU's VRAM and current free VRAM, never from host RAM. Mixlab retains MLX's
existing device limit when that limit is valid for the available VRAM;
otherwise it falls back to 75% of available VRAM. The automatic CUDA cache
limit starts at one eighth of available VRAM.

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
The startup memory-limit line reports the selected device, total VRAM, free
VRAM when available, configured limits, and host RAM. Explicit
`MIXLAB_MLX_MEMORY_LIMIT_MB` and `MIXLAB_MLX_CACHE_LIMIT_MB` values are passed
to MLX on both Metal and CUDA and are labeled `env` in that line. If a CUDA
host cannot report device memory, mixlab keeps MLX's own limits rather than
sizing one from host RAM, and says so on that line.

CUDA allocation failures include the native allocator error, configured
memory and cache limits, and the submitted batch shape. A memory limit is an
allocator guideline, not a guarantee that an oversized workload will fit;
reduce `batch_tokens` or sequence length if a correctly bounded run still
exhausts VRAM.

FFT-heavy training paths can need additional temporary headroom. In particular,
S4D `sobolev_filter` may retain frequency-domain intermediates for backward and
can raise peak memory by several GiB even for a small parameter count. If the
reported peak is flat after warmup but a 24 GiB device intermittently fails an
allocation, a practical starting point is:

```bash
MIXLAB_MLX_MEMORY_LIMIT_MB=16000 MIXLAB_MLX_MEM_LOG_EVERY=100 ./mixlab -mode arch ...
```

Treat this as an allocator bound, not a model-memory estimate. Tune it for the
device while leaving room for runtime and display allocations.

## Gated DeltaNet long sequences

On Apple GPUs, a positive `gated_deltanet.scan_chunk_size` selects a native
Metal scan for `d_k <= 64` and `d_v <= 256`. Shapes with `d_k <= 32`,
`d_v <= 32`, and `scan_chunk_size <= 64` use the exact chunk-parallel path:
independent chunk summaries, short forward/reverse chunk-prefix passes, and
parallel chunk replay. Its analytical backward propagates state cotangents
between chunks and recomputes bounded local windows. Larger supported shapes
retain the native recurrent Metal implementation.

The scan checkpoint allocation is:

```text
B * H * ceil(T / W) * d_k * d_v * sizeof(float32)
```

For `B=4`, `H=4`, `T=16000`, `d_k=16`, `d_v=32`, and `W=8`, this is 62.5 MiB
per active scan. The previous MLX-composed path repeatedly updated a full
`[B,H,num_chunks,chunk,d_v]` output while traversing chunks. Its retained
backward graph could grow approximately as `O(B*H*T^2*d_v/chunk)` even though
the recurrence itself is linear. The fallback now concatenates chunk outputs
once and checkpoints the composed scan, while supported Metal shapes avoid
that graph entirely.

The opt-in six-layer bidirectional regression probe uses `D=128`, `H=4`,
`d_k=16`, `d_v=32`, and `T=16000`:

```bash
MIXLAB_GATED_DELTA_LONG_PROBE=1 \
MIXLAB_GATED_DELTA_PROBE_BATCH=4 \
go test -tags mlx ./train -run TestGatedDeltaNetLongSequenceMemoryProbe -count=1 -v
```

All figures below come from one machine, an Apple M1 Max with 64 GiB unified
memory, and are quoted in the `peak_mib` units the probe itself prints so they
can be compared directly against a local run.

The recurrent Metal path peaks near `10,400 MiB` at batch 4 and `19,980 MiB` at
batch 8, and batch 8 completes rather than being terminated for memory
exhaustion. These are complete training-graph peaks, not just the scan
checkpoint allocation.

Memory and throughput do not travel equally. Peak memory is driven by
allocation sizes and reproduces closely on other Apple GPUs at the same shape;
throughput is device-specific and should be re-measured rather than assumed.

At batch 4 the chunk-parallel path measured `22,078 tok/s` against `7,862 tok/s`
for the recurrent fallback, a `2.81x` whole-model speedup, with peaks of
`10,550 MiB` versus `10,400 MiB`. The scan-only benchmark measured `2.69M tok/s`
forward and `535k tok/s` through forward plus backward, versus `670k` and `94k`
for the recurrent path. Run that benchmark with:

```bash
go test -tags mlx ./gpu -run '^$' \
  -bench BenchmarkGatedDeltaScanLongSequence -benchtime=3x -count=1
```

The opt-in comparison benchmark holds the six-layer, bidirectional,
BatchNorm, continuous-classification fixture constant across S4D, canonical
Mamba-3, and Gated DeltaNet. Run each arm in a separate process because MLX
peak memory is a process-wide high-water mark:

```bash
go test -tags mlx ./train -run '^$' \
  -bench 'BenchmarkBidirectionalRecurrentMixersLongSequenceTraining/gated_deltanet$' \
  -benchtime=1x -count=1
```

Replace the final arm with `s4d` or `mamba3_canonical` for the sibling
baselines. The benchmark performs an untimed compile/warmup step before the
measured step and reports `tokens/s` plus `peak-MiB`.

The first Gated DeltaNet step prints the selected training and scan paths,
including solve, backward, and output-accumulation details when applicable.
Debug fallbacks can be selected with:

```bash
MIXLAB_DISABLE_GATED_DELTA_METAL_SCAN=1       # checkpointed MLX scan
MIXLAB_DISABLE_GATED_DELTA_CHUNK_METAL=1      # recurrent Metal scan
MIXLAB_DISABLE_GATED_DELTA_SCAN_CHECKPOINT=1  # raw MLX autodiff fallback
MIXLAB_DISABLE_GATED_DELTA_METAL_SOLVE=1      # MLX triangular solve on Metal
MIXLAB_DISABLE_GATED_DELTA_CUDA_SOLVE=1       # MLX triangular solve on CUDA
```

The full Metal-disable and raw-autodiff switches materially increase
long-sequence memory. The chunk-disable switch keeps bounded memory and is for
performance comparisons or diagnosing the chunk-parallel path.

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

Throughput and ETA exclude the first complete iteration in each process,
including initial compilation, validation, diagnostics, logging, and checkpoint
work. That iteration displays `tok/s=n/a` and no ETA; JSON telemetry reports
`tokens_per_sec: 0` and `steady_elapsed_seconds: 0` until a subsequent step
completes. This also applies to the first iteration after resuming a checkpoint.
Elapsed wall time still includes startup. Later estimates use cumulative wall
time and actual processed tokens, including periodic validation/checkpoint
overhead, rather than asynchronous GPU collect-call duration.

Telemetry includes current step, loss, learning rate, objective, sequence
length, steady-state tokens/sec, MLX active/cache/peak memory, host RSS, and
best-effort GPU utilization on macOS from `ioreg`. When a training graph
declares active scalar auxiliary losses, it also includes a
`component_losses` object, for example `invariance_loss`, `pll_margin_loss`,
`word_struct_loss`, or `moe_aux_loss`; absent/no-op objectives do not add keys.
Native `ttt_mlp` runs also publish an `extra` object with block-qualified inner
loss before/after update, update norm, state drift, and inner-LR mean/min/max.
The same aggregate values print on a separate `[ttt]` line at normal training
log cadence. These values are sampled after the completed optimizer step by a
separately compiled no-gradient forward on that step's prepared batch. They are
not retained as outputs of every optimizer step, and no per-token or per-chunk
arrays are read back by default. Log steps therefore include one extra forward;
ordinary steps stay on the loss-only compiled graph.
Optimizer telemetry includes committed `optimizer_steps`, cumulative
`skipped_optimizer_steps`, `consecutive_skipped_optimizer_steps`, and
`optimizer_step_skipped` for the current batch.
Mixlab validates the complete persistent optimizer transaction after each
step. If the loss, any gradient, candidate weight, or optimizer moment is
non-finite, the candidate update is discarded atomically and the previous
weights and optimizer state remain active. Raw non-finite gradients are counted
and zeroed before norm clipping so one invalid scalar cannot turn every gradient
and moment into NaN. Three consecutive rejected updates, or one fully
non-finite candidate state, terminate training with an error after restoring
the last committed state. GPU utilization is omitted when the platform does
not expose a no-sudo sampler.

Stateful TTT-MLP inference keeps recurrent arrays on the GPU and compiles graph
variants by chunk fragment and offset. CUDA builds use a fused causal Q/K
convolution primitive with the same bounded-state recurrence and retain a
portable MLX fallback. The 512-through-32k Apple benchmark, CUDA benchmark
command, memory measurements, and reproduction details are in
[TTT-MLP stateful inference](ttt-mlp-stateful-inference.md).

Hugging Face TTT-MLP exports use a vectorized chunk dual form for stateless
forward and fine-tuning, while cached continuation retains the online scan.
For short CPU scoring, PyTorch thread-pool overhead can dominate; use
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` as the starting point for batch-one
evaluation. The exported module does not mutate the application's global
thread settings. Right-padded variable-length batches are supported; left padding
is not compatible with the recurrent position/state contract and is rejected with
`ValueError` rather than scored incorrectly.

HF TTT fine-tuning uses whole-scan activation checkpointing by default. This is
an exact recomputation strategy, not a detached or first-order inner update.
Use Hugging Face's `gradient_checkpointing_disable()` only when sufficient
device memory is available and faster backward execution is more important.
The stateless training path supports `torch.compile(model,
mode="reduce-overhead", fullgraph=True)`. Use fixed-shape batches after the
initial compile and exclude the first forward/backward from timing. The
compile contract checks eager/compiled loss and gradients; the opt-in CUDA
gate compares complete optimizer-step throughput at sequence length 512 and
reports compile latency, peak memory, and best-effort GPU utilization. See
[Hugging Face Export](hf-export.md#ttt-mlp-execution-paths) for the command and
padding constraints.

Mixlab reports analytical forward FLOPs for `ttt_mlp`, but does not report TTT
training-step FLOPs, FLOPs/token, or MFU as precise metrics. The generic
three-times-forward training proxy does not model the inner VJP plus full outer
meta-gradient. Architecture-aware TTT backward accounting is required before
those metrics can be enabled.

### Backward non-finite tracing

For a reproducible finite-forward/non-finite-backward failure, opt into the
per-op backward tracer for a narrow step range:

```bash
MIXLAB_MLX_BACKWARD_TRACE=1 \
MIXLAB_MLX_BACKWARD_TRACE_START=190 \
MIXLAB_MLX_BACKWARD_TRACE_END=192 \
mixlab -mode arch -config model.json -train 'data/train_*.bin'
```

If the start/end variables are omitted, direct tracing is limited to step 1.

For a persistent failure protected by the optimizer circuit breaker, trace the
first retry after a rejected update without perturbing the lead-up:

```bash
MIXLAB_MLX_BACKWARD_TRACE_AFTER_SKIP=1 \
mixlab -mode arch -config model.json -train 'data/train_*.bin'
```

Traced steps run eagerly and inspect every floating-point IR input/output edge.
Mixlab separately reports the earliest op that creates a non-finite forward
output and the highest forward op index whose backward emits a non-finite input
gradient. Diagnostics include op type, edge index/name, non-finite count, and
largest finite magnitude. This is expensive: keep the range small and leave it
disabled for normal training. The optimizer transaction still rejects and
rolls back a bad step.

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
