# CUDA Long-Run Hang Investigation

## Scope

Investigate canonical Mamba3 training stalls without changing model mathematics,
optimizer settings, or claiming the cause from CPU/GPU gauges alone. The reported
stall occurs around 34,300 new steps per process across resume points and GPU
speeds. Logs are periodic, so the exact failing step is not established.

## Plan And Status

1. **Complete: isolate the MLX worker loop.** Compile the pinned dependency's real
   `worker.cpp` with CUDA event delivery stubbed, test repeated callbacks, idle
   waiting, wakeup after idle, and shutdown. The original source reproduced the
   busy-spin; the patch eliminated it. This CPU-only result says nothing about
   actual CUDA ordering or the long-run trigger.
2. **Complete: make the dependency fix reproducible.** Apply a checked patch before
   building MLX; gate downstream image builds on the patch revision. CI tests the
   unpatched and patched source. No MLX version upgrade or custom kernel change.
3. **Complete: add serverless capture and liveness.** Forward boolean `timing`, add
   opt-in `stall_timeout`/`stall_dump_dir`, publish committed-step progress outside
   console output, attempt bounded native stacks before killing a stalled process,
   and keep dashboard backpressure out of the supervisor loop. Test noisy/silent
   stalls, absent progress, skipped commits, debugger errors and cleanup.
4. **Pending CUDA acceptance:** rebuild staging dependency/application images and
   verify GDB attachment on the target platform. Compare CUDA graphs on/off on the
   same patched image, checkpoint, GPU type and data. Both arms must exceed 40,000
   new steps per process; record exact committed counts, loss, throughput, memory
   and native stacks on any stall. Do not promote the image or close the long-run
   bug based solely on the isolated worker test.

## Boundaries

- Model and optimizer behavior remain unchanged.
- Progress reporting is opt-in, local, atomic, and independent of GPU sampling.
- The supervisor records failures and aborts; it does not restart or attempt to
  checkpoint a hung GPU.
- GDB capture is best effort under the host's ptrace policy, not guaranteed by
  running as container root. Only the explicit parent debugger is authorized.
- User-specific jobs, paths and datasets stay outside the public repository.

See [performance diagnostics](performance.md#cuda-long-run-isolation) and the
[RunPod job interface](../docker/README.md#runpod-serverless) for the procedure.
