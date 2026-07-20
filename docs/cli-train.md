# CLI: Training

Training modes consume JSON configs and binary token shards. Model structure,
objectives, optimizer settings, and most training hyperparameters live in the
config file; CLI flags select files, checkpointing, logging, and run-level
overrides.

## `arch`

Train one config:

```bash
./mixlab -mode arch \
  -config examples/plain_3L.json \
  -train 'data/example/train_*.bin'
```

Common flags:

| Flag | Description |
|------|-------------|
| `-config` | Required. JSON architecture config. |
| `-train` | Required. Glob pattern for training shards. |
| `-safetensors FILE` | Export final weights to safetensors after training. |
| `-safetensors-load FILE` | Load model weights only for a warm start. Optimizer and schedule state start fresh. |
| `-resume PATH` | Resume from a complete checkpoint directory, `.resume.json` manifest, model file, or state file. |
| `-checkpoint-dir DIR` | Directory for periodic model and resumable-state checkpoint bundles. |
| `-checkpoint-every N` | Save a complete resumable bundle every `N` training steps. `0` disables. |
| `-eval-after-train` | Run full validation BPB evaluation after training. Preferred alias for legacy `-eval`. |
| `-eval` | Legacy alias for `-eval-after-train`. |
| `-lut-dir DIR` | Directory for BPB lookup tables. Default: `data`. |
| `-log-every N` | Print progress every `N` training steps. `0` uses config/defaults; `MIXLAB_LOG_EVERY` overrides. |
| `-val-every N` | Run validation every `N` training steps. `0` uses config/defaults; `MIXLAB_VAL_EVERY` overrides. |
| `-timing` | Print data/GPU/validation/log timing at progress intervals. |
| `-swa-start N` | Override `training.swa_start`. `0` disables. |
| `-swa-decay X` | Override `training.swa_decay`. |
| `-swa-interval N` | Override `training.swa_interval`. |

### Resume And Extension

Periodic checkpoints remain usable as ordinary model-weight files, but now
also publish a completion manifest and training-state companion:

```text
step_010000.st                       # live model weights (legacy path)
step_010000.state.safetensors        # optimizer and auxiliary EMA tensors
step_010000.resume.json              # completion manifest and scalar state
```

When SWA has started, the existing `.final.safetensors` and
`.swa.safetensors` model files replace the legacy `.st` model path. The resume
manifest always selects the live `.final.safetensors` weights and separately
records SWA state.

Resume the newest complete bundle in a directory:

```bash
./mixlab -mode arch \
  -config model.json \
  -train 'data/train_*.bin' \
  -resume runs/model/checkpoints
```

`-resume` restores model weights, optimizer moments and momentum, attempted and
committed optimizer counters, SWA, data2vec EMA state, early-stop state, global
step, deterministic dropout keys, objective RNG position, and the training data
position. Mixlab restores the loader position by replaying prior batches from
`training.seed`; this avoids serializing prefetch internals but can make startup
noticeable for very large step counts.

The manifest is written last. Directory resume ignores partial model/state
files left by an interrupted write and selects the highest complete step.
Mixlab rejects a changed model/training config, optimizer plan, or training
dataset. For a standard schedule, only increasing `training.steps` is allowed.
The original warmup/cosine schedule runs to its original horizon, then an
extension remains at the original terminal/floor LR: it never rewarms or jumps
back to peak. Phase-schedule extension is rejected in v1; resuming the original
phase horizon is supported. A checkpoint at or beyond configured steps requires
raising `training.steps`.

`-safetensors-load` is intentionally unchanged and remains a weights-only warm
start. It is mutually exclusive with `-resume`.

Optimizer state adds roughly two model-weight copies for AdamW/LAMB. A normal
resumable bundle is therefore about three times model-weight size including the
model itself; active SWA or data2vec EMA adds another copy for each feature.

Quantization flags:

| Flag | Description |
|------|-------------|
| `-quantize MODE` | Weight quantization mode: `none`, `int8`, or `int6`. |
| `-quant-method METHOD` | Quantization clipping method: `quantile` or `sdclip`. |
| `-quant-k X` | SDClip `k` for matrix weights. |
| `-quant-k-embed X` | SDClip `k` for embedding weights. |

Profiling flags:

| Flag | Description |
|------|-------------|
| `-cpuprofile FILE` | Write a Go CPU profile. |
| `-memprofile FILE` | Write a Go heap profile at exit. |
| `-pprof-addr ADDR` | Serve live pprof and Mixlab telemetry HTTP endpoints, for example `127.0.0.1:6060`. |
| `-telemetry-out FILE` | Write periodic Mixlab telemetry snapshots as JSONL. |

## `arch_race`

Train every `.json` config in a directory and print a ranked summary:

```bash
./mixlab -mode arch_race \
  -configs examples/ \
  -train 'data/example/train_*.bin'
```

Common flags:

| Flag | Description |
|------|-------------|
| `-configs DIR` | Required. Directory of JSON configs. |
| `-train` | Required. Glob pattern for training shards. |
| `-safetensors FILE` | Export final weights after each run when supported. |
| `-safetensors-load FILE` | Load weights before each run. |
| `-eval-after-train` / `-eval` | Run full validation BPB evaluation after each run. |
| `-lut-dir DIR` | Directory for BPB lookup tables. |
| `-log-every N` | Progress interval override. |
| `-val-every N` | Validation interval override. |
| `-timing` | Print timing breakdowns. |
| `-pprof-addr ADDR` | Serve live pprof and Mixlab telemetry HTTP endpoints. |
| `-telemetry-out FILE` | Write periodic Mixlab telemetry snapshots as JSONL. |

## Memory And Backend Knobs

For long MLX runs, see [performance.md](performance.md) for cache limits,
memory logging, profiling, and CUDA graph controls.
