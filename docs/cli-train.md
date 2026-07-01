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
| `-safetensors-load FILE` | Load weights before training, resume, or eval-only. |
| `-checkpoint-dir DIR` | Directory for periodic safetensors checkpoints. |
| `-checkpoint-every N` | Save a checkpoint every `N` training steps. `0` disables. |
| `-eval-after-train` | Run full validation BPB evaluation after training. Preferred alias for legacy `-eval`. |
| `-eval` | Legacy alias for `-eval-after-train`. |
| `-lut-dir DIR` | Directory for BPB lookup tables. Default: `data`. |
| `-log-every N` | Print progress every `N` training steps. `0` uses config/defaults; `MIXLAB_LOG_EVERY` overrides. |
| `-val-every N` | Run validation every `N` training steps. `0` uses config/defaults; `MIXLAB_VAL_EVERY` overrides. |
| `-timing` | Print data/GPU/validation/log timing at progress intervals. |
| `-swa-start N` | Override `training.swa_start`. `0` disables. |
| `-swa-decay X` | Override `training.swa_decay`. |
| `-swa-interval N` | Override `training.swa_interval`. |

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
