# Config: Advanced Features

This page is a routing map for features that affect graph layout, compatibility,
or export behavior. Use [config-reference.md](config-reference.md) for exact
schemas.

## Architecture Layout

| Feature | Purpose |
|------|---------|
| `recurrence` | Share weights across block positions. |
| `recurrence_phases` | Change the block execution schedule over training phases. |
| `parallel_residual` / `parallel_group` | Run supported mixer+FFN pairs or heterogeneous attention+recurrent groups in parallel. |
| `unet` | Encoder/decoder split with learned skip connections. |
| `layer_aggregation: "dwa"` | Dense weighted aggregation over prior sublayer outputs. |
| `backout` | Final latent residual subtraction before final norm. |

These features are powerful but tend to interact with export, hidden capture,
and training-only objectives. Unsupported combinations should fail explicitly.

## Objective And Data Layout Features

| Feature | Purpose |
|------|---------|
| Masked objectives | MLM, MNTP, hybrid, and block diffusion behavior. |
| `attention_segment_mask` | Boundary-token-derived segment IDs for packed-sequence block-diagonal attention. |
| `data2vec` | EMA teacher representation targets for masked training rows. |
| `distillation` | Fixed-teacher ensemble KL target for causal and masked vocab training. |
| `mtp` | Multi-token prediction auxiliary loss. |

## Export Compatibility

Hugging Face export is intentionally gated by feature support. Check:

- [hf-export.md](hf-export.md) for workflow and parity mode.
- [hf-export-support-matrix.md](hf-export-support-matrix.md) for supported,
  gated, unsupported, and training-only features.
- [cli-export.md](cli-export.md) for `export-hf` and `parity` flags.

## Performance And Runtime

Use [performance.md](performance.md) for MLX memory limits, cache controls,
timing output, profiling, and CUDA graph controls. Keep runtime knobs outside
architecture JSON unless they need to be reproducible across machines.
