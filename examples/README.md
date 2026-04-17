# mixlab Example Configs

Three example architecture configs for the mixlab training harness, from simplest
to most capable.

## Quick start

```bash
# Build mixlab with MLX backend (required for GPU training)
CGO_ENABLED=1 go build -tags mlx -o mixlab ./cmd/mixlab

# Run any config
./mixlab -mode arch -config examples/<config>.json \
  -train "data/example/train_*.bin"
```

## Configs at a glance

| Config | Architecture | model_dim | seq_len | Steps | Time (M1 Max) | Use case |
|--------|-------------|-----------|---------|-------|---------------|----------|
| [mamba_2L.json](mamba_2L.json) | Mamba SSM (2-layer) | 64 | 128 | 50 | TBD | Smoke test for Mamba blocks |
| [retnet_2L.json](retnet_2L.json) | RetNet (2-layer) | 64 | 128 | 50 | TBD | Smoke test for RetNet blocks |
| [rwkv_2L.json](rwkv_2L.json) | RWKV (2-layer) | 64 | 128 | 50 | TBD | Smoke test for RWKV blocks |
| [perceiver_2L.json](perceiver_2L.json) | Perceiver (2-layer) | 64 | 128 | 50 | TBD | Smoke test for Perceiver blocks |
| [custom_geglu.json](custom_geglu.json) | Custom GeGLU blocks | 64 | 128 | 50 | TBD | Smoke test for custom block JSON format |

## Which config should I use?


transformer.

`training.steps`, branch depths, and model dimensions for your dataset and
hardware budget.

## Architecture overview

### Plain transformer (plain_3L)

Standard sequential processing: tokens are embedded, then passed through
alternating attention and SwiGLU feed-forward blocks. No frequency decomposition.

```
tokens -> embed -> [attn -> swiglu] x 3 -> output
```


before processing. This is a lossless, deterministic transform -- the model does
not need to learn it.

```
                        |                       |
                   high0 blocks            low blocks (deep)
                                           high1 blocks (shallow)
                        |                       |
```

The block lists let you allocate different amounts of processing to each
your data, hardware, and parameter budget.

## New block types (pending implementation)

The following configs exercise block types that are being added by other agents.
They will fail to parse until the block implementations land in `config.go` and
`arch/builder.go`:

- **mamba_2L.json** -- Mamba selective state-space model. Uses gated recurrence
  (OpScan) instead of attention. No `heads` parameter needed.
- **retnet_2L.json** -- Retention network. Uses multi-head retention with
  exponential decay (`decay` parameter).
- **rwkv_2L.json** -- RWKV (Receptance Weighted Key Value). Linear attention
  variant with channel mixing.
- **perceiver_2L.json** -- Perceiver with learned latent array
  (`num_latents` parameter). Cross-attends input to latents, processes latents,
  then broadcasts back.
- **custom_geglu.json** -- Custom block defined entirely in JSON. Demonstrates
  the GeGLU feed-forward architecture using the `weights` and `ops` fields.
  See [custom_geglu.md](custom_geglu.md) for the full custom block format reference.

## Companion documentation

Each config has a companion `.md` file with detailed field explanations, expected
training behavior, and hardware requirements:

- [plain_3L.md](plain_3L.md)
- [custom_geglu.md](custom_geglu.md) -- Custom block JSON format reference
