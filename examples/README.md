# mixlab Example Configs

Example architecture configs from simplest to most advanced. All configs
support JSONC comments (`//`) for inline documentation.

## Quick start

```bash
# Build mixlab with MLX backend
make build

# Run any config
./mixlab -mode arch -config examples/<config>.json \
  -train "data/example/train_*.bin"
```

## Configs at a glance

| Config | Architecture | Key features |
|--------|-------------|--------------|
| [plain_3L.json](plain_3L.json) | 3-layer transformer | Simplest config — attention + SwiGLU |
| [gqa_8h4kv.json](gqa_8h4kv.json) | GQA transformer | 8 query heads, 4 KV heads |
| [token_blend_plain.json](token_blend_plain.json) | Token blending | Learned adjacent-token gate |
| [bigram_plain.json](bigram_plain.json) | Bigram embedding | Hashed bigram context features |
| [softcap_plain.json](softcap_plain.json) | Logit softcap | Bounded logits before loss |
| [mamba_2L.json](mamba_2L.json) | Mamba SSM | Gated recurrence, no attention |
| [retnet_2L.json](retnet_2L.json) | RetNet | Exponential decay retention |
| [rwkv_2L.json](rwkv_2L.json) | RWKV | Linear attention with time decay |
| [perceiver_2L.json](perceiver_2L.json) | Perceiver | Latent bottleneck cross-attention |
| [custom_geglu.json](custom_geglu.json) | Custom block | Gated feed-forward block defined in pure JSON |
| [unet_transformer.json](unet_transformer.json) | U-Net transformer | Skip connections, block scales, residual mixing |
| [recurrent_parallel.json](recurrent_parallel.json) | Recurrent parallel | Depth recurrence, parallel residuals, TTT |

## Which config should I use?

- **Learning mixlab**: Start with `plain_3L.json` — it trains in seconds.
- **Exploring block types**: Try `mamba_2L.json`, `retnet_2L.json`, or `rwkv_2L.json`.
- **Custom architectures**: See `custom_geglu.json` and [custom_geglu.md](custom_geglu.md).
- **Advanced features**: `unet_transformer.json` and `recurrent_parallel.json` cover U-Net skips, recurrence, parallel residuals, block scales, residual mixing, tied embeddings, and TTT.

## Companion documentation

- [plain_3L.md](plain_3L.md) — Plain transformer field reference
- [custom_geglu.md](custom_geglu.md) — Custom block JSON format reference
