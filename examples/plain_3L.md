# plain_3L -- 3-Layer Causal Transformer

## Architecture summary

`plain_3L.json` is the simplest architecture in mixlab: a causal transformer
with `model_dim=128`, `seq_len=128`, and 6 sequential blocks (3 attention
blocks alternating with 3 SwiGLU feed-forward blocks).

## Config fields explained

| Field | Value | Meaning |
|-------|-------|---------|
| `name` | `"plain_3L"` | Human-readable identifier for logging |
| `model_dim` | `128` | Hidden dimension of the transformer (embedding size) |
| `vocab_size` | `1024` | Number of tokens in the vocabulary |
| `seq_len` | `128` | Context window length in tokens |
| `blocks` | 6 entries | The sequential stack of model blocks (see below) |
| `training.steps` | `200` | Number of training steps |
| `training.lr` | `3e-4` | AdamW learning rate |
| `training.grad_clip` | `1.0` | Maximum gradient norm (0 = no clipping) |
| `training.weight_decay` | `0.01` | AdamW weight decay coefficient |
| `training.seed` | `42` | Random seed for reproducibility |
| `training.batch_tokens` | `1024` | Total tokens per training batch |

### Block stack

The `blocks` array alternates between two block types:

| Block | Type | Heads | Purpose |
|-------|------|-------|---------|
| 0 | `plain` | 4 | Multi-head causal self-attention (128 / 4 = 32 dims per head) |
| 1 | `swiglu` | -- | SwiGLU feed-forward network (gated MLP) |
| 2 | `plain` | 4 | Multi-head causal self-attention |
| 3 | `swiglu` | -- | SwiGLU feed-forward network |
| 4 | `plain` | 4 | Multi-head causal self-attention |
| 5 | `swiglu` | -- | SwiGLU feed-forward network |

## Expected behavior

- **Training speed:** Very fast due to small model_dim and short seq_len. Expect under 1 minute for 200 steps on Apple Silicon.
- **Loss trajectory:** Training loss should decrease steadily over 200 steps but will not converge -- this is a quick sanity check, not a full training run.
- **Parameter count:** `mixlab -mode count -config examples/plain_3L.json` reports 1,116,288 parameters, about 4.26 MiB in float32 or 1.06 MiB with int8 quantized export.

## How to run

```bash
# Build with MLX backend (required)
CGO_ENABLED=1 go build -tags mlx -o mixlab ./cmd/mixlab

# Train
./mixlab -mode arch \
  -config examples/plain_3L.json \
  -train "data/example/train_*.bin"
```
