# mixlab JSON Config Reference

This reference covers the current JSON schema used by `mixlab`, based on:

- `arch/config.go` for top-level and training fields
- `arch/ir_bridge.go` for config-to-IR conversion
- `arch/builder.go`, `arch/weight_shapes.go`, and `arch/custom.go` for block behavior, defaults, and custom-block semantics

All examples below are valid JSON fragments unless otherwise noted.

## Top-level model fields

These fields live at the root of the config object.

| Field | Type | Required | Default | Notes |
|------|------|----------|---------|-------|
| `name` | string | No | Source filename/path | Human-readable run name. |
| `model_dim` | integer | Yes | None | Hidden size `D`. Must be `> 0`. |
| `vocab_size` | integer | Yes | None | Token vocabulary size `V`. Must be `> 0` and `<= 65535` (tokens are stored as uint16 in binary shards). |
| `seq_len` | integer | No | `128` | Context length in tokens. Must be `> 0` when set. |
| `mlp_mult` | number | No | `2.67` | FFN expansion multiplier for `plain`, `swiglu`, and `cross_attention` FFN tails. Must be `> 0`. |
| `logit_softcap` | number | No | Disabled | Optional soft cap applied to output logits before loss/export. |
| `bigram_vocab_size` | integer | No | Disabled | Enables model-level hashed bigram embeddings when `> 1`. `0` disables. `1` is invalid. |
| `bigram_dim` | integer | No | `model_dim` when bigrams enabled | Bigram embedding dimension. `0` inherits `model_dim`. Ignored when `bigram_vocab_size == 0`. |
| `tie_embeddings` | boolean | No | `false` | Shares token embedding and output head weights. |
| `block_scales` | boolean | No | `false` | Adds learned per-channel scales to `plain` attention and MLP residual branches, plus the MLP branch in `swiglu`. |
| `resid_mix` | boolean | No | `false` | Adds learned mixing of the current state and original input on `plain` blocks. |
| `parallel_residual` | boolean | No | `false` | Requires `(plain, swiglu)` pairs and emits attention and MLP branches from the same pre-norm input. Cannot be combined with `unet`. |
| `unet` | boolean | No | `false` | Splits the `blocks` list into encoder/decoder halves with learned skip connections. |
| `blocks` | array | Yes | None | Ordered block list. Must contain at least one block. |
| `recurrence` | integer array | No | Disabled | Weight-sharing map for `blocks`; length must equal `blocks`, references must point to the same or earlier block with the same type. |
| `training` | object | No | Defaults applied per field | Training hyperparameters. See [Training section](#training). |

### Minimal sequential model

```json
{
  "name": "plain-small",
  "model_dim": 128,
  "vocab_size": 1024,
  "seq_len": 128,
  "blocks": [
    {"type": "plain", "heads": 4},
    {"type": "swiglu"}
  ],
  "training": {
    "steps": 200,
    "lr": 0.0003,
    "batch_tokens": 1024
  }
}
```

## Block types


### `plain`

Standard causal self-attention block with RMSNorm, RoPE, grouped-query support via `kv_heads`, SiLU FFN tail, and residual connections.

Required fields:

- `type: "plain"`
- `heads`

Optional fields:

- `kv_heads`

Example:

```json
{"type": "plain", "heads": 8, "kv_heads": 4}
```

### `swiglu`

Feed-forward-only block with RMSNorm, SwiGLU gating, and residual connection.

Required fields:

- `type: "swiglu"`

Optional fields:

- None

Example:

```json
{"type": "swiglu"}
```

### `mamba`

Selective state-space block with input projection, local mixing, scan recurrence, gating, output projection, and residual add.

Required fields:

- `type: "mamba"`

Optional fields:

- `inner_dim`

Example:

```json
{"type": "mamba", "inner_dim": 768}
```

### `mamba3`

Mamba-3 style gated scan block with RMSNorm, separate gate / SSM / delta-t projections, learned temporal gating, scan recurrence, and residual add.

Required fields:

- `type: "mamba3"`

Optional fields:

- `inner_dim`

Example:

```json
{"type": "mamba3", "inner_dim": 512}
```

### `retnet`

Retention block with RMSNorm, multi-head retention mask construction, learned per-head decay weights, FFN tail, and residual connections.

Required fields:

- `type: "retnet"`
- `heads`

Optional fields:

- `decay`

Example:

```json
{"type": "retnet", "heads": 8, "decay": 0.95}
```

Notes:

- `decay` is part of the JSON schema and block spec, but the current IR implementation learns decay from block weights instead of consuming the config value directly.

### `rwkv`

Simplified RWKV-style linear-time block with learned token shift, recurrent mixing, and channel mixing.

Required fields:

- `type: "rwkv"`

Optional fields:

- None

Example:

```json
{"type": "rwkv"}
```

### `perceiver`

Latent bottleneck block: cross-attend input into learned latents, self-attend latents, then broadcast back to the sequence.

Required fields:

- `type: "perceiver"`
- `heads`

Optional fields:

- `num_latents`

Example:

```json
{"type": "perceiver", "heads": 4, "num_latents": 32}
```

### `bottleneck`

Same IR structure as `perceiver`, but with a smaller default latent count and intended use as a tighter sequence bottleneck.

Required fields:

- `type: "bottleneck"`
- `heads`

Optional fields:

- `num_latents`

Example:

```json
{"type": "bottleneck", "heads": 4, "num_latents": 4}
```

### `cross_attention`

Cross-attention block where the current stream provides queries and
`source_stream` provides keys and values. No causal mask is applied to the
source stream.

Required fields:

- `type: "cross_attention"`
- `heads`
- `source_stream`

Optional fields:

- None

Example:

```json
{"type": "cross_attention", "heads": 4, "source_stream": "low1"}
```

Notes:

- `source_stream` must name a stream known to the IR builder at compile time.
- No causal mask is applied to the source stream; the query stream can attend to all source positions.

### `token_blend`

Single-weight learned token blending gate. It mixes each token with its shifted predecessor using a sigmoid gate.

Required fields:

- `type: "token_blend"`

Optional fields:

- None

Example:

```json
{"type": "token_blend"}
```

### `custom`

User-defined block composed entirely from declared weights and an ordered list of IR ops.

Required fields:

- `type: "custom"`
- `name`
- `weights`
- `ops`

Optional fields:

- `heads`

Example:

```json
{
  "type": "custom",
  "name": "geglu",
  "weights": [
    {"name": "w_gate", "shape": ["D", "FFN"]},
    {"name": "w_up", "shape": ["D", "FFN"]},
    {"name": "w_down", "shape": ["FFN", "D"]}
  ],
  "ops": [
    {"op": "matmul", "inputs": ["x", "w_gate"], "output": "gate"},
    {"op": "silu", "inputs": ["gate"], "output": "gate_act"},
    {"op": "matmul", "inputs": ["x", "w_up"], "output": "up"},
    {"op": "mul", "inputs": ["gate_act", "up"], "output": "ff"},
    {"op": "matmul", "inputs": ["ff", "w_down"], "output": "ff_out"},
    {"op": "add", "inputs": ["x", "ff_out"], "output": "x"}
  ]
}
```

## Custom blocks

Custom blocks expose a low-level JSON-to-IR interface.

### Weight declaration format

Each `weights` entry is:

```json
{"name": "w_name", "shape": ["SYM0", "SYM1", "..."]}
```

Rules:

- `name` must be unique within the block.
- `shape` must contain one or more symbolic or literal dimensions.
- The declared order is the IR weight order for that block.

### Op format

Each `ops` entry is:

```json
{
  "op": "matmul",
  "inputs": ["x", "w"],
  "output": "tmp",
  "outputs": ["q_rot", "k_rot"],
  "params": {"shape": ["B", "T", "H", "HD"]}
}
```

Rules:

- Use `output` for single-output ops.
- Use `outputs` for multi-output ops such as `rope`.
- At least one of `output` or `outputs` is required.
- `x` always refers to the current stream state.
- Weight names resolve to the declared weights.
- All other names become temporary variables local to the block.

### Shape symbols

These symbols are accepted in custom weight shapes and in `params.shape`:

| Symbol | Meaning |
|------|---------|
| `D` | `model_dim` |
| `H` | `heads` for this custom block, defaulting to `1` when omitted |
| `HD` | `D / H` |
| `T` | Sequence length |
| `B` | Batch size |
| `V` | Vocabulary size |
| `BT` | `B * T` |
| `2D` / `3D` / `4D` / `8D` | Integer multiples of `D` |
| `FFN` | `int(2.67 * D)` |
| `T/2` | Half sequence length |
| `<integer>` | Literal integer dimension |
| `<float>D` | Floating-point multiple of `D`, for example `1.5D` or `2.67D` |

### Supported ops

| Op | Inputs | Outputs | Supported params | Notes |
|------|--------|---------|------------------|-------|
| `matmul` | 2 | 1 | None | Matrix multiply. |
| `add` | 2 | 1 | None | Elementwise add. |
| `sub` | 2 | 1 | None | Elementwise subtract. |
| `mul` | 2 | 1 | None | Elementwise multiply. |
| `div` | 2 | 1 | None | Elementwise divide. |
| `scalar_mul` | 1 | 1 | `scalar` | Multiply tensor by scalar. |
| `sigmoid` | 1 | 1 | None | Elementwise sigmoid. |
| `silu` | 1 | 1 | None | SiLU activation. |
| `gelu` | 1 | 1 | None | GELU activation. |
| `relu` | 1 | 1 | None | ReLU activation. |
| `tanh` | 1 | 1 | None | Tanh activation. |
| `softmax` | 1 | 1 | `axis` | Softmax over an axis. |
| `reshape` | 1 | 1 | `shape` | Shape values can use symbolic dims. |
| `transpose` | 1 | 1 | `axes` | Permute tensor dimensions. |
| `rmsnorm` / `rms_norm` | 2 | 1 | `eps` | Inputs are typically `[x, scale]`. |
| `rope` | 2 | 2 | `T`, `head_dim`, `base` | Rotary embedding helper. Outputs are rotated Q and K. |

### Param keys

The custom-op decoder recognizes these `params` keys:

| Param key | Type | Used by |
|------|------|---------|
| `shape` | array of strings/numbers | `reshape` |
| `axes` | array of integers | `transpose` |
| `axis` | integer | `softmax` |
| `head_dim` | integer | `rope` |
| `T` | string or number | `rope` |
| `scalar` | number | `scalar_mul` |
| `eps` | number | `rmsnorm` |
| `base` | number | `rope` |

### Custom block example with reshape

```json
{
  "type": "custom",
  "name": "reshape_example",
  "heads": 4,
  "weights": [
    {"name": "w", "shape": ["D", "D"]}
  ],
  "ops": [
    {"op": "matmul", "inputs": ["x", "w"], "output": "h"},
    {"op": "reshape", "inputs": ["h"], "output": "h4", "params": {"shape": ["B", "T", "H", "HD"]}},
    {"op": "transpose", "inputs": ["h4"], "output": "h4t", "params": {"axes": [0, 2, 1, 3]}},
    {"op": "reshape", "inputs": ["h4t"], "output": "x", "params": {"shape": ["BT", "D"]}}
  ]
}
```

## Training

The `training` object controls optimization, batching, and stochastic settings.

| Field | Type | Required | Default | Notes |
|------|------|----------|---------|-------|
| `steps` | integer | No | `200` | Total training steps. Must be `> 0`. |
| `lr` | number | No | `3e-4` | Base learning rate. Must be `> 0`. |
| `warmdown_steps` | integer | No | `0` | Cosine warmdown length at the end of training. Must be `>= 0`; values above `steps` are clamped by the scheduler. |
| `target_val_loss` | number | No | `0` | Early-stop threshold on validation loss. `0` disables it. Must be `>= 0`. Checked when validation loss is computed during training. |
| `ttt_steps` | integer | No | `0` | Score-first test-time training updates per validation batch during eval mode and full BPB eval. `0` disables TTT. Must be `>= 0`. |
| `ttt_lr` | number | No | `1e-5` | Learning rate for TTT updates. Must be `>= 0`; keep much smaller than training `lr`. |
| `optimizer` | string | No | `"muon"` | Optimizer for matrix (rank ≥ 2) weights: `"muon"` or `"adamw"`. Embed, head, and scalar groups always use AdamW. |
| `weight_init` | string | No | `"xavier_uniform"` | Initialization for rank ≥ 2 weights: `"xavier_uniform"` or `"normal"`. 1D weights are always ones (norms) or zeros. |
| `weight_init_std` | number | No | `0.02` | Standard deviation for `"normal"` initialization. Ignored when `weight_init` is `"xavier_uniform"`. |
| `grad_clip` | number | No | `0` | Max grad norm. `0` means no clipping. Must be `>= 0`. |
| `weight_decay` | number | No | `0.01` | Global fallback weight decay. Must be `>= 0`. |
| `beta1` | number | No | `0.9` | AdamW beta1. Also seeds Muon momentum when `muon_momentum` is omitted. |
| `beta2` | number | No | `0.95` | AdamW and Muon beta2. |
| `epsilon` | number | No | `1e-8` | AdamW / Muon epsilon. |
| `seed` | integer | No | `42` | RNG seed. `0` is treated as omitted and replaced with `42`. |
| `batch_tokens` | integer | No | `1024` | Tokens per optimization step. Must be divisible by `seq_len`. |
| `shuffle_chunk_tokens` | integer | No | `seq_len` | Token-block shuffle granularity for train/validation loaders. Values `<= 0` inherit `seq_len`; set to `2048` to reproduce the previous fixed-block behavior. |
| `embed_lr` | number | No | `lr` | Learning rate for embedding-class weights. |
| `matrix_lr` | number | No | `lr` | Learning rate for matrix weights. Used with Muon. |
| `scalar_lr` | number | No | `lr` | Learning rate for scalar and vector weights. |
| `head_lr` | number | No | `lr` | Learning rate for the output head. Ignored when `tie_embeddings=true` because the head shares the embedding weight. |
| `muon_momentum` | number | No | `beta1` | Muon momentum term for matrix weights. Must be `>= 0`. |
| `muon_backend_steps` | integer | No | `5` | Muon backend iteration count. Must be `> 0` after defaults. |
| `muon_nesterov` | boolean | No | `true` | Enables Muon Nesterov mode when set or omitted. |
| `embed_weight_decay` | number | No | `weight_decay` | Per-group weight decay for embeddings. |
| `matrix_weight_decay` | number | No | `weight_decay` | Per-group weight decay for matrices. |
| `scalar_weight_decay` | number | No | `weight_decay` | Per-group weight decay for scalars and vectors. |
| `head_weight_decay` | number | No | `weight_decay` | Per-group weight decay for the output head. |
| `swa_start` | integer | No | `0` | Step at which SWA/EMA accumulation starts. Must be `>= 0`. |
| `swa_decay` | number | No | `0.999` | EMA decay for SWA weights. Must be in `[0, 1)`. |
| `swa_interval` | integer | No | `10` | Update frequency for SWA accumulation. |

### Optimizer groups

The trainer classifies weights into four optimizer groups:

| Group | Optimizer | Typical weights | LR field | Weight-decay field |
|------|-----------|-----------------|----------|--------------------|
| Embedding | AdamW | `embed`, `bigram_table` | `embed_lr` | `embed_weight_decay` |
| Head | AdamW | `head` | `head_lr` | `head_weight_decay` |
| Scalar | AdamW | Norm scales, decay vectors, learned scalar scales | `scalar_lr` | `scalar_weight_decay` |
| Matrix | Muon | Projection and FFN matrices | `matrix_lr` | `matrix_weight_decay` |

### Training example

```json
{
  "training": {
    "steps": 20000,
    "lr": 0.0003,
    "warmdown_steps": 1000,
    "target_val_loss": 1.2,
    "grad_clip": 0.3,
    "weight_decay": 0.01,
    "beta1": 0.9,
    "beta2": 0.95,
    "epsilon": 1e-8,
    "seed": 1337,
    "batch_tokens": 4096,
    "shuffle_chunk_tokens": 1024,
    "embed_lr": 0.6,
    "matrix_lr": 0.02,
    "scalar_lr": 0.02,
    "head_lr": 0.008,
    "muon_momentum": 0.99,
    "muon_backend_steps": 5,
    "muon_nesterov": true,
    "swa_start": 10000,
    "swa_decay": 0.999,
    "swa_interval": 10
  }
}
```

## Advanced architecture features

These optional fields extend the sequential block stack without changing the
top-level `blocks` array format.

### `unet`

Turns a sequential `blocks` list into encoder/decoder halves with learned skip weights.

Example:

```json
{
  "unet": true,
  "blocks": [
    {"type": "token_blend"},
    {"type": "plain", "heads": 8},
    {"type": "swiglu"},
    {"type": "plain", "heads": 8},
    {"type": "swiglu"},
    {"type": "plain", "heads": 8}
  ]
}
```

Notes:

- Skip weights are inserted automatically by the IR builder.

### `block_scales`

Adds learned per-channel scaling on selected residual branches.

Example:

```json
{
  "block_scales": true,
  "blocks": [
    {"type": "plain", "heads": 8},
    {"type": "swiglu"}
  ]
}
```

Effect:

- `plain` gains `attn_scale` and `mlp_scale`.
- `swiglu` gains `mlp_scale`.

### `resid_mix`

Adds a learned mix between the current stream state and the original stream input before each `plain` block.

Example:

```json
{
  "resid_mix": true,
  "blocks": [
    {"type": "plain", "heads": 8},
    {"type": "swiglu"},
    {"type": "plain", "heads": 8}
  ]
}
```

Effect:

- Each `plain` block receives an extra `resid_mix` weight of shape `[2, D]`.

### `parallel_residual`

Fuses consecutive `(plain, swiglu)` pairs so the SwiGLU branch uses the same pre-attention RMSNorm input as the plain block.

Example:

```json
{
  "parallel_residual": true,
  "blocks": [
    {"type": "plain", "heads": 8},
    {"type": "swiglu"},
    {"type": "plain", "heads": 8},
    {"type": "swiglu"}
  ]
}
```

Effect:

- `blocks` must be complete `(plain, swiglu)` pairs.
- Each pair omits the separate `swiglu` RMSNorm scale weight.

### `tie_embeddings`

Shares token embedding and output head weights.

Example:

```json
{
  "tie_embeddings": true,
  "blocks": [
    {"type": "plain", "heads": 8}
  ]
}
```

Effect:

- Drops the separate head weight tensor.
- `head_lr` becomes informational only; the shared weight follows the embedding optimizer group.

### `mlp_mult`

Changes the hidden size used by FFN sublayers.

Example:

```json
{
  "mlp_mult": 3.0,
  "blocks": [
    {"type": "plain", "heads": 8},
    {"type": "swiglu"},
    {"type": "cross_attention", "heads": 8, "source_stream": "low1"}
  ]
}
```

Effect:

- `plain`, `swiglu`, and `cross_attention` expand to `round(model_dim * mlp_mult)`, clamped to at least `model_dim`.

## Full example: `recurrent_parallel.json`

This advanced architecture example combines recurrence, parallel residuals,
GQA, tied embeddings, bigram embeddings, logit softcap, and test-time training:

```jsonc
{
  "name": "recurrent_parallel",
  "model_dim": 512,
  "vocab_size": 8192,
  "seq_len": 2048,
  "mlp_mult": 3.0,
  "logit_softcap": 30.0,
  "bigram_vocab_size": 4096,
  "bigram_dim": 128,
  "tie_embeddings": true,
  "parallel_residual": true,
  "block_scales": true,
  "resid_mix": true,
  "recurrence": [0, 1, 2, 3, 4, 5, 2, 3, 4, 5],
  "blocks": [
    {"type": "plain", "heads": 8, "kv_heads": 4},
    {"type": "swiglu"},
    {"type": "plain", "heads": 8, "kv_heads": 4},
    {"type": "swiglu"},
    {"type": "plain", "heads": 8, "kv_heads": 4},
    {"type": "swiglu"},
    {"type": "plain", "heads": 8, "kv_heads": 4},
    {"type": "swiglu"},
    {"type": "plain", "heads": 8, "kv_heads": 4},
    {"type": "swiglu"}
  ],
  "training": {
    "steps": 3000,
    "lr": 3e-4,
    "batch_tokens": 4096,
    "ttt_steps": 1,
    "ttt_lr": 1e-5
  }
}
```

See `examples/recurrent_parallel.json` for the runnable version with inline
comments, and `examples/unet_transformer.json` for the U-Net path.
