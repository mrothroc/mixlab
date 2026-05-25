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
| `mlp_mult` | number | No | `2.67` | FFN expansion multiplier for `plain`, `swiglu`, `geglu`, `mlp`, and `cross_attention` FFN tails. Must be `> 0`. |
| `logit_softcap` | number | No | Disabled | Optional soft cap applied to output logits before loss/export. |
| `smear_embeddings` | boolean | No | `false` | Enables 1-token-lookback smearing on token embeddings before the first block. |
| `smear_embeddings_gate_shape` | string | No | `"pr130"` when smearing is enabled | Gate variant for `smear_embeddings`: `"pr130"`, `"per_channel"`, or `"per_position_per_channel"`. |
| `bigram_vocab_size` | integer | No | Disabled | Enables model-level hashed bigram embeddings when `> 1`. `0` disables. `1` is invalid. |
| `bigram_dim` | integer | No | `model_dim` when bigrams enabled | Bigram embedding dimension. `0` inherits `model_dim`. Ignored when `bigram_vocab_size == 0`. |
| `tie_embeddings` | boolean | No | `false` | Shares token embedding and output head weights. |
| `block_scales` | boolean | No | `false` | Adds learned per-channel scales to `plain` attention and MLP residual branches, plus the MLP branch in `swiglu` and `geglu`. |
| `resid_mix` | boolean | No | `false` | Adds learned mixing of the current state and original input on `plain` blocks. |
| `parallel_residual` | boolean | No | `false` | Top-level form: enables parallel residual on every consecutive `(plain or gated_deltanet, swiglu or geglu)` pair. Per-block form: set `parallel_residual: true` on individual pair-start blocks instead (see [`parallel_residual`](#parallel_residual)). Cannot be combined with `unet`. |
| `unet` | boolean | No | `false` | Splits the `blocks` list into encoder/decoder halves with learned skip connections. |
| `mtp` | object | No | Disabled | Enables parameter-free multi-token prediction during training. See [MTP section](#multi-token-prediction-mtp). |
| `backout` | object | No | Disabled | Enables final-latent residual subtraction before the final RMSNorm. See [Backout section](#backout). |
| `blocks` | array | Yes | None | Ordered block list. Must contain at least one block. |
| `recurrence` | integer array | No | Disabled | Weight-sharing map for `blocks`; length must equal `blocks`, references must point to the same or earlier block with the same type. |
| `recurrence_phases` | array | No | Disabled | Explicit multi-phase block-execution schedule. See [Recurrence phases section](#recurrence-phases). |
| `data` | object | No | Defaults applied per field | Data-loader behavior. See [Data section](#data). |
| `training` | object | No | Defaults applied per field | Training hyperparameters. See [Training section](#training). |
| `eval` | object | No | Disabled | Optional evaluation-only behavior. See [Eval section](#eval). |

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

## Smear Token Embeddings

`smear_embeddings` mixes the previous token embedding into the current token embedding before flattening and before optional bigram/trigram residual embeddings. Position `0` is left unchanged, so the BOS/no-previous-token boundary is not trainable and cannot leak a wrapped final token into the first position.

```json
{
  "smear_embeddings": true,
  "smear_embeddings_gate_shape": "pr130"
}
```

Gate variants:

| Value | Extra weights | Behavior |
|------|---------------|----------|
| `"pr130"` | `smear_gate: [12,1]`, `smear_scale: [1]` | PR #130 parity: for positions `1..T-1`, computes `smear_scale * sigmoid(E[x[t]][:12] @ smear_gate)` and multiplies that scalar gate into `E[x[t-1]]`. Both weights initialize to zero, so the feature starts as an exact no-op. Requires `model_dim >= 12`. |
| `"per_channel"` | `smear_gate: [D]` | Static per-channel gate for the previous token embedding at positions `1..T-1`. Initializes to zero. |
| `"per_position_per_channel"` | `smear_gate: [T,D]` | Static per-position/per-channel gate. The row for position `0` is allocated but unused by the emitted graph, preserving the BOS boundary. Initializes to zero. |

The smear gate weights are routed through the scalar/Adam optimizer group, not the matrix/Muon group.

## Multi-Token Prediction (MTP)

`mtp` adds training-only auxiliary losses for predicting multiple future tokens from the same final hidden state and shared LM head. It does not add per-horizon parameters. Validation, full eval, hidden-stats, and generation use next-token scoring only.

```json
{
  "mtp": {
    "n": 4,
    "loss_weights": [1.0, 0.5, 0.25, 0.125],
    "untie_embed_at_frac": 0.667
  }
}
```

| Field | Type | Required | Default | Notes |
|------|------|----------|---------|-------|
| `n` | integer | No | `1` | Number of future-token losses. `1` preserves current next-token behavior. Must be `>= 1` and `<= seq_len`. |
| `loss_weights` | number array | No | `[1, 0.5, 0.25, ...]` | Per-horizon coefficients. Length must equal `n`; values must be non-negative and sum to `> 0`. The emitted loss is a weighted average. |
| `untie_embed_at_frac` | number | No | `1.0` | Fraction of training at which an initially tied embedding/head pair splits. Values must be in `[0,1]`; `< 1` requires `tie_embeddings: true` and reserves a `head` weight. |

## Backout

`backout` captures the residual stream after a configured physical block and subtracts a learned scalar-weighted copy immediately before the final RMSNorm and LM head. When omitted, the emitted IR and weight layout stay unchanged.

```json
{
  "backout": {
    "save_layer": 7,
    "lambda_init": -1.0
  }
}
```

| Field | Type | Required | Default | Notes |
|------|------|----------|---------|-------|
| `save_layer` | integer | Yes | None | Zero-based block index captured after that block finishes. Must be `>= 0` and `< len(blocks)-1`. |
| `lambda_init` | number | No | `-1.0` | Initial value for learned scalar weight `backout_lambda`. Must be finite. |

`backout_lambda` has shape `[1]`, is initialized from `lambda_init`, and is routed through the scalar optimizer group with no weight decay. `backout` composes with recurrence, parallel residual, tied or untied heads, and MTP. It is not supported with `unet`.

## Recurrence phases

`recurrence_phases` defines an explicit multi-phase block-execution schedule. Each phase declares a training-progress threshold (`frac`) at which it activates and an `order` array listing which block positions execute during that phase. Weight sharing remains controlled by the top-level [`recurrence`](#top-level-model-fields) array; phases only change *which* blocks run at each point in training, not the weight layout.

Use this to faithfully reproduce recipes like "warm up on the first two blocks for the first half of training, then add the rest" without resorting to the simpler single-frac legacy fields.

```json
{
  "blocks": [
    {"type": "plain", "heads": 4},
    {"type": "swiglu"},
    {"type": "plain", "heads": 4},
    {"type": "swiglu"}
  ],
  "recurrence": [0, 1, 0, 1],
  "recurrence_phases": [
    {"frac": 0.0, "order": [0, 1]},
    {"frac": 0.5, "order": [0, 1, 2, 3]}
  ],
  "training": {"steps": 1000}
}
```

In the example above, steps `0..499` execute only blocks 0 and 1; steps `500..999` execute all four. Blocks 2 and 3 share weights with blocks 0 and 1 respectively (per `recurrence`), so the weight layout is identical across phases.

### Phase fields

| Field | Type | Required | Default | Notes |
|------|------|----------|---------|-------|
| `frac` | number | Yes | None | Training-progress threshold in `[0,1)` at which this phase begins. The first phase's `frac` must be exactly `0.0`. Subsequent entries must be strictly ascending and map to distinct integer start steps (`floor(frac * total_steps)`). |
| `order` | integer array | Yes | None | Block positions (zero-indexed into `blocks`) executed during this phase, in execution order. Must be non-empty and contain no duplicate positions within the phase. Positions not listed do not execute. |

### Validation rules

Phase-level:
- The first phase must have `frac: 0.0`.
- `frac` values are strictly ascending and must map to distinct integer start steps. Two phases that round to the same step (because `total_steps` is small) are rejected.
- Every phase's `order` must be non-empty.

Per-phase order:
- Each entry must be a valid block index `[0, len(blocks))` and must not repeat within the phase.
- When the global `recurrence` array shares weights, the **root** block (the position that owns the weights) must appear earlier in the same phase than any block that points at it. Phases that activate a sharer without its root are rejected.
- For `plain` blocks with `kv_source > 0`, the referenced source block must appear earlier in the same phase.
- `parallel_residual` pairs `[i, i+1]` must remain contiguous within any phase that includes either side.
- If `backout` is configured, every phase's `order` must include the `backout.save_layer` index.

### Composition and limits

- Mutually exclusive with `training.recurrence_activation_frac` and `training.recurrence_activation_step`. Setting either alongside `recurrence_phases` is rejected.
- The placeholder `execution_order` and `recurrence_phase_activations` schemas are not implemented; configs using them are rejected explicitly.
- Not supported with `unet`.
- Requires positive total training steps (set via `training.steps` or `training.phases`).
- The trainer rebuilds the IR program at each phase boundary; weight tensors are unchanged across transitions, so the same per-block weight indices apply in every phase.
- `mode count` and `EstimateFLOPs` report the **max-cost phase** (the phase with the most expensive order). Per-phase cost reporting is not currently exposed.

## Block types


### `plain`

Self-attention block with RMSNorm, RoPE, grouped-query support via `kv_heads`, SiLU FFN tail, and residual connections. It defaults to causal attention for causal training objectives and bidirectional attention for masked objectives.

Required fields:

- `type: "plain"`
- `heads`

Optional fields:

- `kv_heads` — grouped-query attention (must divide `heads` evenly)
- `qk_gain` — learnable per-head QK scaling. When set, allocates one trainable scalar per head initialized to this value, applied as `scores = qk_gain * (Q @ K^T / sqrt(d_k))`. Omit or set to `0` for standard scaling.
- `rope_dims` — partial RoPE: apply rotary embeddings to only the first `rope_dims` dimensions per head, leaving the rest position-invariant. Must be even and `<= head_dim`. Omit or set to `0` for full RoPE.
- `xsa` — eXplicit Subspace Attention: after computing `y = softmax(QK^T)V`, projects `y` orthogonal to `V` at each position. Forces attention to contribute information that V doesn't already provide. Zero additional parameters. Compatible with GQA.
- `attention_mask` — `"causal"`, `"bidirectional"`, or `"none"`. Omit to resolve from `training.objective`: causal objectives use `"causal"` and masked objectives use `"bidirectional"`. `"bidirectional"` and `"none"` both use dense softmax with no triangular mask.
- `window_size` — sliding causal attention width. `0` means full causal attention. Valid only when the resolved `attention_mask` is `"causal"`.
- `kv_source` — reuse K/V projections from an earlier block (1-indexed). Saves 2 weight matrices per shared block. Source must be an earlier `plain` block with matching `heads` and `kv_heads`. Not supported with `parallel_residual`.
- `parallel_residual` — when `true`, fuses this block with the immediately following `swiglu` block into a parallel residual pair. See [`parallel_residual`](#parallel_residual) for the full description.

Example:

```json
{"type": "plain", "heads": 8, "kv_heads": 4, "qk_gain": 5.25, "rope_dims": 16, "xsa": true}
```

KV sharing example:

```json
{"blocks": [
  {"type": "plain", "heads": 8, "kv_heads": 4},
  {"type": "swiglu"},
  {"type": "plain", "heads": 8, "kv_heads": 4, "kv_source": 1}
]}
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

### `geglu`

Feed-forward-only block with RMSNorm, GEGLU gating, and residual connection.
It has the same weight layout as `swiglu`, but gates with `GELU(x @ w_gate)` instead of `sigmoid(x @ w_gate)`.

Required fields:

- `type: "geglu"`

Optional fields:

- None

Example:

```json
{"type": "geglu"}
```

### `mlp`

Feed-forward-only block with RMSNorm, one up projection, configurable activation, one down projection, and residual connection.

Required fields:

- `type: "mlp"`

Optional fields:

- `activation` — one of `"silu"` (default), `"gelu"`, `"relu"`, or `"leaky_relu_sq"`.
- `leaky_slope` — negative slope used only by `"leaky_relu_sq"`; defaults to `0.5`.

Weight layout:

- `ffn_norm_scale`: `[D]`
- `w_up`: `[D, FFN]`
- `w_down`: `[FFN, D]`

`leaky_relu_sq` emits `LeakyReLU(x, leaky_slope)` followed by `Square`.

Example:

```json
{"type": "mlp", "activation": "leaky_relu_sq", "leaky_slope": 0.5}
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

### `gated_linear_ssm`

Simplified gated linear SSM with RMSNorm, separate gate / SSM / delta-t projections, learned temporal gating, scan recurrence, and residual add.

This block was historically named `mamba3`, but that config string is deprecated. Use `gated_linear_ssm` for the simplified gated scan. Canonical Mamba-3 is `mamba3-canonical`.

Required fields:

- `type: "gated_linear_ssm"`

Optional fields:

- `inner_dim`

Example:

```json
{"type": "gated_linear_ssm", "inner_dim": 512}
```

### `mamba3-canonical`

Canonical Mamba-3 block with RMSNorm, optional causal depthwise convolution, low-rank delta/lambda/theta projections, grouped MIMO B/C projections, complex-pair rotation, exponential-trapezoidal recurrence, gated output projection, and residual add.

Required fields:

- `type: "mamba3-canonical"`

Optional fields:

- `inner_dim` — SSM width; defaults to `model_dim`.
- `state_size` — recurrent state expansion; defaults to `16` and must be even.
- `n_groups` — grouped B/C projection count; defaults to `4`.
- `dt_rank` — low-rank projection width for delta/lambda/theta; defaults to `max(inner_dim/16, 1)`.
- `conv_kernel` — causal depthwise convolution width; defaults to `4`.
- `use_conv` — whether to keep the short causal convolution; defaults to `true`.
- `scan_chunk_size` — exact affine scan chunk size; defaults to `64`. `0` uses the original full-sequence parallel scan, mainly for debugging.

Example:

```json
{"type": "mamba3-canonical", "inner_dim": 512, "scan_chunk_size": 64}
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

### `gated_deltanet`

Linear-time gated DeltaNet block (Yang et al., 2025): pre-norm RMS, multi-head Q/K/V projections with optional shared K/V, short 1-D conv on Q/K/V, sigmoid output gate, and a chunked delta-rule recurrence with per-channel gating. The recurrent state is matrix-valued (`d_k × d_v` per head) and updated via a chunked associative scan; on CUDA the inner triangular solve uses a custom precompiled kernel (`gpu/cuda_kernels/gated_delta_chunk_solve.cu`).

Required fields:

- `type: "gated_deltanet"`
- `heads` — number of GLA heads.
- `d_k` — key/query dim per head. Total key dim is `heads * d_k`.

Optional fields:

- `d_v` — value dim per head. Defaults to `2 * d_k`. Total value dim is `heads * d_v`.
- `kv_share` — when `true` (default), the K and V projections share a single `[D, heads*d_v]` weight (V projection is reused for K, with `d_v >= d_k` required). When `false`, K and V get separate projections of width `heads*d_k` and `heads*d_v` respectively. The shared form is the recipe used by Yang et al. and saves one projection matrix per block.
- `scan_chunk_size` — chunk size for the chunked delta scan. When omitted, defaults to `64`, the Metal-tested safe chunk width. `0` explicitly uses the naive per-step scan (slower but simpler; useful for debugging). Positive values enable the chunked scan with the custom CUDA kernel when available on CUDA; larger values such as `128` or `256` are explicit performance experiments until validated on the target backend. Must be `>= 0`.
- `parallel_residual` — when `true`, fuses this block with the immediately following `swiglu` block into a parallel residual pair. See [`parallel_residual`](#parallel_residual).

Example:

```json
{"type": "gated_deltanet", "heads": 4, "d_k": 64, "d_v": 128, "kv_share": true, "scan_chunk_size": 64}
```

GLA-pair stacked with parallel residual:

```json
{"blocks": [
  {"type": "gated_deltanet", "heads": 4, "d_k": 64, "parallel_residual": true},
  {"type": "swiglu"}
]}
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
    {"op": "gelu", "inputs": ["gate"], "output": "gate_act"},
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

## Data

The `data` object controls loader-level corpus traversal.

| Field | Type | Required | Default | Notes |
|------|------|----------|---------|-------|
| `no_shard_shuffle` | boolean | No | `false` | When false, loaders keep the historical seeded whole-shard file shuffle. When true, loaders skip whole-shard file-order shuffling and consume shards in sorted glob order while still shuffling token chunks within each shard. |

## Training

The `training` object controls optimization, batching, and stochastic settings.

| Field | Type | Required | Default | Notes |
|------|------|----------|---------|-------|
| `steps` | integer | No | `200` | Total training steps. Must be `> 0`. |
| `lr` | number | No | `3e-4` | Base learning rate. Must be `> 0`. |
| `objective` | string | No | `"causal"` | Training objective: `"causal"`, `"mlm"`, `"mntp"`, or `"hybrid"`. Existing configs default to causal next-token training. |
| `mlm_mask_prob` | number | No | `0.15` | Probability of selecting each eligible token for masked-objective loss. Must be in `[0,1]`. |
| `mlm_mask_token_id` | integer | Required for `mlm`, `mntp`, `hybrid` | None | Token id used as the mask replacement. Must be in `[0, vocab_size)`. |
| `mlm_mask_token_prob` | number | No | `0.8` | Probability that a selected MLM token is replaced with `mlm_mask_token_id`. |
| `mlm_random_token_prob` | number | No | `0.1` | Probability that a selected MLM token is replaced with a random token id. |
| `mlm_kept_unchanged_prob` | number | No | `0.1` | Probability that a selected MLM token is kept unchanged. The three replacement probabilities must sum to `1.0`. |
| `hybrid_clm_fraction` | number | No | `0.5` | For `objective: "hybrid"`, deterministic per-batch probability of using causal next-token training. Must be in `[0,1]`. |
| `hybrid_secondary_objective` | string | No | `"mntp"` | Secondary objective for hybrid training: `"mlm"` or `"mntp"`. |
| `phases` | array | No | Disabled | Optional phase schedule. When non-empty, `steps` is computed as the sum of phase `steps` and top-level `steps`/`lr` are ignored by the training loop. Each phase must define `steps > 0` and `lr > 0`. |
| `warmdown_steps` | integer | No | `0` | Cosine warmdown length at the end of training. Must be `>= 0`; values above `steps` are clamped by the scheduler. |
| `target_val_loss` | number | No | `0` | Early-stop threshold on validation loss. `0` disables it. Must be `>= 0`. Checked when validation loss is computed during training. |
| `first_byte_mask` | boolean | No | `false` | Enables training-time first-byte masked softmax. At UTF-8 codepoint-boundary targets, the optimization loss only normalizes over vocabulary entries whose first byte is syntactically valid as a UTF-8 first byte. Validation, exported per-token NLL, and BPB evaluation use the unmasked loss. Byte-id corpora (`vocab_size <= 256`) work directly; BPE corpora use `tokenizer.json` next to the training shards. |
| `recurrence_activation_frac` | number | No | `0` | Delays model-level `recurrence` execution until `floor(frac * total_steps)`. Before activation, training executes each recurrence root once in first-appearance order while preserving the full recurrence weight layout. Mutually exclusive with `recurrence_activation_step` and top-level `recurrence_phases`; must be in `[0,1]`. |
| `recurrence_activation_step` | integer | No | `0` | Delays model-level `recurrence` execution until this absolute training step. Before activation, training executes each recurrence root once in first-appearance order while preserving the full recurrence weight layout. Mutually exclusive with `recurrence_activation_frac` and top-level `recurrence_phases`; must be `>= 0`. |
| `ttt_steps` | integer | No | `0` | Test-time training updates per validation batch during eval mode and full BPB eval. `0` disables TTT. In `ttt_mode="full"` this is score-first full-weight adaptation; in `ttt_mode="lora"` this is per-batch LoRA adaptation before scoring. Must be `>= 0`. |
| `ttt_mode` | string | No | `"full"` | TTT implementation: `"full"` keeps score-first full-weight fine-tuning, `"lora"` uses per-batch Low-Rank Adaptation at test time and discards adapters after each batch. Must be `"full"` or `"lora"`. |
| `ttt_lr` | number | No | `1e-5` | Learning rate for TTT updates. Must be `>= 0`; keep much smaller than training `lr`. |
| `ttt_rank` | integer | No | `4` | LoRA rank used when `ttt_mode="lora"`. Rank-2-or-higher weights get temporary adapters `W + A @ B` with `A:[M,R]`, `B:[R,N]`; only `A` and `B` train during LoRA-TTT. Must be `> 0`. |
| `hardware_tflops` | number | No | `0` | Peak hardware TFLOPS used to log MFU next to `tok/s`. `0` disables MFU logging. Must be `>= 0`. |
| `optimizer` | string | No | `"muon"` | Optimizer for matrix (rank ≥ 2) weights: `"muon"`, `"muon_eq_r"`, `"normuon"`, or `"adamw"`. `muon_eq_r` applies post-orthogonal row-L2 normalization; `normuon` applies NorMuon neuron-wise second-moment normalization after orthogonalization. Embed, head, and scalar groups always use AdamW. |
| `qat` | string | No | `"none"` | Quantization-aware training mode for rank-2 weights during the training forward pass. `"none"` disables it, `"int8"` applies per-row fake int8 quantization, and `"int6"` applies a coarser fake quantization with STE. |
| `weight_init` | string | No | `"xavier_uniform"` | Initialization for rank ≥ 2 weights: `"xavier_uniform"` or `"normal"`. 1D weights are always ones (norms) or zeros. |
| `weight_init_std` | number | No | `0.02` | Standard deviation for `"normal"` initialization. Ignored when `weight_init` is `"xavier_uniform"`. |
| `grad_clip` | number | No | `0` | Max grad norm. `0` means no clipping. Must be `>= 0`. |
| `weight_decay` | number | No | `0.01` | Global fallback weight decay. Must be `>= 0`. |
| `cautious_weight_decay` | boolean | No | `false` | When true, applies weight decay only to elements where parameter and gradient signs agree. This is an optimizer modifier for AdamW, Muon, MuonEq-R, NorMuon, and SGD paths, not a separate optimizer kind. |
| `cautious_weight_decay_activation_frac` | number | No | `0` | Fraction of training before cautious weight decay activates. Before activation, standard weight decay is used. `0` means active from step 0 when `cautious_weight_decay=true`; must be in `[0,1]`. |
| `beta1` | number | No | `0.9` | AdamW beta1. Also seeds Muon momentum when `muon_momentum` is omitted. |
| `beta2` | number | No | `0.95` | AdamW and Muon beta2. |
| `epsilon` | number | No | `1e-8` | AdamW / Muon epsilon. |
| `seed` | integer | No | `42` | RNG seed. `0` is treated as omitted and replaced with `42`. |
| `batch_tokens` | integer | No | `1024` | Tokens per optimization step. Must be divisible by `seq_len`. |
| `shuffle_chunk_tokens` | integer | No | `seq_len` | Token-block shuffle granularity for train/validation loaders. Values `<= 0` inherit `seq_len`; set to `2048` to reproduce the previous fixed-block behavior. |
| `embed_lr` | number | No | `lr` | Learning rate for embedding-class weights. |
| `matrix_lr` | number | No | `lr` | Learning rate for matrix weights. Used with Muon. |
| `scalar_lr` | number | No | `lr` | Learning rate for scalar and vector weights. |
| `head_lr` | number | No | `lr` | Learning rate for the output head. Ignored when `tie_embeddings=true` unless `mtp.untie_embed_at_frac < 1` reserves and later activates a separate head. |
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

### Training objectives

`objective: "causal"` preserves the existing shifted next-token objective. `objective: "mlm"` masks selected input positions and trains only those positions to predict the original token. `objective: "mntp"` predicts next tokens from bidirectional context while masking the corresponding next-token input position so the answer is not visible. `objective: "hybrid"` deterministically chooses causal or the configured secondary objective once per batch from `training.seed` and the step index.

Masked objectives emit a training loss averaged only over rows where `loss_mask > 0`; the dense `eval_loss` and `per_token_nll` outputs remain available for evaluation/export paths. Top-level `mtp` and `training.first_byte_mask` are not supported with non-causal objectives in this version.

```json
{
  "blocks": [
    {"type": "plain", "heads": 8, "attention_mask": "bidirectional"},
    {"type": "swiglu"}
  ],
  "training": {
    "objective": "hybrid",
    "mlm_mask_prob": 0.15,
    "mlm_mask_token_id": 103,
    "mlm_mask_token_prob": 0.8,
    "mlm_random_token_prob": 0.1,
    "mlm_kept_unchanged_prob": 0.1,
    "hybrid_clm_fraction": 0.5,
    "hybrid_secondary_objective": "mntp"
  }
}
```

### Optimizer groups

The trainer classifies weights into four optimizer groups:

| Group | Optimizer | Typical weights | LR field | Weight-decay field |
|------|-----------|-----------------|----------|--------------------|
| Embedding | AdamW | `embed`, `bigram_table` | `embed_lr` | `embed_weight_decay` |
| Head | AdamW | `head` | `head_lr` | `head_weight_decay` |
| Scalar | AdamW | Norm scales, decay vectors, learned scalar scales | `scalar_lr` | `scalar_weight_decay` |
| Matrix | Muon | Projection and FFN matrices | `matrix_lr` | `matrix_weight_decay` |

### Training phases

Use `training.phases` to run multiple contiguous LR segments in one job. Each
phase applies its `lr` for its own `steps`, and the total training length is
the sum of all phase steps.

When `phases` is present:

- top-level `steps` and `lr` are ignored by the training loop
- `warmdown_steps` still applies, but only within the final phase
- the trainer logs phase transitions using `label` when provided

Example:

```json
{
  "training": {
    "phases": [
      {"steps": 100, "lr": 0.0001, "label": "warmup"},
      {"steps": 4000, "lr": 0.001, "label": "main"},
      {"steps": 900, "lr": 0.0001, "label": "cooldown"}
    ],
    "batch_tokens": 16384,
    "seed": 42
  }
}
```

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

## Eval

The optional `eval` object controls evaluation-only adaptation. When omitted,
or when `ttt_mode` is omitted or `"none"`, evaluation runs as the standard
single pass.

| Field | Type | Required | Default | Notes |
|------|------|----------|---------|-------|
| `ttt_mode` | string | No | `"none"` | Eval-time TTT mode. `"none"` disables eval-time adaptation. `"legal_chunk_sgd"` enables score-first chunk SGD. |
| `chunk_tokens` | integer | No | `32768` when enabled | Number of validation tokens per adaptation chunk. Must be `> 0`, at least `training.batch_tokens`, and divisible by `training.batch_tokens`. |
| `ttt_epochs` | integer | No | `3` when enabled | SGD epochs over each already-scored chunk. Must be `> 0`. |
| `ttt_lr` | number | No | `0.005` when enabled | Base SGD learning rate. Must be `> 0`. |
| `ttt_momentum` | number | No | `0.9` when enabled | SGD momentum. Must be in `[0,1)`. Set `0.0` for plain SGD without momentum. |
| `ttt_lr_schedule` | string | No | `"cosine"` when enabled | LR schedule across chunks: `"cosine"` uses `0.5 * (1 + cos(pi * i / N)) * ttt_lr`; `"constant"` keeps `ttt_lr`. |

Legal score-first TTT scores each chunk before adapting on it. Adapted weights
then carry into the next validation chunk, but they stay in memory only and are
never written back to safetensors.

```json
{
  "eval": {
    "ttt_mode": "legal_chunk_sgd",
    "chunk_tokens": 32768,
    "ttt_epochs": 3,
    "ttt_lr": 0.005,
    "ttt_momentum": 0.9,
    "ttt_lr_schedule": "cosine"
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

Fuses a `(plain or gated_deltanet, swiglu or geglu)` pair into a parallel residual: both branches read the same pre-norm input and their outputs sum into the residual together (instead of the GLU branch reading the post-attention residual).

Two forms are supported:

**Top-level (uniform).** Apply to every block pair:

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

When the top-level flag is `true`, every consecutive pair must be `(plain or gated_deltanet, swiglu or geglu)`.

**Per-block (scoped).** Set `parallel_residual: true` on individual pair-start blocks; the top-level flag stays `false` (default). Pairs not marked stay sequential. This lets you mix paired and unpaired layers in the same model — e.g., the leaderboard pattern where only late attention layers use parallel residual:

```json
{
  "blocks": [
    {"type": "gated_deltanet", "heads": 4, "d_k": 64},
    {"type": "swiglu"},
    {"type": "gated_deltanet", "heads": 4, "d_k": 64},
    {"type": "swiglu"},
    {"type": "plain", "heads": 4, "parallel_residual": true},
    {"type": "swiglu"},
    {"type": "plain", "heads": 4, "parallel_residual": true},
    {"type": "swiglu"}
  ]
}
```

Effect:

- A pair start must be `plain` or `gated_deltanet`; the next block must be `swiglu` or `geglu`.
- Each paired GLU block drops its `ffn_norm_scale` weight (the pair shares the start block's RMSNorm).
- `kv_source` is not supported on a paired `plain` block.
- Cannot be combined with `unet`.
- **Weight sharing:** when using `recurrence` (or `weight_group`) to share weights between blocks, paired and unpaired GLU blocks cannot share weights with each other — they have different shapes (paired blocks lack `ffn_norm_scale`). Use distinct group names for paired vs. unpaired roles.

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

- `plain`, `swiglu`, `geglu`, `mlp`, and `cross_attention` expand to `round(model_dim * mlp_mult)`, clamped to at least `model_dim`.

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
    "ttt_mode": "lora",
    "ttt_steps": 1,
    "ttt_lr": 1e-5,
    "ttt_rank": 4
  }
}
```

See `examples/recurrent_parallel.json` for the runnable version with inline
comments, and `examples/unet_transformer.json` for the U-Net path.
