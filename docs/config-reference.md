# mixlab JSON Config Reference

This reference covers the current JSON schema used by `mixlab`, based on:

- `arch/config.go` for top-level and training fields
- `arch/ir_bridge.go` for config-to-IR conversion
- `arch/builder.go`, `arch/weight_shapes.go`, and `arch/custom.go` for block behavior, defaults, and custom-block semantics

All examples below are valid JSON fragments unless otherwise noted.

For a shorter path through the schema, start with:

- [Config: Model Basics](config-model.md)
- [Config: Blocks](config-blocks.md)
- [Config: Training](config-training.md)
- [Config: Advanced Features](config-advanced.md)

Check a config without initializing MLX or loading data:

```bash
mixlab -mode validate -config model.json
```

## Top-level model fields

These fields live at the root of the config object.

For Hugging Face directory export, see [Hugging Face Export](hf-export.md). The current `export-hf` core path supports causal and masked-LM heads for supported sequential `plain` attention stacks plus `swiglu`/`geglu`/`mlp`/`moe` blocks, cache-safe causal `ttt_mlp` stacks, non-cached canonical Mamba-3 causal/classification stacks, trained native classification heads, GQA, `qk_norm`, `qk_gain`, causal windowing, mask variants, DeBERTa relative attention, `plain` gated FFN tails, configurable RMSNorm/LayerNorm for core GPT-style blocks, and embedding feature channels; unsupported blocks or config features fail explicitly.

| Field | Type | Required | Default | Notes |
|------|------|----------|---------|-------|
| `name` | string | No | Source filename/path | Human-readable run name. |
| `model_dim` | integer | Yes | None | Hidden size `D`. Must be `> 0`. |
| `vocab_size` | integer | Token inputs only | None | Token vocabulary size `V`. Must be `> 0` and `<= 65535` for `token_embedding`. Omit it for `input_adapter.kind: "linear_frames"` or `"discrete_codebooks"`. |
| `seq_len` | integer | No | `128` | Context length in tokens. Must be `> 0` when set. |
| `input_adapter` | object | No | `{"kind":"token_embedding"}` | Selects the model input representation. `linear_frames` accepts float32 `[B,T,F]`. `discrete_codebooks` accepts int32 `[B,T,Q]`, requires `num_codebooks >= 1` and `codebook_vocab_size >= 2`, and supports `fusion: "attention_mlp"` (default) or `"mean"`; `fusion_hidden_dim` defaults to `model_dim`. Both non-token adapters support `norm: "none"|"layernorm"` and native classification only in v1. |
| `mlp_mult` | number | No | `2.67` | FFN expansion multiplier for `plain`, `swiglu`, `geglu`, `mlp`, `moe` experts, and `cross_attention` FFN tails. Must be `> 0`. |
| `logit_softcap` | number | No | Disabled | Optional soft cap applied to output logits before loss/export. |
| `smear_embeddings` | boolean | No | `false` | Enables 1-token-lookback smearing on token embeddings before the first block. |
| `smear_embeddings_gate_shape` | string | No | `"pr130"` when smearing is enabled | Gate variant for `smear_embeddings`: `"pr130"`, `"per_channel"`, or `"per_position_per_channel"`. |
| `char_vocab_size` | integer | No | Disabled | Enables tokenizer-level byte/char feature embeddings when `>= 257`. `0` disables. Padding id `0` is reserved; ByteLevel byte values map to ids `1..256`. |
| `char_dim` | integer | No | `model_dim` when char features enabled | Character feature embedding dimension. If different from `model_dim`, a learned projection maps the summed feature bag to the model dimension. |
| `char_max_per_token` | integer | No | `16` when char features enabled | Fixed sparse char slots per token id. Must be `> 0` when char features are enabled. |
| `positional_embedding` | string | No | `"rope"` | Model-level position mode. `"rope"` preserves legacy rotary attention in non-relative `plain` blocks. `"learned_absolute"` adds `position_embeddings: [max_positions, model_dim]`, disables RoPE in non-relative `plain` blocks, and rejects explicit RoPE/relative-position fields in v1. `"none"` disables RoPE without adding a learned table. |
| `max_positions` | integer | No | `seq_len` | Row count for learned absolute position embeddings and exported HF max-position metadata. Must be `>= seq_len`. |
| `embedding_dropout` | number | No | `0` | Training-only inverted dropout applied after token/position/feature embedding channels are summed and before block 0. Eval, generation, parity, and HF export run with dropout disabled. Must be in `[0,1]`. |
| `hf_export_format` | string | No | `"mixlab"` | Hugging Face export format. `"mixlab"` writes the maintained custom-code Mixlab model. `"gpt2"` writes a native `model_type: "gpt2"` directory and is accepted only for strict GPT-2-compatible sequential `plain` stacks. |
| `bigram_vocab_size` | integer | No | Disabled | Enables model-level hashed bigram embeddings when `> 1`. `0` disables. `1` is invalid. |
| `bigram_dim` | integer | No | `model_dim` when bigrams enabled | Bigram embedding dimension. `0` inherits `model_dim`. Ignored when `bigram_vocab_size == 0`. |
| `trigram_vocab_size` | integer | No | Disabled | Enables model-level hashed trigram embeddings when `> 1`. `0` disables. Trigram IDs are computed within each sequence row and never cross row boundaries. |
| `trigram_dim` | integer | No | `model_dim` when trigrams enabled | Trigram embedding dimension. `0` inherits `model_dim`. Ignored when `trigram_vocab_size == 0`. |
| `tie_embeddings` | boolean | No | `false` | Shares token embedding and output head weights. |
| `dropout` | number | No | `0` | Legacy training dropout applied to both hidden/residual projections and attention probabilities unless `hidden_dropout` or `attn_dropout` override it. Must be in `[0,1]`. Eval, generation, parity, and HF export run with dropout disabled. |
| `hidden_dropout` | number | No | `dropout` | Training dropout on attention output projections, FFN output projections, and MoE deltas. Explicit `0` disables hidden dropout even when `dropout` is nonzero. |
| `attn_dropout` | number | No | `dropout` | Training dropout on `plain` attention probabilities after softmax and before multiplying by values. Explicit `0` disables attention-probability dropout even when `dropout` is nonzero. |
| `tie_dropout` | boolean | No | `false` | Samples each S4D internal/residual dropout mask as `[B,1,D]` and broadcasts it over sequence positions. Non-S4D blocks keep their normal dropout behavior. Block-level `tie_dropout` can enable the same behavior for one S4D block. |
| `mlm_head` | string | No | `"linear"` | Masked-objective prediction head. `"linear"` keeps the legacy bare LM head. `"bert"` uses `LayerNorm(affine=false) -> Linear(D,D)+bias -> GELU -> LayerNorm(affine=false) -> Dropout -> tied embedding output + bias` for MLM/MNTP/hybrid masked steps and `AutoModelForMaskedLM` export. Requires `tie_embeddings: true` and a masked objective path. |
| `layer_aggregation` | string | No | `"none"` | Optional dense weighted aggregation over static embeddings and previous sublayer outputs. `"dwa"` enables GPT-BERT-style DWA on supported sequential blocks. See [Dense Weighted Aggregation](#dense-weighted-aggregation-dwa). |
| `norm_type` | string | No | `"rmsnorm"` | Normalization used by supported blocks and the final model norm. `"rmsnorm"` preserves the legacy layout; `"layernorm"` emits LayerNorm; `"batchnorm"` enables native BatchNorm for fixed-shape classification. |
| `norm_eps` | number | No | `1e-5` | Epsilon for supported block/final norms. Must be `> 0`. |
| `norm_affine` | boolean | No | `true` | Whether LayerNorm uses learned scale/bias. `false` is supported for `norm_type: "layernorm"`; RMSNorm and BatchNorm remain affine-only. |
| `batchnorm_momentum` | number | No | `0.1` | Running-stat update momentum for `norm_type: "batchnorm"`. Must be in `(0,1]`; rejected for other norm types. |
| `norm_placement` | string | No | `"pre"` | Supported values are `"pre"`, `"post"`, `"post_residual"`, and `"sandwich"`. Existing `"post"` normalizes each sublayer delta before residual add. `"post_residual"` implements standard `Norm(x + F(x))` for S4D-only stacks. `"sandwich"` uses both pre-input and post-delta norms. |
| `final_norm` | boolean | No | `true` | Controls the model-level norm after the block stack. Set `false` for post-residual architectures whose reference omits a final norm. |
| `ffn_internal_norm` | boolean | No | `false` | Adds an internal norm to supported FFN paths before the down projection: after the activation/product in `swiglu`/`geglu`, after activation in `mlp`, and after the `plain` FFN tail activation. |
| `block_scales` | boolean | No | `false` | Adds learned per-channel scales to supported residual branches. `plain`, `swiglu`, `geglu`, and `moe` use them by default when enabled; recurrent branches can use them when `residual_scale_init` is set. |
| `resid_mix` | boolean | No | `false` | Adds learned mixing of the current state and original input on `plain` blocks. |
| `parallel_residual` | boolean | No | `false` | Top-level form: enables parallel residual on every consecutive `(plain or gated_deltanet, swiglu/geglu/moe)` pair. Per-block form: set `parallel_residual: true` on individual pair-start blocks instead (see [`parallel_residual`](#parallel_residual)). Use block-level `parallel_group` for heterogeneous groups of 2+ branches. Cannot be combined with `unet`. |
| `unet` | boolean | No | `false` | Splits the `blocks` list into encoder/decoder halves with learned skip connections. |
| `rc_equivariant` | boolean | No | `false` | Native DNA reverse-complement parameter sharing. `model_dim` is the per-orientation width; parameter count is unchanged and backbone compute is approximately doubled. V1 supports bidirectional MLM and mean-pooled classification with `plain`/`gated_deltanet` mixers and pointwise FFNs. Requires manifest-backed DNA data and zero dropout; rejects causal generation, HF export, RC augmentation, feature embeddings, recurrence, parallel composition, and auxiliary/multihead objectives. |
| `mtp` | object | No | Disabled | Enables parameter-free multi-token prediction during training. See [MTP section](#multi-token-prediction-mtp). |
| `backout` | object | No | Disabled | Enables final-latent residual subtraction before the final model norm. See [Backout section](#backout). |
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

### Continuous frame input

`input_adapter.kind: "linear_frames"` projects generic float features from
`[B,T,F]` to `[B,T,model_dim]` before the ordinary backbone. The adapter is
modality-neutral: waveform samples, numeric sensor channels, and genomic
tracks use the same representation. It adds `input_adapter_proj: [F,D]`, an
optional bias, and optional affine LayerNorm. No token embedding, vocabulary
head, or dummy vocabulary is allocated.

```json
{
  "name": "continuous-classifier",
  "model_dim": 32,
  "seq_len": 16000,
  "positional_embedding": "none",
  "input_adapter": {
    "kind": "linear_frames",
    "feature_dim": 1,
    "bias": true,
    "norm": "none"
  },
  "blocks": [
    {"type": "mamba3-canonical", "inner_dim": 32, "state_size": 8, "n_groups": 4},
    {"type": "swiglu"}
  ],
  "training": {
    "objective": "classification",
    "classification": {"num_labels": 10, "pooling": "last"},
    "batch_tokens": 16000
  }
}
```

Position handling uses the existing top-level `positional_embedding` policy;
there is intentionally no second `input_adapter.positions` field.
`"none"` is the normal raw-signal setting, while `"learned_absolute"` adds the
existing model-level position table. Continuous input requires a
`continuous_frames` dataset manifest with matching `feature_dim`, `seq_len`,
and classification label count. See [Continuous input](continuous-input.md).

For magnitude-bearing low-dimensional signals, keep `input_adapter.norm` at
`"none"`. LayerNorm follows the `F -> D` projection, so with `feature_dim: 1`
and the zero-initialized projection bias it cancels input magnitude exactly and
passes only the sign at initialization. Mixlab warns when post-projection
LayerNorm is selected with `feature_dim <= 4`; higher-dimensional use remains
available when scale invariance is intentional.

### Discrete codebook input

`input_adapter.kind: "discrete_codebooks"` consumes integer `[B,T,Q]` codec
tokens. It allocates one `[Q*V,D]` table and indexes each codebook through the
integer-safe offset `code + codebook*V`. The default `attention_mlp` fusion
uses `Linear(D,H) -> ReLU -> Linear(H,1,bias=false)`, softmax over `Q`, and a
weighted sum. `fusion: "mean"` allocates no attention weights.

`num_codebooks` and `codebook_vocab_size` are required. `fusion_hidden_dim`
defaults to `model_dim`; `norm` accepts `none` or `layernorm`. V1 requires
native classification, `vocab_size: 0`/omitted, and a matching
`discrete_codebooks` manifest. See
[Discrete codebook input](discrete-codebooks-input.md).

## Smear Token Embeddings

`smear_embeddings` mixes the previous token embedding into the current token embedding before flattening and before optional char/bigram/trigram residual embeddings. Position `0` is left unchanged, so the BOS/no-previous-token boundary is not trainable and cannot leak a wrapped final token into the first position.

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

## Token-Level Char Features

`char_vocab_size` enables a fixed sparse feature channel keyed by token id. For each token, the model gathers up to `char_max_per_token` rows of `char_table` indexed by precomputed integer ids, sums them, optionally projects the result to `model_dim`, multiplies it by learned `char_scale`, and adds it after the token embedding and before bigram/trigram feature channels. Id `0` is reserved padding and contributes zero.

The engine is tokenizer-agnostic: it consumes a `[vocab_size, char_max_per_token]` table of integer ids in `[0, char_vocab_size)` from a `char_features.bin` file located next to the training shards (for `train`/`eval`/`hiddenstats`) or next to the config/weights (for `generate` and distillation teachers).

### File format

`char_features.bin` is a 256-int32 little-endian header followed by `vocab_size * char_max_per_token` little-endian uint16 ids:

| Header index | Meaning |
|---|---|
| `0` | Magic (`20260526`) |
| `1` | Version (`1`) |
| `2` | `vocab_size` (must match config) |
| `3` | `char_vocab_size` (must match config) |
| `4` | `char_max_per_token` (must match config) |
| `5..` | Reserved for tooling; the engine does not interpret these |

The bundled `scripts/prepare.py` writes this format for HuggingFace ByteLevel BPE tokenizers — each token's constituent bytes become its char ids (offset by `+1` so id `0` stays reserved for padding):

```bash
mixlab -mode prepare -input data.txt -prepare-output-dir data/example \
  -vocab-size 1024 -char-vocab-size 257 -char-max-per-token 16
```

Other tokenizers can write their own `char_features.bin` by emitting a matching header + uint16 payload — the engine does not validate how the ids were derived.

## Dense Weighted Aggregation (DWA)

`layer_aggregation: "dwa"` enables GPT-BERT-style dense weighted aggregation. The model keeps an accumulator containing the static embedding state and every completed supported sublayer residual output. At aggregation point `k`, Mixlab adds a trainable `dwa_alpha_k` vector of length `k + 2`, initialized to all zeros except the last element set to `1.0`, and replaces the running hidden state with the weighted sum of the accumulator.

Aggregation points are:

- `plain`: after the attention residual and after the FFN residual
- `swiglu`, `geglu`, `mlp`, `moe`: after the block residual

V1 supports DWA on normal sequential `plain`/`swiglu`/`geglu`/`mlp`/`moe` stacks. It rejects recurrence, recurrence phases, custom execution order, U-Net, parallel residual, `kv_source`, `skip_attention`, and recurrent/custom block types until those paths have explicit parity coverage. DWA alpha weights are appended after normal model weights so existing base weight indices remain stable.

```json
{
  "layer_aggregation": "dwa",
  "blocks": [
    {"type": "plain", "heads": 6},
    {"type": "geglu"}
  ]
}
```

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

`backout` captures the residual stream after a configured physical block and subtracts a learned scalar-weighted copy immediately before the final model norm and LM head. When omitted, the emitted IR and weight layout stay unchanged.

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

Every block object requires a `type` field. The selected type determines which
additional block fields are valid.

### `plain`

Self-attention block with configurable model norm, RoPE, grouped-query support via `kv_heads`, configurable FFN tail, and residual connections. It defaults to causal attention for causal training objectives and bidirectional attention for current masked objective graphs.

Required fields:

- `type: "plain"`
- `heads`

Optional fields:

- `kv_heads` — grouped-query attention (must divide `heads` evenly)
- `qk_norm` — learned RMSNorm scales applied to each Q and K head after projection/GQA expansion and before RoPE or relative-attention score construction. Adds `q_norm_scale` and `k_norm_scale` weights of shape `[head_dim]`; a `kv_source` consumer block adds only `q_norm_scale` because it reuses K from the source block.
- `qk_gain` — learnable per-head QK scaling. When set, allocates one trainable scalar per head initialized to this value, applied as `scores = qk_gain * (Q @ K^T / sqrt(d_k))`. Omit or set to `0` for standard scaling.
- `differential_attention` — enable DIFF Transformer attention. `heads` is the number of differential heads; for a baseline with `N` ordinary heads, use `N/2` differential heads. Each differential head splits Q/K into two sub-maps, computes two masked softmax maps, subtracts the second with learned shared-per-layer λ, applies per-head RMSNorm scaled by `(1 - lambda_init)`, then projects with `wo`.
- `differential_lambda_init` — optional DIFF λ override for this block. Omit to use the depth schedule `0.8 - 0.6 * exp(-0.3 * layer_depth)`, where the first block has `layer_depth=1`.
- `attn_bias` — add learned zero-initialized biases to the Q/K/V/O projections. With `relative_attention_parameterization: "shared_qk_reuse"`, the same Q/K biases are reused when projecting the shared relative embedding table.
- `attn_value_gate` — widen the V projection to emit both values and a GELU gate. The gate multiplies the merged attention output before the output projection. This is compatible with grouped-query attention by keeping the value slice at `kv_heads * head_dim` and the gate slice at `model_dim`; it cannot be used on a `kv_source` consumer block.
- `attn_post_norm` — attention residual normalization placement. Omit or set to `"inherit"` to preserve the global `norm_placement` behavior: post/sandwich configs apply `post_attn_norm` after the output projection, while pre-norm configs do not add attention post-norm. Set to `"before_outproj"` to normalize the merged attention state before `wo`, set to `"after_outproj"` to force the legacy post-output placement, or set to `"none"` to disable attention post-norm for that block.
- `rope_dims` — partial RoPE: apply rotary embeddings to only the first `rope_dims` dimensions per head, leaving the rest position-invariant. Must be even and `<= head_dim`. Omit or set to `0` for full RoPE.
- `rope_convention` — rotary pairing convention. Omit or set to `"adjacent_pair"` for Mixlab's default adjacent-dimension pairs `(0,1),(2,3),...`; set to `"half_rotation"` for the split-half convention used by some Hugging Face models. Only applies to standard RoPE attention and is rejected with `relative_attention`.
- `relative_attention` — `"deberta_p2c_c2p"` enables DeBERTa/GPT-BERT-style disentangled content-to-position and position-to-content relative attention bias. Omit, set to `""`, or set to `"none"` for standard RoPE attention.
- `relative_attention_window` — position bucket size for DeBERTa/GPT-BERT relative attention. Defaults to `128` when `relative_attention` is enabled. The learned table has `2 * relative_attention_window - 1` rows, centered at `relative_attention_window - 1`, and relative positions use GPT-BERT log bucketing rather than linear clipping.
- `relative_attention_parameterization` — relative-attention weight layout. Omit, set to `""`, or set to `"per_block_projections"` for the legacy Mixlab layout where every relative-attention block owns `relative_embeddings`, `w_pos_key`, and `w_pos_query`. Set to `"shared_qk_reuse"` for GPT-BERT-style sharing: the model owns one `shared_relative_embeddings` table and each block reuses its own `wk` and `wq` projections for position keys/queries. Shared mode requires all participating blocks to use the same effective `relative_attention_window`.
- `relative_attention_embedding_norm` — optional model-level norm for `shared_qk_reuse`. Omit or set to `"none"` for no normalization. Set to `"layernorm"` to apply one affine LayerNorm, using the configured `norm_eps`, to the shared relative embedding table before each block reuses its Q/K projections. All shared relative-attention blocks must use the same value.
- `ffn_activation` — FFN tail activation inside the `plain` block. Omit or set to `"silu"` for the legacy `ff1 -> SiLU -> ff2` tail. Set to `"gelu_new"` for GPT-2 tanh-approx GELU, `"gelu"` for exact erf GELU, `"geglu"` for `ff_gate -> GELU` multiplied by `ff1`, or `"swiglu"` for `ff_gate -> SiLU` multiplied by `ff1`, then `ff2`. Gated tails add one `ff_gate` matrix of shape `[model_dim, round(model_dim * mlp_mult)]`.
- `ffn_pre_norm` — add a second pre-FFN norm after the attention residual and before the `plain` FFN tail. This is required for strict GPT-2-style blocks and composes with `norm_type: "layernorm"`.
- `ffn_bias` — add zero-initialized learned biases to `ff1` and `ff2`. Use with `attn_bias` and `ffn_pre_norm` for GPT-2-compatible block tails.

The `plain` block's relative-attention operator matches the DeBERTa/GPT-BERT C2P/P2C bucket and index semantics. The default `per_block_projections` layout remains Mixlab's original bias-free architecture with per-block projected position tensors. `shared_qk_reuse` switches only the relative-attention parameterization to the GPT-BERT-style shared embedding table and Q/K projection reuse; use `attn_bias` and `attn_value_gate` independently when matching recipes that need biased projections or value gating.

DIFF attention is experimental and currently supports causal, bidirectional, and windowed causal masks plus RoPE applied independently to both Q/K sub-maps. V1 rejects `relative_attention`, `qk_norm`, `qk_gain`, `attn_value_gate`, `sparse_attn_gate`, `xsa`, `kv_source`, `kv_heads`, `parallel_group`, and attention post-norm. Explicit `rope_dims` is measured against the DIFF sub-head dimension, not the full differential head width. The two-softmax difference produces large early-training gradients, so set a finite `training.grad_clip` (e.g. `1.0`, as in the example); without gradient clipping the loss can diverge in the first few steps.
- `xsa` — eXplicit Subspace Attention: after computing `y = softmax(QK^T)V`, projects `y` orthogonal to `V` at each position. Forces attention to contribute information that V doesn't already provide. Zero additional parameters. Compatible with GQA.
- `attention_mask` — `"causal"`, `"bidirectional"`, or `"none"`. Omit to resolve from `training.objective`: causal objectives use `"causal"` and current masked objective graphs use `"bidirectional"`. For `training.objective: "hybrid"`, Mixlab ignores block-level `attention_mask` during training and chooses the mask from the concrete per-batch objective: causal batches use `"causal"` and MLM/MNTP batches use `"bidirectional"`. For `training.objective: "block_diffusion"`, `plain` attention uses a prefix-plus-block mask (committed prefix attends causally, the active block attends bidirectionally within itself and over the prefix) and `window_size` is rejected rather than approximating that rule. `"bidirectional"` and `"none"` both use dense softmax with no triangular mask.
- `window_size` — sliding causal attention width. `0` means full causal attention. Valid only when the resolved `attention_mask` is `"causal"`.
- `kv_source` — reuse K/V projections from an earlier block (1-indexed). Saves 2 weight matrices per shared block. Source must be an earlier `plain` block with matching `heads` and `kv_heads`. Not supported with `parallel_residual` or `parallel_group`.
- `parallel_residual` — when `true`, fuses this block with the immediately following FFN block into a legacy parallel residual pair. See [`parallel_residual`](#parallel_residual).
- `parallel_group` — when set on the first block, runs that many contiguous blocks as a shared-input parallel group. V1 supports `plain`, `gated_deltanet`, and `hgrn2` token mixers plus an optional final `swiglu`, `geglu`, or `moe` FFN branch.
- `residual_scale_init` — explicit init for this block's residual scale tensor(s), such as `0.0` for a zero-gated branch. Requires top-level `block_scales: true`.

Example:

```json
{"type": "plain", "heads": 8, "kv_heads": 4, "qk_norm": true, "qk_gain": 5.25, "rope_dims": 16, "xsa": true}
```

DeBERTa relative attention example:

```json
{"type": "plain", "heads": 8, "relative_attention": "deberta_p2c_c2p", "relative_attention_window": 128}
```

GPT-BERT-style shared relative embedding example:

```json
{"type": "plain", "heads": 8, "relative_attention": "deberta_p2c_c2p", "relative_attention_window": 32, "relative_attention_parameterization": "shared_qk_reuse"}
```

Differential attention example:

```json
{"type": "plain", "heads": 4, "attention_mask": "bidirectional", "differential_attention": true}
```

KV sharing example:

```json
{"blocks": [
  {"type": "plain", "heads": 8, "kv_heads": 4},
  {"type": "swiglu"},
  {"type": "plain", "heads": 8, "kv_heads": 4, "kv_source": 1}
]}
```

Notes:

- DeBERTa relative attention is mutually exclusive with `rope_dims` in v1.
- `kv_source` cannot be used by a relative-attention block, and a relative-attention block cannot be used as a KV source.
- `shared_qk_reuse` uses one model-level relative embedding table; all shared relative-attention blocks must use the same effective `relative_attention_window`.

### `swiglu`

Feed-forward-only block with configurable model norm, SwiGLU gating, and residual connection.

Required fields:

- `type: "swiglu"`

Optional fields:

- None

Example:

```json
{"type": "swiglu"}
```

### `geglu`

Feed-forward-only block with configurable model norm, GEGLU gating, and residual connection.
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

Feed-forward-only block with configurable model norm, one up projection, configurable activation, one down projection, and residual connection.

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

### `moe`

Sparse feed-forward block with one RMSNorm, a bias-free linear router, top-k token routing, per-expert FFN weights, residual add, and an auxiliary load-balancing loss. V1 experts are feed-forward blocks only: `swiglu`, `geglu`, or `mlp`.

Required fields:

- `type: "moe"`
- `num_experts`

Optional fields:

- `top_k` — number of experts selected per token. Defaults to `min(2, num_experts)`.
- `expert_block` — expert FFN spec. Defaults to `{"type": "swiglu"}`. Supported types are `swiglu`, `geglu`, and `mlp`.
- `router` — router type. Omitted/`"linear"` is supported.
- `load_balance_loss_weight` — coefficient for the auxiliary load-balancing loss. Defaults to `0.01`; set `0.0` to report routing stats without adding the aux loss to training loss.

Weight layout:

- `moe_norm_scale`: `[D]`
- `router_w`: `[D, num_experts]`
- per-expert FFN weights in expert index order
- optional `moe_scale`: `[D]` when `block_scales` is enabled

Routing uses softmax over all router logits, selects the top-k experts per token, renormalizes selected probabilities to sum to `1`, combines selected expert outputs by those probabilities, and drops no tokens. The emitted training `loss` adds the weighted MoE auxiliary loss; `eval_loss` and `per_token_nll` remain task-loss values. `mode count` reports total parameters and active parameters per token for MoE configs, and FLOP estimates count router cost plus top-k active experts.

Examples:

```json
{"type": "moe", "num_experts": 4, "top_k": 2, "expert_block": {"type": "swiglu"}}
```

```json
{"type": "moe", "num_experts": 4, "top_k": 1, "expert_block": {"type": "mlp", "activation": "relu"}}
```

### `legacy_mamba`

Compatibility-only gated fixed-decay recurrence. This implementation was
formerly exposed as `mamba`, but it is not a reference Mamba implementation:
its "convolution" is a same-position dense projection and its scan uses a
fixed learned per-channel decay rather than Mamba's input-dependent selective
state-space parameters.

The ambiguous `type: "mamba"` spelling is retired and fails validation. To
load an existing Mixlab checkpoint, change only its config block type from
`mamba` to `legacy_mamba`; the weight layout is unchanged. For a
paper-aligned implementation, use `mamba3-canonical`.

Required fields:

- `type: "legacy_mamba"`

Optional fields:

- `inner_dim`

Example:

```json
{"type": "legacy_mamba", "inner_dim": 768}
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
- `dt_min` — lower initialization bound for the learned time step; defaults to `0.001`.
- `dt_max` — upper initialization bound for the learned time step; defaults to `0.1` and must be greater than `dt_min`.
- `bidirectional` — shared-weight two-direction mixing for classification, MLM, or MNTP. Mixlab reverses only each row's valid prefix, sums forward and backward deltas, and adds the residual once. Parameter count is unchanged and mixer FLOPs are approximately doubled. Native-only in v1.

Example:

```json
{"type": "mamba3-canonical", "inner_dim": 512, "scan_chunk_size": 64}
```

### `s4d`

Linear time-invariant diagonal state-space token mixer with a closed-form
kernel and differentiable FFT-convolution training path. Pair it with a
`swiglu` or `geglu` block for channel mixing.

Required fields:

- `type: "s4d"`

Optional fields:

- `state_size` - full conjugate-pair state size per channel. Defaults to `64`
  and must be positive and even.
- `init` - initialization family. Omit or set to `"s4d-lin"` for
  `A_n = -1/2 + i*pi*n`. Other families, including S4D-LegS, are rejected in
  v1 rather than approximated.
- `freq_scale` - finite positive multiplier for the imaginary S4D-Lin pole
  initialization, `A_n = -1/2 + i * freq_scale * pi * n`. Defaults to `1.0`
  and does not add a checkpoint tensor.
- `sobolev_filter` - optional trainable transfer-function filter. Omit or set
  `false` to disable it; `true` enables reference defaults. The object form
  accepts `beta_init` (default `0.0`), `learning_rate` (default `0.01`),
  `trainable` (default `true`), `weight_decay` (default `0.0`), and
  `granularity` (`"channel"` by default or `"layer"`). `trainable:false`
  checkpoints and counts beta but excludes it from optimizer updates; its
  ignored `learning_rate` may then be zero. Channel
  granularity stores `[model_dim]`; layer granularity stores `[1]` and
  broadcasts it over features. Optional `bounds:[min,max]` uses
  `mid + half_range*tanh(raw_beta)` and requires `beta_init` strictly inside
  the interval. It applies
  `(1 + frequency_bin/fft_length)^beta` to the convolution spectrum, and leaves
  the direct `D*x` term unchanged. Omitted controls retain channel-wise,
  trainable, zero-decay reference behavior.
- `dt_min` - lower bound for the per-channel log-uniform time-step
  initialization. Defaults to `0.001`.
- `dt_max` - upper bound for the per-channel log-uniform time-step
  initialization. Defaults to `0.1` and must exceed `dt_min`.
- `bidirectional` - when `true`, learns independent forward/backward `C`
  tensors while sharing `A`, `B`, and `dt`. The two length-`T` kernels are
  combined into the official length-`2T` circular FFT kernel, including its
  one-position reverse offset. Output width remains `model_dim`.
  Bidirectional S4D is accepted only for classification, MLM, or MNTP; causal
  objectives reject it to prevent future-token leakage. Padded classification
  inputs are masked before convolution and after the residual path.
- `n_ssm` - independent `A`/`B` groups. Omitted uses one group per model
  channel, preserving the legacy layout. Must divide `model_dim`. Shared groups
  follow the upstream interleaved channel mapping `channel % n_ssm`.
- `discretization` - `"zoh"` by default or `"bilinear"` for the pinned
  reference S4 implementation.
- `trainable_b` - learns complex `B` tensors. Omitted keeps fixed `B=1`.
- `state_lr` - optional dt/A/B learning rate. When present, S4D dt/A and
  trainable B use an adaptive optimizer at this rate with no decay. These
  structural parameters never receive weight decay, even when `state_lr` is
  omitted or `weight_decay_policy` is `"all"`; C/D use the global LR.
- `tie_dropout` - shares this block's internal and residual dropout masks over
  sequence positions.
- `output_transform` - omit or set `"none"` for the compact Mixlab path.
  Set `"glu"` to add the reference-style `D -> 2D` output projection, bias,
  and GLU after the S4D GELU activation.
- `residual_scale_init` - optional S4D residual-scale initialization when
  top-level `block_scales` is enabled.

Omitting the reference fields keeps fixed `B=1`, per-channel A, ZOH
discretization, and the legacy scalar optimizer grouping exactly. The normal
training path uses FFT convolution; the equivalent recurrent scan is an
internal parity-test path, not a public execution mode or generation cache.
When `state_lr` is set, dt/A/B use that rate without decay, and C/D use global
LR and the selected decay policy. Sobolev
beta uses its configured learning rate and decay. Because the filter needs
the complete FFT, it is not supported by the internal recurrent parity mode.

```json
{"type": "s4d", "state_size": 64, "init": "s4d-lin", "n_ssm": 2, "bidirectional": true, "discretization": "bilinear", "trainable_b": true, "state_lr": 0.001, "freq_scale": 3, "sobolev_filter": {"beta_init": 0, "learning_rate": 0.01, "trainable": true, "weight_decay": 0, "granularity": "channel"}, "output_transform": "glu"}
```

See [S4D diagonal state-space block](s4d.md) for the numerical reference
contract and runtime boundary.

### BatchNorm boundary

`norm_type: "batchnorm"` computes one statistic per channel across the
combined batch/time axes, matching `torch.nn.BatchNorm1d` on `[B,D,T]`.
Training uses batch statistics and updates checkpointed running mean/variance;
validation and native evaluation use only those running statistics. V1
requires `training.objective: "classification"`, `norm_placement: "pre"`,
affine normalization, fixed unpadded records, and more than one batch-time
sample. Recurrence, weight groups, SWA, and Hugging Face export are rejected
until their running-stat semantics are defined.

BatchNorm can replace the outer pre-norm of `plain`, `swiglu`, `geglu`, `mlp`,
`s4d`, `mamba3-canonical`, and `gated_deltanet` blocks. For the two modern
recurrent mixers, only that outer norm changes: canonical Mamba B/C and
post-scan normalization and Gated DeltaNet's per-head output normalization
remain RMSNorm. Other placements and `ffn_internal_norm` fail with a
block-specific validation error.

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

Linear-time gated DeltaNet block (Yang et al., 2025): pre-norm RMS, multi-head Q/K/V projections with optional shared K/V, short 1-D conv on Q/K/V, sigmoid output gate, and a delta-rule recurrence with per-channel gating. The recurrent state is matrix-valued (`d_k × d_v` per head). Apple GPUs use an exact chunk-parallel Metal forward/backward for small per-head states and a bounded recurrent Metal fallback for larger supported states. CUDA uses a chunked associative scan with a custom precompiled triangular-solve kernel (`gpu/cuda_kernels/gated_delta_chunk_solve.cu`). Other shapes and backends use the checkpointed MLX-composed implementation.

Required fields:

- `type: "gated_deltanet"`
- `heads` — number of GLA heads.
- `d_k` — key/query dim per head. Total key dim is `heads * d_k`.

Optional fields:

- `d_v` — value dim per head. Defaults to `2 * d_k`. Total value dim is `heads * d_v`.
- `kv_share` — when `true` (default), the K and V projections share a single `[D, heads*d_v]` weight (V projection is reused for K, with `d_v >= d_k` required). When `false`, K and V get separate projections of width `heads*d_k` and `heads*d_v` respectively. The shared form is the recipe used by Yang et al. and saves one projection matrix per block.
- `scan_chunk_size` — scan chunk/window bound. When omitted, defaults to `64`. `0` explicitly uses the naive per-step MLX scan (slower and intended for correctness debugging). On Metal, positive values select a native bounded-memory scan when `d_k <= 64` and `d_v <= 256`. The exact chunk-parallel implementation is used when `d_k <= 32`, `d_v <= 32`, and the chunk size is at most `64`; larger supported states use the recurrent Metal fallback. Backward recomputation windows are capped at `8` and may be reduced to fit threadgroup memory. On CUDA, the value controls the associative-scan chunk and custom triangular solve. Unsupported shapes use the checkpointed MLX-composed fallback. Must be `>= 0`.
- `bidirectional` — shared-weight two-direction mixing for classification, MLM, or MNTP. The valid prefix is reversed per row, both recurrent deltas are summed, and padding is zeroed before the next block. Parameter count is unchanged and mixer FLOPs are approximately doubled. Native-only in v1.
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

### `hgrn2`

HGRN2 token-mixer block with pre-norm RMSNorm, value/query/forget projections, an output-only matrix-state recurrence, per-head output normalization, output projection, and residual add. This block is a sequence mixer only; pair it with `swiglu` or `geglu` blocks for channel mixing.

Required fields:

- `type: "hgrn2"`
- `heads` — number of recurrent heads. `model_dim` must be divisible by `heads`.

Optional fields:

- `d_state` — key/query state dimension per head. Defaults to `model_dim / heads`.

Implementation notes:

- The recurrent state has shape `d_state × (model_dim / heads)` per head.
- The scan is lowered to composed MLX array operations and autodiff; no custom kernel or custom backward path is used.

Example:

```json
{"type": "hgrn2", "heads": 6, "d_state": 128}
```

### `ttt_mlp`

Experimental nonlinear Test-Time Training token mixer. Each sequence row owns
an independent per-head two-layer MLP state. The block updates that state from
the current chunk's self-supervised reconstruction loss, then reads it through
the query view. The recurrent state resets for every sequence row and every
forward call; it never persists across training batches.

Required fields:

- `type: "ttt_mlp"`
- `heads` - number of inner MLP heads. `model_dim` must be divisible by
  `heads`, and the resulting head dimension must be even for chunk-relative
  RoPE.

Optional fields:

- `chunk_size` - mini-batch TTT chunk width. Defaults to `16`; the final
  ragged chunk is supported.
- `inner_hidden_mult` - inner MLP hidden width relative to head dimension.
  Defaults to `4`.
- `inner_lr_base` - learned inner-SGD base learning rate before head-dimension,
  token-position, and sigmoid-gate scaling. Defaults to `0.1`.
- `inner_lr_init` - value of the inner base learning rate at outer step zero,
  before head-dimension, token-position, and learned-gate scaling. Defaults to
  `0.01` and must not exceed `inner_lr_base`.
- `inner_lr_warmup_steps` - outer steps used to linearly warm the inner rate
  from `inner_lr_init` to `inner_lr_base`. Defaults to `5000`; explicit `0`
  disables warmup.

The v1 algorithm is fixed to the reference two-layer tanh-GELU MLP, affine
LayerNorm residual, one SGD update, learned per-token rate, full meta-gradient,
shared Q/K projection with independent causal depthwise convolutions,
chunk-relative RoPE, post LayerNorm, and GELU output gate. It is a token mixer,
so pair it with `swiglu` or `geglu` for channel mixing.

V1 supports only normal sequential causal training. It rejects masked,
hybrid, diffusion, and multihead objectives; recurrence/custom/U-Net execution;
parallel groups; segment packing; and auxiliary training objectives. Hugging
Face export supports cache-safe causal stacks composed only from `ttt_mlp`
plus pointwise `swiglu`, `geglu`, or `mlp`; the generated PyTorch model exposes
the same request-owned MLP, partial-gradient, convolution, and offset cache.
Native batch-one inference supports request-owned cached state
for stacks composed only from `ttt_mlp` plus pointwise `swiglu`, `geglu`, or
`mlp` blocks; other mixers continue using replay. Native training telemetry reports per-block inner loss before
and after update, update norm, state drift, and inner-LR mean/min/max. These
diagnostics are sampled with live post-step weights through a separately cached
no-gradient graph at normal log cadence; they are not retained by every
optimizer step. Analytical forward FLOPs remain available, but training FLOPs
and MFU are omitted because the generic backward estimate does not represent
TTT's inner VJP and full meta-gradient.

Example:

```json
{"type":"ttt_mlp","heads":4,"chunk_size":16,"inner_hidden_mult":4,"inner_lr_base":0.1,"inner_lr_init":0.01,"inner_lr_warmup_steps":5000}
```

### `mlstm`

mLSTM token-mixer block with pre-norm RMSNorm, Q/K/V projections, input and forget gate preactivations, stabilized matrix-memory recurrence, per-head output normalization, output gate, output projection, and residual add. This block is a sequence mixer only; pair it with `swiglu` or `geglu` blocks for channel mixing.

Required fields:

- `type: "mlstm"`
- `heads` — number of recurrent heads.
- `d_k` — key/query dimension per head.
- `d_v` — value dimension per head.

Optional fields:

- None

Implementation notes:

- The recurrent state has shape `d_k × d_v` per head, plus the stabilized normalizer state.
- The v1 implementation does not include short convolution, sliding-window variants, inference-state caching, custom kernels, or a custom backward path.

Example:

```json
{"type": "mlstm", "heads": 8, "d_k": 64, "d_v": 128}
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
| `where` / `select` | 3 | 1 | None | Elementwise select: condition, true value, false value. |
| `lt` / `less_than` | 1 or 2 | 1 | `scalar` when one input | Elementwise less-than comparison. |
| `gt` / `greater_than` | 1 or 2 | 1 | `scalar` when one input | Elementwise greater-than comparison. |
| `ge` / `gte` / `greater_eq` | 1 or 2 | 1 | `scalar` when one input | Elementwise greater-than-or-equal comparison. |
| `le` / `lte` / `less_eq` | 1 or 2 | 1 | `scalar` when one input | Elementwise less-than-or-equal comparison. |
| `eq` / `equal` | 1 or 2 | 1 | `scalar` when one input | Elementwise equality comparison. |
| `min` / `minimum` | 1 or 2 | 1 | `scalar` when one input | Elementwise minimum. |
| `max` / `maximum` | 1 or 2 | 1 | `scalar` when one input | Elementwise maximum. |
| `sigmoid` | 1 | 1 | None | Elementwise sigmoid. |
| `silu` | 1 | 1 | None | SiLU activation. |
| `gelu` | 1 | 1 | None | GELU activation. |
| `relu` | 1 | 1 | None | ReLU activation. |
| `tanh` | 1 | 1 | None | Tanh activation. |
| `exp` | 1 | 1 | None | Elementwise exponential. |
| `log` | 1 | 1 | None | Elementwise natural logarithm. |
| `sqrt` | 1 | 1 | None | Elementwise square root. |
| `rsqrt` | 1 | 1 | None | Elementwise reciprocal square root. |
| `reciprocal` | 1 | 1 | None | Elementwise reciprocal. |
| `pow` | 1 or 2 | 1 | `exponent` when one input | Elementwise power. |
| `abs` | 1 | 1 | None | Elementwise absolute value. |
| `clamp` | 1 | 1 | `min`, `max` | Elementwise clamp. |
| `softmax` | 1 | 1 | `axis` | Softmax over an axis. |
| `mean` / `mean_axis` | 1 | 1 | `axis` | Mean reduction over an axis. |
| `arange` | 0 | 1 | `start`, `end` | Integer range `[start, end)`. |
| `full` | 0 | 1 | `shape`, `value` | Constant float tensor. |
| `reshape` | 1 | 1 | `shape` | Shape values can use symbolic dims. |
| `transpose` | 1 | 1 | `axes` | Permute tensor dimensions. |
| `dropout` | 1 | 1 | `rate` | Inverted dropout; active only in training. |
| `causal_mask` | 1 | 1 | `T`, `window_size` | Applies Mixlab's causal or sliding-window attention mask. `T` defaults to current sequence length; `window_size: 0` means full causal. |
| `rmsnorm` / `rms_norm` | 2 | 1 | `eps` | Inputs are typically `[x, scale]`. |
| `layernorm` / `layer_norm` | 1 or 3 | 1 | `eps` | Non-affine LayerNorm with one input, or affine LayerNorm with `[x, scale, bias]`. |
| `rope` | 2 | 2 | `T`, `head_dim`, `base` | Rotary embedding helper. Outputs are rotated Q and K. |

### Param keys

The custom-op decoder recognizes these `params` keys:

| Param key | Type | Used by |
|------|------|---------|
| `shape` | array of strings/numbers | `reshape` |
| `axes` | array of integers | `transpose` |
| `axis` | integer | `softmax`, `mean_axis` |
| `head_dim` | integer | `rope` |
| `T` | string or number | `rope`, `causal_mask` |
| `window_size` | integer | `causal_mask` |
| `scalar` | number | `scalar_mul`, one-input comparisons, one-input `min`/`max` |
| `value` | number | `full` |
| `eps` | number | `rmsnorm`, `layernorm` |
| `base` | number | `rope` |
| `rate` | number | `dropout` |
| `exponent` | number | one-input `pow` |
| `min` / `max` | number | `clamp` |

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
| `objective` | string | No | `"causal"` | Training objective: `"causal"`, `"mlm"`, `"mntp"`, `"hybrid"`, `"block_diffusion"`, `"multihead"`, or `"classification"`. Existing configs default to causal next-token training. |
| `lr_schedule` | string | No | `"cosine"` | Outer learning-rate schedule: existing step-driven `"cosine"` behavior or validation-driven `"newbob"`. NewBob is classification-only in v1. |
| `newbob` | object | Required for `lr_schedule: "newbob"` | Disabled | NewBob controls: `annealing_factor`, `improvement_threshold`, `patient`, and `metric` (`"val_loss"` or `"val_error_rate"`). |
| `val_every_steps` | integer | Required for NewBob | omitted | Positive completed-optimizer-step cadence for configured full-split classification validation. |
| `val_examples` | integer | No | `0` | Maximum classification validation examples for configured validation; `0` evaluates the full finite split. |
| `classification` | object | Required for `objective: "classification"` | Disabled | Native sequence-level single-label classification. Requires `num_labels >= 2`; optional `pooling` is `"last"` or `"mean"`, optional `classifier_dropout` is in `[0,1]`, and optional `bias` defaults to `true`. |
| `diffusion` | object | No | Defaults | Block-diffusion corruption and sampler knobs. Only valid with `objective: "block_diffusion"` or `objective: "hybrid"` plus `hybrid_secondary_objective: "block_diffusion"`. Multihead configs put diffusion settings on the block-diffusion head instead. Omit it to use conservative defaults. |
| `heads` | array | Required for `objective: "multihead"` | None | Per-head objective specs for shared-trunk multihead training. V1 supports head objectives `"causal"`, `"mlm"`, `"mntp"`, `"block_diffusion"`, `"rtd"`, and `"energy"`. |
| `export_head` | string | No | First non-native-only scorer head | Multihead scorer head exported by `export-hf`. Must not name a `block_diffusion`, `rtd`, or `energy` head. |
| `diffusion_head` | string | No | First block-diffusion head, when present | Multihead denoiser head used by `generate-diffusion` and `score-diffusion`. |
| `rtd` | object | No | Disabled | ELECTRA-style replaced-token detection auxiliary for multihead training. Supports tied-generator RTD using an MLM/MNTP generator head or a dedicated small MLM generator plus a binary detector head. |
| `minimal_pair` | object | Required with an `energy` head or scorer span-PLL ranking | Disabled | Explicit minimal-pair data for native energy/ranking heads or exportable MLM/MNTP scorer regularization. Supports `source: "jsonl"` or compiled `source: "bin"` with `path`, `loss: "logistic"` or `"hinge"`, `margin`, `pair_batch_fraction`, `energy_aggregation`, `score_source`, `score_head`, and `loss_weight`. |
| `invariance` | object | No | Disabled | Structured two-view symmetric-KL consistency loss for single MLM/MNTP or a multihead masked export head. Supports `source`, `path`, `loss: "sym_kl"`, `weight`, `batch_fraction`, `target: "masked_position"`, and optional `skip_token_ids`. |
| `pll_margin` | object | No | Disabled | Directional paired annotated-span PLL margin for MLM/MNTP or a multihead masked export head. Supports `source`, `path`, `margin`, `weight`, `anchor_weight`, `batch_fraction`, `target: "annotated_span"`, and optional `skip_token_ids`. `"distractor_span"` is accepted as an input alias. |
| `word_structural_objective` | object | No | Disabled | StructBERT-style local shuffle auxiliary for MLM/MNTP vocab heads. When enabled, Mixlab shuffles short spans among unmasked positions and adds a separate current-position reconstruction loss through the existing vocab head. |
| `z_loss` | number | No | `0` | Training-only logit z-regularization weight. When positive, Mixlab adds `z_loss * mean(logsumexp(logits)^2)` to optimizer loss while keeping `eval_loss` and `per_token_nll` as the primary task loss. Masked objectives average z-loss over positive `loss_mask` rows only. Must be finite and `>= 0`. |
| `mlm_mask_prob` | number | No | `0.15` | Probability of selecting each eligible token for masked-objective loss. Must be in `[0,1]`. |
| `mlm_mask_prob_schedule` | array | No | Disabled | Stepwise mask-probability schedule as `[[step, probability], ...]`. Entries must start at step `0`, use strictly increasing integer steps, and probabilities in `[0,1]`. Overrides `mlm_mask_prob` on MLM/MNTP/hybrid masked steps. |
| `mlm_mask_prob_schedule_mode` | string | No | `"step"` | Schedule interpolation mode: `"step"` holds each entry until the next step, while `"linear"` linearly interpolates between adjacent entries and holds the final probability afterward. |
| `mlm_mask_unit` | string | No | `"token"` | MLM position-selection unit: independent `"token"` sampling or tokenizer-derived `"whole_word"` groups. Whole-word mode requires supported `tokenizer.json` metadata next to the shards. |
| `mlm_mask_unit_schedule` | object array | No | Disabled | Stepwise categorical schedule such as `[{"step":0,"unit":"whole_word"},{"step":5000,"unit":"token"}]`. The first entry must start at step `0`. |
| `mlm_mask_token_id` | integer | Required for `mlm`, `mntp`, `hybrid`, `block_diffusion`, `multihead` masked heads | None | Token id used as the mask replacement. For `block_diffusion`, v1 temporarily reuses this field as the mask token source until an objective-neutral `mask_token_id` alias exists. Must be in `[0, vocab_size)`. |
| `mlm_mask_token_prob` | number | No | `0.8` | Probability that a selected MLM token is replaced with `mlm_mask_token_id`. |
| `mlm_random_token_prob` | number | No | `0.1` | Probability that a selected MLM token is replaced with a random token id. |
| `mlm_kept_unchanged_prob` | number | No | `0.1` | Probability that a selected MLM token is kept unchanged. The three replacement probabilities must sum to `1.0`. |
| `hybrid_clm_fraction` | number | No | `0.5` | For `objective: "hybrid"`, deterministic probability of using causal next-token training. In per-batch mode it is applied once per batch; in per-example mode it is applied once per sequence. Must be in `[0,1]`. |
| `hybrid_clm_fraction_schedule` | array | No | Disabled | Stepwise causal-fraction schedule as `[[step, fraction], ...]`. Entries must start at step `0`, use strictly increasing integer steps, and fractions in `[0,1]`. Overrides `hybrid_clm_fraction` by step. |
| `hybrid_clm_fraction_schedule_mode` | string | No | `"step"` | Schedule interpolation mode: `"step"` holds each causal fraction until the next entry, while `"linear"` interpolates between adjacent entries. |
| `hybrid_secondary_objective` | string | No | `"mntp"` | Secondary objective for hybrid training: `"mlm"`, `"mntp"`, or `"block_diffusion"`. |
| `hybrid_mix_granularity` | string | No | `"batch"` | Hybrid mixing granularity: `"batch"` preserves the existing one-objective-per-step behavior, while `"example"` samples causal vs secondary objective independently for each sequence in the batch and applies matching per-sequence attention masks. |
| `attention_segment_mask` | string | No | Disabled | Optional packed-sequence block-diagonal attention masking. Omit, `""`, or `"none"` disables it; `"boundary_token"` derives per-row segment IDs from `attention_segment_boundary_token_id` and applies segment masking to `plain` self-attention during training. |
| `attention_segment_boundary_token_id` | integer | Required for `attention_segment_mask: "boundary_token"` | None | Boundary token id already present in packed training shards. The boundary token starts a new segment and belongs to that new segment. Must be in `[0, vocab_size)`. |
| `example_framing` | object | No | Disabled | Optional raw-stream example framing. When set, the loader slices each shard into `content_len` chunks, wraps every row as `[bos_id] + content + [eos_id]`, and masks the final row position from causal loss. Requires `seq_len = content_len + 2`. |
| `distillation` | object | No | Disabled | Optional fixed-teacher ensemble distillation block for causal, MLM, MNTP, or compatible hybrid masked training. |
| `data2vec` | object | No | Disabled | Optional online EMA representation-distillation auxiliary loss for masked objectives. |
| `phases` | array | No | Disabled | Optional phase schedule. When non-empty, `steps` is computed as the sum of phase `steps` and top-level `steps`/`lr` are ignored by the training loop. Each phase must define `steps > 0` and `lr > 0`. |
| `lr_schedule_steps` | integer | No | `steps` | Independent horizon for the standard cosine LR schedule. This is useful when a reference run stops after a fixed epoch count before its configured scheduler horizon. Must be `>= 0`; `0`/omitted uses `steps`. It cannot be combined with `phases`. |
| `warmup_steps` | integer | No | Legacy default | Absolute warmup length for the standard cosine schedule. Must be `>= 0`; values above the effective LR schedule horizon are clamped. Mutually exclusive with `warmup_ratio`. When both warmup fields are omitted, Mixlab keeps the historical `min(100, lr_schedule_steps)` warmup. |
| `warmup_ratio` | number | No | Disabled | Fraction of the effective LR schedule horizon to use for warmup, rounded to the nearest step. Must be in `[0,1]`. Mutually exclusive with `warmup_steps`. For example, `0.016` gives about a 1.6% warmup. |
| `hold_steps` | integer | No | Legacy default | Constant-peak-LR plateau after warmup in the standard cosine schedule. Must be `>= 0`; values above the remaining schedule horizon are clamped. When omitted, Mixlab keeps the historical `200`-step hold. Set `0` explicitly to start cosine decay immediately after warmup. |
| `warmdown_steps` | integer | No | `0` | Cosine warmdown length at the end of the LR schedule. Must be `>= 0`; values above the schedule horizon are clamped. |
| `min_lr_fraction` | number | No | `0` | Absolute learning-rate floor as a fraction of peak LR across cosine decay and warmdown. Must be in `[0,1)`. `0` preserves the historical near-zero endpoint; `0.1` keeps LR at or above 10% of peak. |
| `target_val_loss` | number | No | `0` | Early-stop threshold on validation loss. `0` disables it. Must be `>= 0`. Checked when validation loss is computed during training. |
| `early_stop` | object | No | Disabled | Optional validation early-stop policy. V1 supports validation-loss plateau patience and step-gated `val_gt` aborts. See below. |
| `first_byte_mask` | boolean | No | `false` | Enables training-time first-byte masked softmax. At UTF-8 codepoint-boundary targets, the optimization loss only normalizes over vocabulary entries whose first byte is syntactically valid as a UTF-8 first byte. Validation, exported per-token NLL, and BPB evaluation use the unmasked loss. Byte-id corpora (`vocab_size <= 256`) work directly; BPE corpora use `tokenizer.json` next to the training shards. |
| `recurrence_activation_frac` | number | No | `0` | Delays model-level `recurrence` execution until `floor(frac * total_steps)`. Before activation, training executes each recurrence root once in first-appearance order while preserving the full recurrence weight layout. Mutually exclusive with `recurrence_activation_step` and top-level `recurrence_phases`; must be in `[0,1]`. |
| `recurrence_activation_step` | integer | No | `0` | Delays model-level `recurrence` execution until this absolute training step. Before activation, training executes each recurrence root once in first-appearance order while preserving the full recurrence weight layout. Mutually exclusive with `recurrence_activation_frac` and top-level `recurrence_phases`; must be `>= 0`. |
| `ttt_steps` | integer | No | `0` | Test-time training updates per validation batch during eval mode and full BPB eval. `0` disables TTT. In `ttt_mode="full"` this is score-first full-weight adaptation; in `ttt_mode="lora"` this is per-batch LoRA adaptation before scoring. Must be `>= 0`. |
| `ttt_mode` | string | No | `"full"` | TTT implementation: `"full"` keeps score-first full-weight fine-tuning, `"lora"` uses per-batch Low-Rank Adaptation at test time and discards adapters after each batch. Must be `"full"` or `"lora"`. |
| `ttt_lr` | number | No | `1e-5` | Learning rate for TTT updates. Must be `>= 0`; keep much smaller than training `lr`. |
| `ttt_rank` | integer | No | `4` | LoRA rank used when `ttt_mode="lora"`. Rank-2-or-higher weights get temporary adapters `W + A @ B` with `A:[M,R]`, `B:[R,N]`; only `A` and `B` train during LoRA-TTT. Must be `> 0`. |
| `hardware_tflops` | number | No | `0` | Peak hardware TFLOPS used to log MFU next to `tok/s`. `0` disables MFU logging. Must be `>= 0`. |
| `optimizer` | string | No | `"muon"` | Optimizer selector: `"muon"`/`"muon_eq_r"`/`"normuon"` use Muon variants for matrix weights and AdamW for embed/head/scalar groups; `"adamw"` uses AdamW for all groups; `"lamb"` uses LAMB for all groups. |
| `compute_dtype` | string | No | `"float32"` | MLX training compute dtype: `"float32"` or experimental `"bf16"`. BF16 keeps fp32 master weights and optimizer state, and currently supports `plain`, `swiglu`, `geglu`, `mlp`, and `moe` blocks. Unsupported blocks and QAT fail fast instead of silently falling back. |
| `qat` | string | No | `"none"` | Quantization-aware training mode for rank-2 weights during the training forward pass. `"none"` disables it, `"int8"` applies per-row fake int8 quantization, and `"int6"` applies a coarser fake quantization with STE. |
| `qat_start` | integer | No | `0` | Delays QAT until this zero-based training step. Before activation the normal unquantized training graph is used. Must be `>= 0`; positive values require `qat` to be enabled. |
| `weight_init` | string | No | `"xavier_uniform"` | Initialization policy: `"xavier_uniform"`, `"normal"`, `"gptbert"`, `"gpt2"`, or `"pytorch_linear"`. The PyTorch mode applies `nn.Linear`-compatible fan-in uniform initialization only to affine tensors identified by architecture metadata; specialized tensors and embeddings retain their existing initialization. |
| `weight_init_std` | number | No | `0.02` | Standard deviation for `"normal"` and `"gpt2"` initialization. Ignored by `"xavier_uniform"`, `"gptbert"`, and `"pytorch_linear"`. |
| `grad_clip` | number | No | `0` | Max grad norm. `0` means no clipping. Must be `>= 0`. |
| `weight_decay` | number | No | `0.01` | Global fallback weight decay. Must be `>= 0`; explicit `0` disables decay for groups that inherit it. |
| `weight_decay_policy` | string | No | `"matrix_only"` | `"matrix_only"` preserves the existing no-decay treatment for scalar/vector parameters. `"all"` enables decay for every ordinary trainable parameter; explicit exemptions for S4D A/B/dt and Mamba3/Gated DeltaNet `A_log`/`dt_bias` still win. See [dynamics optimizer policy](dynamics-optimizer-policy.md). |
| `cautious_weight_decay` | boolean | No | `false` | When true, applies weight decay only to elements where parameter and gradient signs agree. This is an optimizer modifier for AdamW, LAMB, Muon, MuonEq-R, NorMuon, and SGD paths, not a separate optimizer kind. |
| `cautious_weight_decay_activation_frac` | number | No | `0` | Fraction of training before cautious weight decay activates. Before activation, standard weight decay is used. `0` means active from step 0 when `cautious_weight_decay=true`; must be in `[0,1]`. |
| `beta1` | number | No | `0.9` | AdamW beta1. Also seeds Muon momentum when `muon_momentum` is omitted. |
| `beta2` | number | No | `0.95` | AdamW and Muon beta2. |
| `epsilon` | number | No | `1e-8` | AdamW / Muon epsilon. |
| `lamb_beta1` | number | No | `0.9` | LAMB beta1. Used only when `optimizer: "lamb"`; must be in `[0,1)`. |
| `lamb_beta2` | number | No | `0.999` | LAMB beta2. Used only when `optimizer: "lamb"`; must be in `[0,1)`. |
| `lamb_eps` | number | No | `1e-6` | LAMB epsilon. Used only when `optimizer: "lamb"`; must be `> 0`. |
| `lamb_trust_ratio_cap` | number | No | `10.0` | Upper bound for the LAMB trust ratio `||w|| / ||update||`. `0` disables capping for experiments; values must be finite and `>= 0`. |
| `seed` | integer | No | `42` | RNG seed. `0` is treated as omitted and replaced with `42`. |
| `batch_tokens` | integer | No | `1024` | Tokens per optimization step. Must be divisible by `seq_len` normally; with `length_buckets`, it is a ceiling used to derive each bucket's row count. Mutually exclusive with `batch_size`. |
| `batch_size` | integer | No | omitted | Fixed records per step under `length_buckets`. Every bucket uses this row count while tokens per step vary as `batch_size * bucket_width`. Requires `length_buckets` and is mutually exclusive with `batch_tokens`. |
| `length_buckets` | integer array | No | omitted | Optional strictly increasing classification sequence widths. Supported for `linear_frames` and `discrete_codebooks`; each width must be `<= seq_len`. When `batch_tokens` is used, the largest bucket must also be `<= batch_tokens`. |
| `seq_len_schedule` | array | No | Disabled | Stepwise training sequence-length schedule as `[[step, seq_len], ...]`. The top-level `seq_len` remains the maximum/eval/inference length. Scheduled lengths must start at step `0`, be strictly increasing by step, stay in `[1, seq_len]`, and divide `batch_tokens`. V1 rejects this with distillation or active data2vec. |
| `shuffle_chunk_tokens` | integer | No | `seq_len` | Token-block shuffle granularity for train/validation loaders. Values `<= 0` inherit `seq_len`; set to `2048` to reproduce the previous fixed-block behavior. |
| `reverse_complement_prob` | number | No | `0` | Deterministic DNA reverse-complement augmentation probability in `[0,1]`. It applies per packed segment for record shards and per framed row for continuous nucleotide streams. Requires a manifest-backed FASTA DNA dataset; RNA and legacy/text shards reject nonzero values. Randomness is derived from `training.seed` and step, independent of loader prefetch. |
| `embed_lr` | number | No | `lr` | Learning rate for embedding-class weights. |
| `matrix_lr` | number | No | `lr` | Learning rate for matrix weights. Used with Muon. |
| `scalar_lr` | number | No | `lr` | Learning rate for scalar and vector weights. |
| `head_lr` | number | No | `lr` | Learning rate for the output head. Ignored when `tie_embeddings=true` unless `mtp.untie_embed_at_frac < 1` reserves and later activates a separate head. |
| `muon_momentum` | number | No | `beta1` | Muon momentum term for matrix weights. Must be `>= 0`. |
| `muon_backend_steps` | integer | No | `5` | Muon backend iteration count. Must be `> 0` after defaults. |
| `newton_schulz_variant` | string | No | `"fixed"` | Muon Newton-Schulz coefficient schedule. `"fixed"` uses the canonical fixed coefficients; `"polar_express"` uses the supported per-iteration Polar Express coefficients. Ignored by AdamW and LAMB. |
| `muon_nesterov` | boolean | No | `true` | Enables Muon Nesterov mode when set or omitted. |
| `embed_weight_decay` | number | No | `weight_decay` | Per-group weight decay for embeddings. Explicit `0` disables this group's decay. |
| `matrix_weight_decay` | number | No | `weight_decay` | Per-group weight decay for matrices. Explicit `0` disables this group's decay. |
| `scalar_weight_decay` | number | No | `weight_decay` | Per-group weight decay for scalars and vectors. Explicit `0` disables this group's decay. |
| `head_weight_decay` | number | No | `weight_decay` | Per-group weight decay for the output head. Explicit `0` disables this group's decay. |
| `swa_start` | integer | No | `0` | Step at which SWA/EMA accumulation starts. Must be `>= 0`. |
| `swa_decay` | number | No | `0.999` | EMA decay for SWA weights. Must be in `[0, 1)`. |
| `swa_interval` | integer | No | `10` | Update frequency for SWA accumulation. |

### SWA/EMA averaged weights

`training.swa_start`, `training.swa_decay`, and `training.swa_interval` enable exponential moving-average weight tracking during training. `swa_start: 0` disables the feature. The same values can be overridden at runtime with `-swa-start`, `-swa-decay`, and `-swa-interval`; explicit CLI overrides are validated before training and logged during initialization.

When `-safetensors <base>.safetensors` is used without populated SWA/EMA weights, Mixlab preserves the existing behavior and writes exactly that path. Once SWA/EMA weights are populated, Mixlab writes two artifacts instead:

- `<base>.final.safetensors` for the live final trainer weights
- `<base>.swa.safetensors` for the averaged SWA/EMA weights

Periodic checkpointing follows the same convention after averaged weights exist, producing `step_N.final.safetensors` and `step_N.swa.safetensors`. Checkpoints before the first averaging update keep the legacy `step_N.st` path.

Each periodic model checkpoint also has `step_N.state.safetensors` and a completion marker named `step_N.resume.json`. Use `-resume <checkpoint-dir-or-manifest>` to restore the live model weights, optimizer state, global step, schedule, loader position, deterministic RNG position, SWA/data2vec EMA, and early-stop state. `-safetensors-load` remains a weights-only warm start. Standard-schedule extension may increase `training.steps`; it continues at the original terminal LR after the original horizon and never repeats warmup. See [CLI: Training](cli-train.md#resume-and-extension) for compatibility checks, phase-schedule limits, replay startup cost, and storage overhead.

The Hugging Face exporter uses whichever checkpoint is passed through `-safetensors-load`. Pass the `.swa.safetensors` file to export averaged weights, or `.final.safetensors` to export live final weights.

### Training objectives

`training.objective: "classification"` replaces the language-model loss with a
padding-aware sequence classifier while retaining the complete LM weight layout
as a warm-startable prefix. Its two appended weights are
`head_classifier_proj: [model_dim, num_labels]` and
`head_classifier_bias: [num_labels]`. The bias initializes to zero.

`training.classification.pooling` defaults to `"mean"` when every token mixer
is bidirectional, including bidirectional `plain`, `s4d`, `mamba3-canonical`,
and `gated_deltanet` blocks. It defaults to `"last"` for causal or mixed stacks.
Mean pooling uses the validity-mask-weighted mean. Last pooling selects the
final non-padding token. Both reject empty rows.
`classifier_dropout` defaults to top-level `hidden_dropout`.

The classifier projection follows the model-wide `training.weight_init`
policy. The default remains Xavier uniform for backward compatibility. Use
`weight_init: "pytorch_linear"` when reproducing a PyTorch classifier; Mixlab
then initializes both the classifier weight and its optional bias from
`Uniform(-1/sqrt(model_dim), +1/sqrt(model_dim))`.

Classification requires a manifest-backed
`mixlab_labeled_sequence_shard_v1` dataset with
`sequence_layout: "one_record_per_row"` and task metadata
`{"type":"single_label_classification","num_labels":N}`. V1 supports
single-label cross-entropy only. Masked-objective auxiliaries, diffusion,
segment packing, sequence-length schedules, and reverse-complement augmentation
are rejected rather than silently composed. Native classification checkpoints
export through `AutoModelForSequenceClassification` when their backbone/config
surface is HF supported. The trained projection and bias, `num_labels`, pooling
mode, and classifier dropout are preserved. Export still fails explicitly for
a gated backbone such as `gated_deltanet` or for native-only composition such
as `rc_equivariant`.

`blocks: []` is valid only for native classification with
`input_adapter.kind: "linear_frames"` or `"discrete_codebooks"`. This is the
explicit linear-probe path: adapter output goes directly through the optional
final norm, pooling, and classifier. Specify classification pooling explicitly
for these configs. Token embeddings and all language-model objectives still
require at least one block.

`objective: "causal"` preserves the existing shifted next-token objective. `objective: "mlm"` masks selected input positions and trains only those positions to predict the original token. `objective: "mntp"` predicts next tokens from bidirectional context while masking the corresponding next-token input position so the answer is not visible. `objective: "hybrid"` deterministically mixes causal and the configured secondary objective from `training.seed` and the step index. Hybrid secondary objectives can be `"mlm"`, `"mntp"`, or `"block_diffusion"`. `objective: "block_diffusion"` trains block-wise masked diffusion: each example masks a randomly selected active block under a prefix-plus-block attention mask, and the masked cross-entropy loss is taken over the masked positions. `objective: "multihead"` trains one shared trunk over expanded rows with distinct named heads, currently intended for scorer-plus-denoiser recipes such as MNTP/BERT-MLM scoring plus block-diffusion denoising. Sample from a trained diffusion checkpoint with `-mode generate-diffusion` (see [cli.md](cli.md#generate-diffusion)). The default `hybrid_mix_granularity: "batch"` chooses one objective per batch. `hybrid_mix_granularity: "example"` chooses independently per sequence inside the batch for MLM/MNTP secondary objectives, so `hybrid_clm_fraction: 0.0625` gives approximately 6.25% causal sequences and 93.75% masked sequences over time.

Hybrid training uses objective-specific attention masks regardless of any block-level `attention_mask`: causal batches or sequences are always causal, MLM/MNTP batches or sequences are always bidirectional, and block-diffusion batches use the prefix-plus-block diffusion mask. In per-example mode, `plain` attention receives a row-level mask so causal and bidirectional MLM/MNTP sequences can share one training step safely. V1 rejects `hybrid_mix_granularity: "example"` with `hybrid_secondary_objective: "block_diffusion"` because per-row diffusion block boundaries are not represented in the shared hybrid-example mask. Validation, full eval, generation, and other next-token scoring paths use causal attention. `plain.window_size` is rejected for hybrid configs with any masked secondary steps, and always rejected for block-diffusion objective paths.

Masked objectives emit a training loss averaged only over rows where `loss_mask > 0`; the dense `eval_loss` and `per_token_nll` outputs remain available for evaluation/export paths. Top-level `mtp` and `training.first_byte_mask` are not supported with non-causal objectives in this version.

`block_diffusion` requires `training.mlm_mask_token_id` in v1. This is compatibility debt: the field supplies the mask token for diffusion today, but a future objective-neutral `mask_token_id` alias should replace the MLM-specific spelling without changing token semantics. V1 only supports sequential `plain` self-attention plus `swiglu`, `geglu`, `mlp`, and `moe` blocks. It rejects `plain.window_size`, `training.attention_segment_mask`, fixed-teacher distillation, data2vec, recurrent/SSM blocks, cross-attention, and custom blocks. These restrictions apply both to pure block-diffusion configs and hybrid configs whose secondary objective is block diffusion. Pure block-diffusion checkpoints remain native-only for HF export. Multihead checkpoints can export the configured non-diffusion scorer head while intentionally skipping native-only denoiser weights.

`training.objective: "multihead"` requires at least two uniquely named heads and a positive total raw `loss_weight`. Each head supports `objective`, `loss_weight`, `layer_aggregation` (`"none"` or `"dwa"`), `output_head` (`"bert_mlm"` for MLM/MNTP, `"linear"`, `"binary"` for RTD, or `"scalar"` for energy), `tie_embeddings`, `final_norm`, and an optional per-head `diffusion` object for `block_diffusion` heads. Multihead v1 rejects top-level `training.diffusion`, distillation, data2vec, MTP, first-byte masked loss, example framing, segment attention masking, recurrence phases, custom blocks, and `plain.window_size`. `export-hf` exports only `export_head`; head-level DWA is represented as head-scoped aggregation for the exported scorer head. `generate-diffusion` and `score-diffusion` use `diffusion_head`.

`training.rtd` enables ELECTRA-style replaced-token detection inside multihead training. Set `generator: "tied"` to sample replacements from an existing MLM or MNTP `generator_head`, or set `generator` to a dedicated object such as `{"type": "dedicated", "model_dim": 192, "layers": 4, "heads": 4, "mlp_mult": 4.0, "generator_loss_weight": 1.0}` to train a separate bidirectional MLM mini-transformer. Dedicated generator weights are prefixed `rtd_generator_*`, are checkpointed normally, and are skipped by HF scorer export. Exactly one head must set `objective: "rtd"` with `output_head: "binary"`. `mask_prob` defaults to `mlm_mask_prob`, `sample_temperature` defaults to `1.0`, and `discriminator_loss_weight` defaults to `50.0`; the RTD loss contribution is `head.loss_weight * discriminator_loss_weight * BCE`. Dedicated mode also adds `generator_loss_weight * generator_mlm_loss`. The detector predicts `1=original`, `0=replaced` over every token. HF export skips RTD state and exports the configured scorer head only; `score-electra` reads native detector logits and reports summed `log P(original)` scores for token-id sequences.

`training.minimal_pair` enables clean/corrupt pair ranking inside multihead training. Pair JSONL records use `{"id":"...","clean":[...],"corrupt":[...],"family":"..."}` with token IDs in range; span-aware configs may also include `clean_span:[start,end]` and `corrupt_span:[start,end]` half-open ranges. Use `mixlab -mode prepare-pairs` to validate JSONL and optionally compile it to a compact `.mpair` artifact, then set `source: "bin"` for faster training startup. `mixlab -mode prepare` can also generate corpus-only pair JSONL with weighted broad corruption families (`agreement`, `attractor`, `word_order`, `npi_licensor`, `quantifier_scope`, and `filler_gap`), induced morphology tables, duplicate filtering, rejection counters, and optional audit samples.

`training.invariance` adds a structured two-view consistency regularizer without adding model weights. Invariance JSONL records use `{"id":"...","family":"...","view_a":[...],"view_a_pos":N,"view_b":[...],"view_b_pos":M}`. The positions are required, are masked with `mlm_mask_token_id`, and must identify the same unchanged token in both views. Mixlab compares only those two vocabulary distributions using symmetric KL. For numerical stability, probabilities below `1e-8` are floored and renormalized before the divergence, bounding only collapsed-tail contributions. `skip_token_ids` can reject configured tokenizer special IDs at the annotated positions; the mask token is always rejected. `source:"file"` auto-detects JSONL and compiled invariance-pair binaries; `source:"jsonl"` and `source:"bin"` are explicit alternatives. Active invariance supports single-objective MLM/MNTP and multihead masked export heads, requires an even number of sequence rows per batch, and is training-only. `weight:0` is an exact no-op with no pair-file load or extra graph. Use `mixlab -mode prepare-pairs` to validate or compile the records.

`training.pll_margin` adds a directional paired masked-span PLL objective without adding model weights. Pair JSONL records use `{"id":"...","family":"...","view_pos":[...],"target_pos_positions":[...],"view_neg":[...],"target_neg_positions":[...],"target_ids":[...]}`. The two position arrays must be non-empty and strictly increasing, and each must select exactly the shared `target_ids` span in its corresponding view. Mixlab masks all annotated positions in both views, sums `log_softmax(logits)[target]` across the span, and optimizes `softplus(margin - (pll_pos - pll_neg)) + anchor_weight * -pll_pos`. `weight` defaults to `0.1`: the anchor is naturally large at the start of from-scratch training, so increase the coefficient only while monitoring auxiliary and primary losses. Pair rows do not contribute ordinary MLM/MNTP CE, so the contrast is never trained as a normal reconstruction target. An auxiliary pair is excluded when either sequence has a non-finite token-logit row. The loss uses a stable analytical backward and bounds each auxiliary token-logit gradient to `[-1,1]` before model-wide clipping. Active PLL margin supports single MLM/MNTP or multihead masked export heads, requires an even number of rows, and is training-only. It rejects batch-mutating pair auxiliaries and other unsupported objective combinations in v1. `weight:0` is graph-identical to disabled mode. Use `mixlab -mode prepare-pairs` to validate or compile the artifact; `scripts/make_distractor_margin_pairs.py` is an optional corpus-only producer for one agreement-attractor use case.

`training.word_structural_objective` enables a StructBERT-style local order auxiliary on MLM/MNTP vocab-logit paths. Object-present defaults are `enabled:true`, `fraction:0.05`, `span:3`, `loss_weight:1.0`, and `skip_token_ids:[mlm_mask_token_id]`. Mixlab first applies primary MLM/MNTP corruption, then shuffles non-overlapping spans drawn only from unmasked, unskipped positions and adds a separate current-position reconstruction loss through the existing vocab head. Multihead configs may set `heads:["scorer"]`; omitted `heads` defaults to `export_head`. Omit the object or set `enabled:false` for exact disabled parity. `loss_weight:0` keeps the shuffle active and only disables the additive loss term.

The default `score_source: "energy_scalar"` preserves the native energy-head path. Energy heads default to `energy_aggregation: "mean"`, which preserves the legacy whole-sequence scalar energy; set `energy_aggregation: "differing_span"` to emit per-token scalar energies and train/score on the explicit or alignment-derived differing span. Lower energy is preferred. Set `score_source: "mlm_span_pll"` to add an exportable scorer regularizer instead: Mixlab appends clean/corrupt pair rows, masks only the differing span, sums `log_softmax(logits)[target]` over that span on the selected MLM/MNTP `score_head`, and applies the same logistic or hinge pairwise loss. This mode requires `energy_aggregation: "differing_span"` and uses `loss_weight` as the regularizer coefficient. `score-ebm` reads native energy outputs or scorer span-PLL scores depending on `score_source`; pair summaries use positive `margin` for a correct clean preference in both modes. For scorer configs, explicit `score-ebm -score-pll-aggregation full_seq`, `differing_span`, or `dependent_window` compute deterministic score-time masked PLL from per-position log-probs, skipping `mlm_mask_token_id` plus any IDs passed with `-score-pll-skip-token-ids`; `dependent_window` expands the differing span by `-score-pll-window K`. `-score-pll-attribution-dump` writes per-pair token log-prob traces for auditing. These scoring modes do not change training. `-score-emit-token-energy` is energy-only. HF export skips native-only heads and exports the configured scorer head; `mlm_span_pll` adds no export-time tensors.

`training.diffusion` holds both corruption-time (training) and sampler-time (generation) knobs so `-mode generate-diffusion` shares the trained block size. Corruption fields shape masking during training; sampler fields drive the `generate-diffusion` denoising loop.

`training.diffusion` fields:

| Field | Applies to | Type | Required | Default | Notes |
|------|------------|------|----------|---------|-------|
| `block_size` | Training and generation | integer | No | `16` when it divides `seq_len`, otherwise `seq_len` | Active diffusion block length. Must be `> 0`, `<= seq_len`, and divide `seq_len` in v1. |
| `min_mask_fraction` | Training | number | No | `0.05` | Minimum sampled training mask fraction. Must be in `[0,1]` and `<= max_mask_fraction`. |
| `max_mask_fraction` | Training | number | No | `1.0` | Maximum sampled training mask fraction. Must be in `(0,1]`. |
| `steps_per_block` | Generation | integer | No | `block_size` | Sampler denoising passes per block. Must be `> 0`. |
| `confidence_threshold` | Generation | number | No | `0.8` | Sampler confidence threshold for committing positions. Must be in `[0,1]`. |
| `commit_floor` | Generation | integer | No | `1` | Minimum positions to commit on a sampler pass when none reaches the threshold. Must be in `[1, block_size]`. |
| `timestep_conditioning` | Multihead diffusion heads | string | No | `"none"` | `"adaln"` enables per-block adaptive scale/shift conditioning from the sampled mask fraction. |
| `timestep_conditioning_dim` | Multihead diffusion heads | integer | No | `128` for AdaLN | Hidden size for the AdaLN timestep MLP. Must be `> 0` when `timestep_conditioning: "adaln"`. |

For diffusion experiments, compare against causal and MLM baselines that keep the same supported Transformer backbone, vocabulary, context length, optimizer settings, data, and training-token budget. Use causal validation BPB/per-token NLL for apples-to-apples model comparison, and report diffusion masked training loss separately because it is not the same metric as next-token validation loss.

`attention_segment_mask: "boundary_token"` is for packed training streams that already contain a reliable document/segment marker. Mixlab derives `segment_ids` from the unmasked input tokens, then masks `plain` self-attention so tokens attend only within their segment. Causal, bidirectional, and per-example hybrid masks still apply inside each segment. V1 supports `plain` self-attention stacks with position-wise FFN/MoE blocks; fixed-teacher distillation, recurrent blocks, cross-attention, custom token mixers, and inference/generation segmentation are out of scope.

Manifest-backed `mixlab_sequence_shard_v1` nucleotide datasets enable the same
segment-attention machinery automatically and provide exact segment IDs from
the record packer. Their fixed vocabulary uses `<MASK>=3`, so MLM configs must
set `training.mlm_mask_token_id: 3`. Causal and MLM/MNTP objectives are
supported; full continuous-stream BPB evaluation is intentionally disabled.

Manifest-backed nucleotide `mixlab_token_shard_v1` datasets with
`sequence_layout: "continuous_stream"` require causal `training.example_framing`.
They load the same nucleotide vocabulary and reverse-complement metadata but do
not enable `attention_segment_mask` or declare `segment_ids` in the model graph,
so attention, recurrent, and SSM mixers can train on the same shards.

With `rc_equivariant: true`, Mixlab additionally loads the DNA complement map
for packed MLM and labeled one-record classification datasets. Every input is
paired with its segment-local reverse complement and evaluated by the same
weights. LM logits are position-aligned and vocabulary-complemented before
averaging; classification averages aligned hidden branches and therefore
requires padding-aware mean pooling. Special tokens are self-complementary and
do not move.

Manifest-backed text datasets with
`sequence_layout: "one_record_per_row"` enable per-record causal framing
without a model-config field. Every source record becomes one
`[BOS] content [EOS] [PAD]...` row, loss is active through the EOS target, and
all PAD targets are ignored. The manifest supplies `record_seq_len` and
semantic `pad`, `bos`, and `eos` token IDs; its length must match config
`seq_len`. V1 supports causal training only and rejects an additional segment
mask or `training.example_framing` because neither composes meaningfully with
one-record-per-row data.

`example_framing` is for raw continuous token shards that should train as independent fixed examples. It shuffles `content_len` raw-token chunks, drops ragged shard tails, prepends `bos_id`, appends `eos_id`, and masks the final EOS input position so it never predicts the next example. Nucleotide stream manifests require the framing IDs to match `nucleotide_vocab.json` (`BOS=1`, `EOS=2`). V1 is causal-training-only and rejects masked objectives, hybrid, block diffusion, MTP, first-byte masked loss, distillation, data2vec, attention segment masking, TTT eval settings, and `seq_len_schedule`.

```json
{
  "seq_len": 514,
  "training": {
    "batch_tokens": 8224,
    "example_framing": {"content_len": 512, "bos_id": 1, "eos_id": 2}
  }
}
```

`mlm_mask_prob_schedule` changes the mask ratio deterministically. With the default `"step"` mode, `[[0, 0.30], [5000, 0.15]]` uses a 30% mask rate until step 5000, then 15%. With `"linear"` mode, Mixlab interpolates between adjacent entries, so the same schedule gradually decays from 30% to 15% over the first 5000 steps. It applies to MLM, MNTP, and masked hybrid batches; causal hybrid batches ignore it because they do not use masked-objective rows.

`mlm_mask_unit_schedule` independently selects the masking unit. Whole-word
mode groups eligible pieces per sequence row using tokenizer vocabulary
markers, samples complete groups under a per-row token budget, and keeps the
existing 80/10/10 replacement decision per selected token. The realized rate
can be below the target when no remaining whole group fits the budget; training
telemetry reports the target, realized rate, group sizes, and underfilled rows.
Supported metadata conventions are Metaspace with an always-prepended marker,
ByteLevel BPE with prefix-space behavior, and WordPiece continuation prefixes.
Special tokens and `mlm_mask_token_id` are ineligible. V1 applies the unit only
to MLM paths, including MLM hybrid steps/rows, multihead MLM heads, and tied RTD
generator probes; MNTP, diffusion, and word-structural selection are unchanged.
To convert a corpus-pass curriculum into zero-based steps, let `T` be the sum of
training-shard token counts and `B` be `batch_tokens`. For ten passes with a
WWM-to-token transition after seven passes, use
`steps = floor(10*T/B)` and transition step `ceil(7*T/B)`. An independent
sequence-length transition after two passes uses `ceil(2*T/B)`.

`hybrid_clm_fraction_schedule` changes the probability of selecting causal hybrid batches over time. With `"step"` mode, `[[0, 0.75], [10000, 0.25]]` uses mostly causal batches before step 10000 and more masked/diffusion batches afterward. With `"linear"` mode, Mixlab interpolates the causal fraction between entries. This schedule is useful for hybrid block-diffusion runs that warm up with causal next-token training before increasing denoising exposure.

`weight_init: "gptbert"` uses truncated normal initialization with `std = sqrt(2 / (5 * model_dim))`. Output projections in `plain`, `swiglu`, `geglu`, `mlp`, and MoE expert FFNs are additionally scaled by `sqrt(1 / (2 * (1 + block_index)))`, where `block_index` is the zero-based top-level block position.

`weight_init: "gpt2"` matches the Hugging Face GPT-2 initializer: rank-2-or-higher weights and embedding tables use `Normal(0, weight_init_std)`, defaulting to `0.02`; 1D norm scales initialize to `1` and biases to `0`; residual output projections (`plain.wo`, `plain.ff2`, GLU/MLP `w_down`, and MoE expert down projections) use the constant scale `weight_init_std / sqrt(2 * len(blocks))` across all layers.

`weight_init: "pytorch_linear"` matches `torch.nn.Linear.reset_parameters()` for affine tensors explicitly identified by the architecture: weights and paired biases use `Uniform(-1/sqrt(fan_in), +1/sqrt(fan_in))`. This includes continuous `linear_frames` input projections, S4D GLU output projections, and native classification heads. Per-weight initialization modes, SSM state parameters, embeddings, and other unmarked tensors keep their existing initialization. Omission remains byte-compatible with the historical Xavier default.

`lr_schedule_steps` separates the standard cosine schedule horizon from the number of optimizer updates. For example, `steps: 180000` with `lr_schedule_steps: 200000` stops after 180,000 updates while evaluating every learning rate against a 200,000-step cosine. When omitted, both horizons remain identical as before.

`lr_schedule: "newbob"` selects a stateful classification-only schedule driven
by configured validation. It requires `newbob` and a positive
`val_every_steps`; validation runs at completed-step boundaries and at the
final update independently of progress logging. `val_examples: 0` means the
full finite validation split, while a positive value takes an exact
deterministic prefix. `newbob.metric` is a lower-is-better `"val_loss"` or
`"val_error_rate"`. The first observation never anneals. Later observations
use relative improvement against the immediately previous metric; when it is
below `improvement_threshold`, the LR is multiplied by `annealing_factor`
after the configured patience is exhausted. A previous metric of zero is
defined as zero improvement. NewBob cannot be combined with phases,
`lr_schedule_steps`, or cosine warmup/hold/warmdown fields. Resumable
checkpoints preserve its live LR, previous metric, and patience counter.

`seq_len_schedule` changes the training graph shape at configured step boundaries while keeping `batch_tokens` fixed. Mixlab caches one program per active scheduled shape/objective combination and switches at boundaries; validation, full eval, generation, and final unmasked loss use the top-level maximum `seq_len`. This v1 schedule is intentionally disabled with fixed-teacher distillation and active data2vec because those teacher runtimes are built around the configured sequence length.

`early_stop` observes validation loss when validation runs. The existing `target_val_loss` is still the simplest success threshold. `early_stop.patience` stops after that many consecutive validation checks without an improvement greater than `min_delta` after `min_steps`; `early_stop.val_gt` stops at the first validation check at or after `at_step` whose loss remains above the configured value. Omitted or zero-valued `patience` / `val_gt` disables that sub-policy.

```json
{
  "training": {
    "early_stop": {"metric": "val", "patience": 3, "min_delta": 0.01, "min_steps": 1000}
  }
}
```

```json
{
  "blocks": [
    {"type": "plain", "heads": 8},
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

### Training distillation

`training.distillation` enables same-corpus internal teacher distillation during causal, MLM, MNTP, or compatible hybrid masked training. For each distilled student batch, Mixlab evaluates all teacher checkpoints without gradients on the same corrupted inputs, ensembles their token distributions, and trains the student with:

`loss_weight_ce * hard_target_cross_entropy + loss_weight_kl * KL(P_teacher || P_student)`

For masked objectives, hard-target CE and teacher KL are both averaged only over the masked loss rows. MNTP uses the existing no-leakage alignment: the visible mask is at the next-token input position while logits and KL are read from the predicting row. Hybrid batch-granularity causal steps skip teacher evaluation; hybrid example-granularity batches keep primary CE over the existing hybrid loss mask but apply KL only to masked rows. Validation, per-token NLL export, checkpoints, SWA, scoring, and full evaluation remain student-only.

```json
{
  "training": {
    "objective": "mntp",
    "mlm_mask_token_id": 1,
    "distillation": {
      "teacher_checkpoints": [
        "runs/teacher_a.safetensors",
        "runs/teacher_b.safetensors"
      ],
      "teacher_configs": [
        "configs/teacher_a.json",
        "configs/teacher_b.json"
      ],
      "loss_weight_ce": 0.5,
      "loss_weight_kl": 0.5,
      "ensemble_strategy": "mean_logits",
      "temperature": 1.0
    }
  }
}
```

| Field | Type | Required | Default | Notes |
|------|------|----------|---------|-------|
| `teacher_checkpoints` | string array | Yes when `loss_weight_kl > 0` | None | Mixlab safetensors files using the existing `w{index}_{name}` tensor naming. Must match `teacher_configs` length. |
| `teacher_configs` | string array | Yes when `loss_weight_kl > 0` | None | Teacher architecture configs. Teacher `vocab_size` and `seq_len` must match the student; masked distillation also requires matching `mlm_mask_token_id`, and matching tokenizer hashes when tokenizer artifacts are discoverable. Runtime `batch_tokens` is forced to the student batch size. |
| `loss_weight_ce` | number | Yes | None | Weight for the normal hard-target cross-entropy term. Must be finite and `>= 0`. |
| `loss_weight_kl` | number | Yes | None | Weight for teacher-target KL. Must be finite and `>= 0`; the CE and KL weights must sum to `> 0`. `loss_weight_kl: 0` is treated as disabled distillation and requires `loss_weight_ce: 1`. |
| `ensemble_strategy` | string | No | `"mean_logits"` | `"mean_logits"` averages raw teacher logits before softmax. `"mean_logprobs"` averages teacher log-probabilities geometrically, then renormalizes. |
| `temperature` | number | No | `1.0` | Distillation temperature. Teacher probabilities are computed from temperature-scaled logits and KL is scaled by `temperature^2`. |

V1 supports active distillation with `objective: "causal"`, `"mlm"`, `"mntp"`, and `"hybrid"` when the hybrid secondary objective is `"mlm"` or `"mntp"` and masked steps can run. It rejects multihead, block diffusion, data2vec, top-level `mtp`, first-byte masked loss, example framing, segment attention masking, and sequence-length schedules.

### Data2Vec EMA representation distillation

`training.data2vec` enables an online EMA teacher that runs on the unmasked batch and trains the student to regress normalized teacher hidden states at masked-objective positions. It is separate from `training.distillation`: data2vec uses an EMA copy of the current student weights and hidden-state SmoothL1 loss, while distillation uses fixed teacher checkpoints and token-distribution KL.

V1 supports `objective: "mlm"`, `objective: "mntp"`, and `objective: "hybrid"` when masked secondary steps can run. Hybrid causal batches skip the data2vec teacher forward and contribute zero data2vec loss; masked hybrid batches use the configured secondary objective. Lookahead submission is disabled while data2vec is active so the EMA teacher for step `s+1` includes the completed optimizer update from step `s`.

Current limitations: data2vec is training-only, rejects `training.distillation`, top-level `mtp`, `training.first_byte_mask`, recurrence schedules/weight sharing, U-Net, and parallel residual in v1, and uses a correctness-first CPU EMA readback/re-upload path.

```json
{
  "training": {
    "objective": "hybrid",
    "mlm_mask_token_id": 103,
    "hybrid_secondary_objective": "mntp",
    "data2vec": {
      "loss_weight": 0.1,
      "ema_tau": 0.999,
      "top_k_layers": 2,
      "smooth_l1_beta": 1.0,
      "target_norm": "layer_norm"
    }
  }
}
```

| Field | Type | Required | Default | Notes |
|------|------|----------|---------|-------|
| `loss_weight` | number | No | `1.0` | Weight added to the primary training loss. Explicit `0` disables data2vec without changing the graph. |
| `ema_tau` | number | No | `0.999` | EMA decay when no ramp is configured. Values in `[0,1]` are accepted; `1.0` freezes the teacher. |
| `ema_tau_start` / `ema_tau_end` | number | No | `ema_tau` | Linear tau ramp endpoints when `ema_tau_ramp_steps > 0`. |
| `ema_tau_ramp_steps` | integer | No | `0` | Number of steps over which tau ramps linearly, clamped after the ramp. |
| `top_k_layers` | integer | No | `8` | Number of final top-level block outputs to average for teacher targets. Must be `<= len(blocks)`. |
| `smooth_l1_beta` | number | No | `1.0` | Huber/SmoothL1 transition point. Must be `> 0`. |
| `target_norm` | string | No | `"layer_norm"` | `"layer_norm"`, `"instance_norm"`, and `"feature_norm"` all normalize each token vector over feature dimension; `"none"` is for debugging. |
| `target_norm_eps` | number | No | `1e-5` | Epsilon for target normalization. |
| `mask_source` | string | No | `"objective"` | V1 reuses the MLM/MNTP objective loss mask. |
| `mask_prob` | number | No | `0.0` | Reserved for future separate data2vec masks; must be in `[0,1]`. |
| `predictor_hidden_dim` | integer | No | `0` | `0` uses a single linear `D -> D` predictor; positive values use a two-layer GELU predictor. |

### Optimizer groups

The trainer classifies weights into four optimizer groups. Muon-family optimizers affect the matrix group only; `optimizer: "adamw"` and `optimizer: "lamb"` apply to all groups.

| Group | Optimizer | Typical weights | LR field | Weight-decay field |
|------|-----------|-----------------|----------|--------------------|
| Embedding | AdamW or LAMB | `embed`, `char_table`, `bigram_table`, `trigram_table` | `embed_lr` | `embed_weight_decay` |
| Head | AdamW or LAMB | `head` | `head_lr` | `head_weight_decay` |
| Scalar | AdamW or LAMB | Norm scales, decay vectors, learned scalar scales | `scalar_lr` | `scalar_weight_decay` |
| Matrix | Muon variant, AdamW, or LAMB | Projection and FFN matrices | `matrix_lr` | `matrix_weight_decay` |

S4D blocks with `state_lr` add metadata-driven adaptive groups: dt/A/B use
`state_lr` with no decay; C/D use global
`lr`. Under `weight_decay_policy: "all"`, C/D and other ordinary scalar/vector
parameters receive their configured decay, matching optimizers that do not
exclude biases and norm parameters.

### Training phases

Use `training.phases` to run multiple contiguous LR segments in one job. Each
phase applies its `lr` for its own `steps`, and the total training length is
the sum of all phase steps.

When `phases` is present:

- top-level `steps` and `lr` are ignored by the training loop
- `warmup_steps`, `warmup_ratio`, and `hold_steps` are ignored; model phase warmup/cooldown explicitly as phases when needed
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
    "warmup_ratio": 0.016,
    "hold_steps": 0,
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
- `residual_scale_init` can override a block's scale initialization. In parallel groups, setting a recurrent branch to `0.0` starts that branch as an additive no-op while keeping it trainable.

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

Fuses a `(plain or gated_deltanet, swiglu/geglu/moe)` pair into a parallel residual: both branches read the same pre-norm input and their outputs sum into the residual together (instead of the FFN branch reading the post-attention residual).

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

When the top-level flag is `true`, every consecutive pair must be `(plain or gated_deltanet, swiglu/geglu/moe)`.

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

- A pair start must be `plain` or `gated_deltanet`; the next block must be `swiglu`, `geglu`, or `moe`.
- Each paired FFN block drops its `ffn_norm_scale` or `moe_norm_scale` weight (the pair shares the start block's RMSNorm).
- `kv_source` is not supported on a paired `plain` block.
- Cannot be combined with `unet`.
- **Weight sharing:** when using `recurrence` (or `weight_group`) to share weights between blocks, paired and unpaired FFN blocks cannot share weights with each other — they have different shapes (paired blocks lack their own norm scale). Use distinct group names for paired vs. unpaired roles.

**Explicit heterogeneous groups.** Set `parallel_group` on the first block to run two or more contiguous branches from the same pre-norm input. This is useful for attention plus recurrent/SSM side branches:

```json
{
  "block_scales": true,
  "blocks": [
    {"type": "plain", "heads": 8, "attention_mask": "bidirectional", "parallel_group": 3},
    {"type": "hgrn2", "heads": 8, "residual_scale_init": 0.0},
    {"type": "geglu"}
  ]
}
```

V1 group rules:

- `parallel_group` must be set only on the first block and must be at least `2`.
- Supported token-mixer branches are `plain`, `gated_deltanet`, and `hgrn2`; at least two token mixers are required.
- A single optional FFN branch (`swiglu`, `geglu`, or `moe`) may appear only as the final group member.
- All followers share the first block's pre-norm and omit their own leading norm weight.
- `plain.kv_source`, top-level `parallel_residual: true`, U-Net, data2vec, and DWA are rejected with explicit errors in v1.

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

- `plain`, `swiglu`, `geglu`, `mlp`, `moe` experts, and `cross_attention` expand to `round(model_dim * mlp_mult)`, clamped to at least `model_dim`.

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
