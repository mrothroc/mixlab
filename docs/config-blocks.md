# Config: Blocks

Blocks define the model's executable stack. This page is a map of the block
families; use [config-reference.md](config-reference.md#block-types) for every
field and validation rule.

## Common Stack Patterns

```json
{
  "blocks": [
    {"type": "plain", "heads": 8},
    {"type": "swiglu"},
    {"type": "plain", "heads": 8},
    {"type": "moe", "num_experts": 4, "top_k": 2, "expert_block": {"type": "swiglu"}}
  ]
}
```

Token mixers such as `plain`, recurrent/SSM blocks, `hgrn2`, and `mlstm` are
normally paired with FFN/channel mixers such as `swiglu`, `geglu`, `mlp`, or
`moe`.

## Block Families

| Block | Role |
|------|------|
| `plain` | Transformer attention block with optional FFN tail. |
| `swiglu`, `geglu`, `mlp` | Dense FFN/channel mixers. |
| `moe` | Top-k routed FFN replacement with load-balancing auxiliary loss. |
| `mamba`, `mamba3-canonical`, `gated_linear_ssm` | State-space token mixers. |
| `retnet`, `rwkv`, `gated_deltanet` | Retention/recurrent token mixers. |
| `hgrn2`, `mlstm` | Correctness-first recurrent token mixers. |
| `perceiver`, `bottleneck`, `cross_attention`, `token_blend` | Specialized mixing/adapter blocks. |
| `custom` | JSON-declared block using the supported custom op surface. |

## `plain` Attention Clusters

`plain` has the largest field surface. Think of it in clusters:

| Cluster | Fields |
|------|--------|
| Attention shape | `heads`, `kv_heads`, `attention_mask`, `window_size`, `differential_attention`, `differential_lambda_init` |
| Position handling | `rope_dims`, `rope_convention`, `relative_attention`, `relative_attention_window`, `relative_attention_parameterization` |
| Projection extras | `attn_bias`, `attn_value_gate`, `qk_norm`, `qk_gain`, `xsa`, `sparse_attn_gate` |
| FFN tail | `ffn_activation`, `ffn_pre_norm`, `ffn_bias` |
| Composition | `kv_source`, `skip_attention`, `parallel_residual`, `parallel_group`, `residual_scale_init`, `weight_group` |

For strict GPT-2-compatible blocks, use learned absolute positions, affine
LayerNorm, tied embeddings, `attn_bias: true`, `ffn_pre_norm: true`,
`ffn_bias: true`, and `ffn_activation: "gelu_new"` or `"gelu"`.

For DIFF Transformer-style attention, set `differential_attention: true` on
`plain`. The `heads` value is the number of differential heads, so use half the
ordinary baseline head count when matching parameter/FLOP scale.

## Parallel Hybrid Branches

Use `parallel_group` on the first block when attention and recurrent/SSM
branches should read the same pre-norm input and contribute additively:

```json
[
  {"type": "plain", "heads": 8, "attention_mask": "bidirectional", "parallel_group": 3},
  {"type": "hgrn2", "heads": 8, "residual_scale_init": 0.0},
  {"type": "geglu"}
]
```

`residual_scale_init: 0.0` requires `block_scales: true` and starts that
branch as an additive no-op. V1 supports `plain`, `gated_deltanet`, and
`hgrn2` token-mixer branches plus an optional final `swiglu`, `geglu`, or
`moe` FFN branch.

## Custom Blocks

Custom blocks are documented in the full reference under
[Custom blocks](config-reference.md#custom-blocks), including weight
declarations, supported ops, shape symbols, and examples.
