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
| Attention shape | `heads`, `kv_heads`, `attention_mask`, `window_size` |
| Position handling | `rope_dims`, `rope_convention`, `relative_attention`, `relative_attention_window`, `relative_attention_parameterization` |
| Projection extras | `attn_bias`, `attn_value_gate`, `qk_norm`, `qk_gain`, `xsa`, `sparse_attn_gate` |
| FFN tail | `ffn_activation`, `ffn_pre_norm`, `ffn_bias` |
| Composition | `kv_source`, `skip_attention`, `parallel_residual`, `weight_group` |

For strict GPT-2-compatible blocks, use learned absolute positions, affine
LayerNorm, tied embeddings, `attn_bias: true`, `ffn_pre_norm: true`,
`ffn_bias: true`, and `ffn_activation: "gelu_new"` or `"gelu"`.

## Custom Blocks

Custom blocks are documented in the full reference under
[Custom blocks](config-reference.md#custom-blocks), including weight
declarations, supported ops, shape symbols, and examples.
