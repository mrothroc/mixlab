# Architecture guide

mixlab compiles JSON configs into a typed intermediate representation and runs
that IR on GPU.

```text
JSON config --> Go IR builder --> IR program (typed ops) --> MLX runtime
                                                         --> Metal (macOS)
                                                         --> CUDA (Linux)
```

The same config can run locally on Apple Silicon and on Linux CUDA backends.

## Config workflow

Start by copying `examples/plain_3L.json` and changing a few high-impact
fields:

| Knob | Field | Effect |
|------|-------|--------|
| Width | `model_dim` | Hidden size. Larger means more capacity and memory. |
| Depth | `blocks` | More blocks mean more capacity and runtime. |
| Heads | `heads` on `plain` blocks | Attention heads. Must divide `model_dim`. |
| Vocab | `vocab_size` | Must match the tokenizer used to create shards. |
| Context | `seq_len` | Sequence length in tokens. |

To try a different block family, change the `"type"` field in `blocks`. See
[config-reference.md](config-reference.md) for every field and
[../examples/README.md](../examples/README.md) for runnable configs.

## Block families

Built-in blocks include:

| Family | Blocks |
|--------|--------|
| Attention | `plain`, `cross_attention` |
| Feed-forward | `swiglu`, `geglu`, `mlp`, `moe` |
| State-space and recurrent | `mamba`, `gated_linear_ssm`, `mamba3-canonical`, `retnet`, `rwkv`, `hgrn2`, `mlstm` |
| Bottleneck and layout | `perceiver`, `bottleneck`, U-Net layout, parallel residuals |
| Token features | `token_blend`, n-gram embeddings, character feature embeddings |
| Extension | `custom`, external Go block registrations |

Important `plain` attention options include GQA via `kv_heads`, QK gain,
partial RoPE, DeBERTa-style relative attention, V-orthogonal XSA, shared K/V
sources, causal/bidirectional masks, and packed-segment masks.

## Training objectives

The training stack supports causal next-token modeling, masked language
modeling, MNTP, hybrid causal/masked objectives, fixed-teacher distillation,
and online EMA self-distillation through data2vec-style targets. Some
combinations are intentionally rejected until their semantics are explicit.

See [config-reference.md](config-reference.md#training) for the current
compatibility rules.

## Optimizers and training features

mixlab includes grouped optimizer support, Muon-style matrix optimization,
AdamW, LAMB, weight decay controls, gradient clipping, schedule knobs, SWA/EMA,
LoRA-TTT, quantization-aware training, safetensors import/export, and
validation/export modes.

Use `mixlab -mode count -config your_config.json` before longer runs to inspect
parameter count, active MoE parameter count, size estimates, FLOPs, and IR ops.

## Advanced architecture example

`examples/recurrent_parallel.json` demonstrates a broader config surface:
depth recurrence, parallel residuals, GQA, tied embeddings, and Muon-style
optimization.

```jsonc
{
  "name": "recurrent_parallel",
  "model_dim": 512,
  "vocab_size": 8192,
  "seq_len": 2048,
  "tie_embeddings": true,
  "parallel_residual": true,
  "block_scales": true,
  "resid_mix": true,
  "recurrence": [0,1,2,3,4,5,2,3,4,5]
}
```

See also `examples/unet_transformer.json` for U-Net skip connections and the
other examples for focused feature paths.

## Custom blocks

Custom blocks define weights and IR ops directly in JSON:

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

See `examples/custom_geglu.json`, `examples/custom_geglu.md`, and
[config-reference.md#custom-blocks](config-reference.md#custom-blocks) for the
schema, shape symbols, supported op reference, and constraints.

## Go extension points

Need ops or blocks beyond JSON custom blocks? Create a Go package that imports
`github.com/mrothroc/mixlab/arch`, registers block types through
`arch.RegisterBlock()`, and emits IR using the public op API such as
`prog.MatMul()` and `prog.RMSNorm()`.

Registered blocks compile into the same binary and inherit Metal/CUDA backends,
training loop integration, optimizers, safetensors support, checkpointing,
profiling, and CLI modes. See [../arch/registry.go](../arch/registry.go) for
the registration API.
