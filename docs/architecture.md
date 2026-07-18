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

## Model input boundary

Current models use the discrete-token input adapter: integer token IDs are
looked up in the model embedding table, optional learned positions and sparse
feature channels are applied, and the result is flattened to `[B*T, model_dim]`
before block 0. Single-head and multihead builders share this adapter emitter,
so their input weight ordering and feature composition cannot drift.

There is intentionally no public `input_adapter` config field yet because this
release has only one useful runtime implementation. The adapter becomes a
public choice when continuous frame projection is added. Dataset representation
and modality are recorded independently in the optional
[`mixlab.dataset.json`](data.md#dataset-manifest) artifact.

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
online EMA self-distillation through data2vec-style targets, and block-wise
masked diffusion. Some combinations are intentionally rejected until their
semantics are explicit.

Block diffusion is modeled as an objective and sampler over an existing
backbone, not as a block type. v1 trains with a prefix-plus-block attention
mask and a masked cross-entropy loss, and generates with `-mode
generate-diffusion` (block-wise confidence-based denoising). The config surface
is limited to sequential `plain` self-attention plus `swiglu`, `geglu`, `mlp`,
and `moe` blocks. Mamba and other recurrent/SSM mixers, cross-attention, custom
blocks, segment masks, windowed attention, data2vec, distillation, and
diffusion-aware HF export are deferred until their denoising semantics have
dedicated implementation and tests. Use matching causal and MLM configs as
baselines when evaluating diffusion experiments.

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
