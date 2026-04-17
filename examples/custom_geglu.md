# custom_geglu -- Custom Block JSON Format Example

## Overview

The `custom` block type lets you define feed-forward blocks in JSON without
writing Go code. A custom block declares its own weight tensors (with
symbolic shape dimensions) and a sequence of IR operations that compose them.

This example defines a GeGLU-style gated feed-forward block.

## Config structure

```json
{
  "type": "custom",
  "name": "geglu",
  "weights": [
    {"name": "w_gate", "shape": ["D", "FFN"]},
    {"name": "w_up",   "shape": ["D", "FFN"]},
    {"name": "w_down", "shape": ["FFN", "D"]}
  ],
  "ops": [
    {"op": "matmul",  "inputs": ["x", "w_gate"], "output": "gate"},
    {"op": "silu",    "inputs": ["gate"],         "output": "gate_act"},
    {"op": "matmul",  "inputs": ["x", "w_up"],   "output": "up"},
    {"op": "mul",     "inputs": ["gate_act", "up"], "output": "ff"},
    {"op": "matmul",  "inputs": ["ff", "w_down"], "output": "ff_out"},
    {"op": "add",     "inputs": ["x", "ff_out"],  "output": "x"}
  ]
}
```

## Field reference

### Block-level fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | yes | Must be `"custom"` |
| `name` | string | yes | Human-readable block name (used in logging) |
| `weights` | array | yes | Weight tensor declarations |
| `ops` | array | yes | Sequence of IR operations to execute |

### Weight declaration

Each entry in `weights` declares a learnable parameter tensor.

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Weight name, referenced in `ops` inputs |
| `shape` | array of strings | Symbolic dimensions (see shape symbols below) |

### Op declaration

Each entry in `ops` describes one IR operation.

| Field | Type | Description |
|-------|------|-------------|
| `op` | string | IR opcode name (see supported ops below) |
| `inputs` | array of strings | Input tensor names (weights, intermediates, or `"x"` for block input) |
| `output` | string | Output tensor name. Use `"x"` to write back to the block's hidden state |

## Shape symbols

Shape dimensions use symbolic names that are resolved at IR build time based on
the model configuration.

| Symbol | Resolves to | Description |
|--------|------------|-------------|
| `D` | `model_dim` | Hidden dimension |
| `H` | `heads` | Number of attention heads |
| `HD` | `model_dim / heads` | Per-head dimension |
| `FFN` | `model_dim * 8 / 3` | Standard feed-forward inner dimension (~2.67x D) |
| `2D` | `2 * model_dim` | 2x hidden dimension |
| `4D` | `4 * model_dim` | 4x hidden dimension |
| `<float>D` | `float * model_dim` | Arbitrary multiplier (e.g. `1.5D`) |
| `T` | `seq_len` | Sequence length |
| `B` | batch size | Batch size |
| `V` | `vocab_size` | Vocabulary size |

Integer literals are also allowed for fixed-size dimensions.

## Supported IR opcodes

### Linear algebra
- `matmul` -- matrix multiply: `output = a @ b`

### Element-wise arithmetic
- `add` -- `output = a + b`
- `sub` -- `output = a - b`
- `mul` -- `output = a * b`
- `div` -- `output = a / b`
- `scalar_mul` -- `output = a * scalar`
- `square` -- `output = a^2`

### Activations
- `sigmoid` -- `output = sigmoid(a)`
- `silu` -- `output = a * sigmoid(a)` (also called swish)
- `gelu` -- `output = a * Phi(a)`
- `relu` -- `output = max(0, a)`
- `tanh` -- `output = tanh(a)`

### Reductions
- `softmax` -- softmax along last axis
- `mean_axis` -- mean along specified axis
- `cumsum` -- cumulative sum

### Shape manipulation
- `reshape` -- reshape tensor
- `transpose` -- transpose axes
- `slice` -- slice along axis
- `concat` -- concatenate tensors
- `squeeze` -- remove size-1 dimensions

### Math
- `sqrt`, `rsqrt`, `sin`, `cos`, `exp`, `outer`, `argsort`

### Special
- `rmsnorm` -- RMS normalization with learned scale
- `rope` -- rotary position embeddings
- `causal_mask` -- apply causal attention mask
- `cross_entropy` -- cross-entropy loss

## Data flow

The block receives its input as the tensor named `"x"`. Operations read from
and write to named tensors. The final value of `"x"` after all ops execute
becomes the block's output, which feeds into the next block.

Weight tensors declared in `weights` are available by name in any op's `inputs`.
Intermediate tensors created by one op's `output` are available to subsequent ops.

```
Block input ("x")
    |
    v
  matmul(x, w_gate) -> gate
  silu(gate)         -> gate_act
  matmul(x, w_up)   -> up
  mul(gate_act, up)  -> ff
  matmul(ff, w_down) -> ff_out
  add(x, ff_out)     -> x          <-- residual connection
    |
    v
Block output ("x")
```

## Design tips

1. **Always include a residual connection.** The last op should typically be
   `add(x, ...)` to write the residual sum back to `"x"`.

2. **Use `FFN` for inner dimensions.** The standard ~2.67x multiplier balances
   capacity against parameter count. Use `4D` for wider blocks if budget allows.

3. **Choose gate activations deliberately.** This example uses `silu` on the
   gate path; other custom blocks can use a different supported activation.

4. **Keep custom blocks simple.** Complex custom blocks are harder to debug.
   Start with a known architecture (like GeGLU) and modify incrementally.

## How to run

```bash
CGO_ENABLED=1 go build -tags mlx -o mixlab ./cmd/mixlab

./mixlab -mode arch \
  -config examples/custom_geglu.json \
  -train "data/example/train_*.bin"
```

## Status

This config requires the `custom` block type to be registered in mixlab's config
validator and IR builder. Until then, parsing will fail with an "invalid type"
error. See the corresponding custom block implementation for the
reference implementation.
