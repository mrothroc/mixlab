# mixlab

A general-purpose ML architecture exploration tool. Define model architectures
in JSON, compile them to a typed Go IR, and train them on GPU through the MLX
backend without writing Python. `mixlab` supports plain transformer stacks,
fully custom JSON-defined blocks.

**Platforms:** macOS (Apple Silicon) and Linux (NVIDIA CUDA via Docker). Windows is not supported.

## Quickstart (macOS)

```bash
# 1. Install Go 1.24+ (https://go.dev/dl/) and MLX
pip install mlx

# 2. Build mixlab
make build

# 3. Download example training data (~5 MB from Project Gutenberg)
bash scripts/download_example_data.sh

# 4. Train a 3-layer attention model
./mixlab -mode arch \
    -config examples/plain_3L.json \
    -train 'data/example/train_*.bin'
```

You should see training loss printed every 100 steps. The example config trains
for 200 steps and finishes in seconds.

## Quickstart (Docker / NVIDIA GPU)

```bash
# Pull the pre-built base image (or build your own — see docker/README.md)
docker build -f docker/app.Dockerfile -t mixlab .

# Smoke test
docker run --gpus all mixlab -mode smoke

# Train (mount your data directory)
docker run --gpus all -v $(pwd)/data:/data mixlab \
    -mode arch -config /examples/plain_3L.json -train '/data/*.bin'
```

## Features

- JSON-first model definition: no Go or Python changes required for most experiments.
- Built-in block families: `plain`, `swiglu`, `mamba`, `mamba3`, `retnet`,
  `rwkv`, `perceiver`, `bottleneck`, `cross_attention`, `token_blend`, `custom`.
  residual mixing, tied embeddings, hashed bigram embeddings, configurable MLP width.
- Trainer features: grouped optimizer settings, Muon for matrix weights, AdamW
  for scalar/head/embed groups, SWA/EMA averaging, validation-loss early
  stopping via `training.target_val_loss`, safetensors import/export.
- Custom blocks: declare weights and op graphs directly in JSON.

## Install

### macOS (Apple Silicon)

Requires Go 1.24+ and MLX. The Makefile handles compiler flags.

```bash
# Install MLX (if not already present)
pip install mlx

# Build
make build
```

This produces a `mixlab` binary in the project root.

### Docker (NVIDIA CUDA)

For Linux or any machine with an NVIDIA GPU:

```bash
# Build the container (from the repo root)
docker build -t mixlab-cuda -f docker/app.Dockerfile .

# Run smoke test
docker run --gpus all mixlab-cuda -mode smoke

# Train a model (mount your data directory)
docker run --gpus all -v $(pwd)/data:/data mixlab-cuda \
    -mode arch -config /examples/plain_3L.json -train '/data/train_*.bin'
```

## Usage

mixlab has four modes, selected with `-mode`:

| Mode | Description |
|------|-------------|
| `arch` | Train a single architecture from a JSON config. The default mode. |
| `arch_race` | Train every JSON config in a directory and compare results. |
| `smoke` | Run diagnostic checks (MLX availability, GPU health). |
| `prepare` | Tokenize raw text or JSONL into binary training shards. |

### arch (default)

```bash
./mixlab -mode arch -config examples/plain_3L.json -train 'data/example/train_*.bin'
```

Additional flags:

| Flag | Description |
|------|-------------|
| `-eval` | Run full validation BPB evaluation after training |
| `-safetensors FILE` | Export weights to safetensors after training |
| `-safetensors-load FILE` | Load weights before training (resume or eval-only) |
| `-quantize MODE` | Weight quantization: `none` (default), `int8`, `int6` |
| `-lut-dir DIR` | Directory for BPB lookup tables (default: `data`) |

### arch_race

```bash
./mixlab -mode arch_race -configs examples/ -train 'data/example/train_*.bin'
```

Trains every `.json` config in the given directory and prints a ranked summary.

### prepare

```bash
./mixlab -mode prepare -input raw_text/ -output data/shards/ -vocab-size 1024
```

| Flag | Description |
|------|-------------|
| `-input` | Input text file, JSONL file, or directory |
| `-output` | Output directory for binary shards |
| `-vocab-size` | BPE vocabulary size (default: 1024) |
| `-val-split` | Fraction of tokens reserved for validation (default: 0.1) |
| `-tokenizer-path` | Path to a pre-trained `tokenizer.json` (optional) |
| `-text-field` | JSON field name for text in JSONL input (default: `text`) |

## Architecture

```text
JSON config --> Go IR builder --> IR program (typed ops) --> MLX runtime
                                                         --> Metal (macOS)
                                                         --> CUDA (Linux)
```

The Go IR builder compiles JSON configs into a typed intermediate
representation. The runtime executes that IR on GPU and applies grouped
optimization: AdamW for embedding/head/scalar groups and Muon for matrix
weights.

Sequential models can run as plain stacks or as U-Net layouts with learned skip
frequency streams. Recent config features such as `block_scales`,
`resid_mix`, `tie_embeddings`, and `mlp_mult` all lower directly into the IR.

## Block types

These built-in block types are available in JSON configs:

| Type | Description |
|------|-------------|
| `plain` | Causal self-attention + FFN. Requires `heads`. Optional `kv_heads` for grouped-query attention. |
| `swiglu` | SwiGLU feed-forward block with residual connection. |
| `mamba` | Mamba selective state-space block. Optional `inner_dim`. |
| `mamba3` | Mamba-3 style gated scan block with learned delta-t gating. Optional `inner_dim`. |
| `retnet` | RetNet retention block. Requires `heads`. Optional `decay` field in config. |
| `rwkv` | RWKV-style recurrent mixing block. |
| `perceiver` | Perceiver latent bottleneck. Requires `heads`. Optional `num_latents`. |
| `bottleneck` | Smaller latent bottleneck block. Requires `heads`. Optional `num_latents`. |
| `cross_attention` | Cross-attention from the current stream into `source_stream`. Requires `heads`, `source_stream`. |
| `token_blend` | Learned token blending gate over adjacent positions. |
| `custom` | User-defined block with declared weights and IR ops. |


## Advanced Architecture Example

`examples/recurrent_parallel.json` demonstrates the most advanced config surface —
depth recurrence, parallel residuals, GQA, tied embeddings, and Muon optimizer:

```jsonc
{
  "name": "recurrent_parallel",
  "model_dim": 512,
  "vocab_size": 8192,
  "seq_len": 2048,
  "tie_embeddings": true,       // head reuses embedding weights
  "parallel_residual": true,    // attn + MLP see same input
  "block_scales": true,         // learned per-block output scaling
  "resid_mix": true,            // learned blend with original embedding
  "recurrence": [0,1,2,3,4,5,2,3,4,5]  // layers 3-4 reuse weights from 1-2
}
```

See also `examples/unet_transformer.json` for a U-Net architecture with skip
connections, and all other examples for commented configs covering every feature.

## Custom blocks

Custom blocks let you define novel architectures entirely in JSON.

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

See `examples/custom_geglu.json` for a runnable example and
[`docs/config-reference.md`](docs/config-reference.md) for the full JSON schema,
shape symbols, op list, training fields, and v0.7 feature notes.

## Performance

On Apple M1 Max (Metal): ~8.5 seconds per 100 training steps at `d=1024`,
`seq_len=1024`. Smaller models (`d=128`) train in seconds.

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines on submitting
changes, code style, testing, and adding new block types.

## License

MIT. See [LICENSE](LICENSE).
