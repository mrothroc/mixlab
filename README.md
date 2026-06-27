# mixlab

Explore ML architectures fast. Define a model in JSON, train it on your Mac
in seconds, then ship the same config to a cloud GPU for full-scale runs. One
JSON file, two platforms, no Python model code.

```text
laptop (Metal)                          cloud GPU (CUDA)
mixlab -config my_model.json    ===>    mixlab -config my_model.json
       -train 'data/*.bin'                     -train 'data/*.bin'
```

mixlab compiles JSON configs into a typed Go IR and executes them on GPU
through the MLX backend.

**Platforms:** macOS on Apple Silicon and Linux with NVIDIA CUDA via Docker.

## Quickstart

### macOS

Prerequisites:

- Apple Silicon Mac (M1 or later)
- Go 1.24+ ([go.dev/dl](https://go.dev/dl/))
- Python 3.10+ for data preparation
- Xcode Command Line Tools (`xcode-select --install`)

```bash
# 1. Set up Python environment and install dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install mlx numpy tokenizers

# 2. Build mixlab
make build

# 3. Download example data (~5 MB of public domain text)
bash scripts/download_example_data.sh

# 4. Train a 3-layer attention model
./mixlab -mode arch \
    -config examples/plain_3L.json \
    -train 'data/example/train_*.bin'
```

You should see training loss printed every 100 steps. The example config trains
for 200 steps and finishes in seconds.

### Docker / NVIDIA GPU

```bash
# Pull the pre-built CLI image, or build your own; see docker/README.md.
docker pull michaelrothrock/mixlab:latest

# Smoke test
docker run --gpus all michaelrothrock/mixlab:latest -mode smoke

# Train (mount your data directory)
docker run --gpus all -v $(pwd)/data:/data michaelrothrock/mixlab:latest \
    -mode arch -config /examples/plain_3L.json -train '/data/*.bin'
```

## Install

### Homebrew

```bash
brew install mrothroc/tap/mixlab
```

This installs MLX automatically as a dependency. For data preparation and
FineWeb downloads, also install Python dependencies:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy tokenizers datasets
```

### Build from source

Requires Go 1.24+ and MLX (`brew install mlx` or `pip install mlx`).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install mlx numpy tokenizers
make build
```

This produces a `mixlab` binary in the project root.

### Docker

For Linux with an NVIDIA GPU. The pre-built images support A100, A40,
RTX 3090, RTX 4090, L40, and L40S (sm_80/86/89). For other GPUs, see
[docker/README.md](docker/README.md) to build with your architecture.

## Features

- JSON-first model definition: no Go or Python changes required for most experiments.
- Built-in attention, state-space, recurrent, FFN, MoE, and custom block families.
- Causal, masked-LM, MNTP, hybrid, distillation, and data2vec-style training objectives.
- Architecture features such as GQA, DeBERTa relative attention, U-Net layouts,
  parallel residuals, recurrence, residual mixing, n-gram embeddings, character
  feature embeddings, and packed-sequence segment masks.
- Optimizers including Muon-style matrix optimization, AdamW, and LAMB.
- Safetensors import/export, Hugging Face export for supported model surfaces,
  quantization, evaluation exports, checkpointing, profiling, and Go API hooks.

## When to use mixlab

Good fit:

- Rapid architecture iteration: edit a JSON config, train on Metal, compare results.
- Mac-first workflows that later scale to CUDA with the same config.
- Comparing block families and objective variants on the same data.
- Teaching and research workflows where visible JSON configs make choices explicit.
- Fast block development through `arch.RegisterBlock()` or JSON custom blocks.

Not the right tool for:

- Production distributed training across many GPUs.
- Custom CUDA kernel development.
- Replacing a full training framework such as PyTorch or JAX.

mixlab is an architecture exploration tool, not a general-purpose training
framework. It trades generality for speed of iteration.

## Documentation

- [Docs index](docs/README.md)
- [CLI usage](docs/cli.md)
- [Data preparation](docs/data.md)
- [Architecture guide](docs/architecture.md)
- [Config guides](docs/config-model.md)
- [Configuration reference](docs/config-reference.md)
- [Examples](examples/README.md)
- [Hugging Face export](docs/hf-export.md)
- [Performance and profiling](docs/performance.md)
- [Docker](docker/README.md)
- [Releasing](docs/releasing.md)

## Common Commands

```bash
# Train one config
./mixlab -mode arch -config examples/plain_3L.json -train 'data/example/train_*.bin'

# Compare every config in examples/
./mixlab -mode arch_race -configs examples/ -train 'data/example/train_*.bin'

# Count parameters, blocks, FLOPs, and IR ops
./mixlab -mode count -config examples/plain_3L.json

# Evaluate a checkpoint
./mixlab -mode eval -config examples/plain_3L.json \
    -safetensors-load weights.st -train 'data/example/train_*.bin'

# Export a supported checkpoint to Hugging Face
./mixlab -mode export-hf -config examples/plain_3L.json \
    -safetensors-load weights.safetensors -export-dir runs/plain_3L/hf \
    -tokenizer-path data/example/tokenizer.json

# Verify the Hugging Face export against native Mixlab inference
./mixlab -mode parity -config examples/plain_3L.json \
    -safetensors-load weights.safetensors -hf runs/plain_3L/hf \
    -train 'data/example/val_*.bin'

# Generate token IDs from a checkpoint
./mixlab -mode generate -config examples/plain_3L.json \
    -safetensors-load weights.st -prompt token_ids:0,1,2
```

See [docs/cli.md](docs/cli.md) for the full mode and flag reference.
For long MLX runs, see [performance and profiling](docs/performance.md#mlx-memory-bounds)
for cache and memory-limit environment variables.

## Contributing

Before submitting changes, run `make test` from the repository root. The block
registry in [arch/registry.go](arch/registry.go) and config validation in
[arch/config.go](arch/config.go) must both be updated when adding a new block
type.

## License

MIT. See [LICENSE](LICENSE).

## Author

[Michael Rothrock](https://michael.roth.rocks) · [GitHub](https://github.com/mrothroc)

Other work:

- [Trust Topology](https://michael.roth.rocks/research/trust-topology/) — a framework for engineering reliability from unreliable AI agents
- [claude-code-log-analyzer](https://github.com/mrothroc/claude-code-log-analyzer) — compute overlap ratios on your own Claude Code session logs
