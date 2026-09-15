<!-- short: GPU ML architecture exploration. Define models in JSON, train on Metal or CUDA. -->
# mixlab

Explore ML architectures fast. Define a model in JSON, train it on your Mac in
seconds, then ship the same config to a cloud GPU for full-scale runs. One JSON
file, two platforms, no Python model code.

```text
laptop (Metal)                          cloud GPU (CUDA)
mixlab -config my_model.json    ===>    mixlab -config my_model.json
       -train 'data/*.bin'                     -train 'data/*.bin'
```

mixlab compiles JSON configs into a typed Go IR and executes them on GPU through
the MLX backend. This image is the CUDA half of that; on Apple Silicon install
with `brew install mrothroc/tap/mixlab`.

## Run it now

Example configs ship inside the image, so these work with no data and no setup
of your own:

```bash
# Verify MLX can see the GPU
docker run --gpus all michaelrothrock/mixlab:latest -mode smoke

# Parameters, blocks, FLOPs and IR ops for a bundled example
docker run --gpus all michaelrothrock/mixlab:latest \
    -mode count -config /examples/plain_3L.json
```

Train on your own data by mounting it. `WORKDIR` is `/data`:

```bash
docker run --gpus all -v $(pwd)/data:/data michaelrothrock/mixlab:latest \
    -mode arch -config /examples/plain_3L.json -train '/data/*.bin'
```

`ENTRYPOINT` is `mixlab`, so flags pass straight through and a bare
`docker run michaelrothrock/mixlab` prints the full flag reference.

## GPU requirements

`--gpus all` and the [NVIDIA Container
Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
are required. MLX JIT-compiles CUDA kernels at runtime and needs the driver
bind-mounted into the container; without it the GPU backend reports unavailable.

Pre-built images target **`sm_80`, `sm_86`, `sm_89`, and `sm_90`** — A100, A30,
A40, A6000, RTX 3090, RTX 4090, L4, L40, L40S, H100. Architectures are compiled into
`libmlx.so` at build time, so a tag supports exactly what it was built with.
Check any tag directly:

```bash
docker history --no-trunc michaelrothrock/mixlab:latest \
    | grep -o 'MLX_CUDA_ARCHITECTURES="[^"]*"'
```

For anything not listed, build with your own compute capability: see
[docker/README.md](https://github.com/mrothroc/mixlab/blob/main/docker/README.md).

## Tags

| Tag | What it is |
|-----|------------|
| `latest` | The mixlab CLI. Use this unless you are deploying to RunPod. |
| `runpod` | Adds a RunPod serverless handler that accepts JSON jobs with setup/post commands and all mixlab flags. Set it as the container image on a RunPod endpoint. |

Both tags embed the data-preparation runtime, so `mixlab -mode prepare` works
without a source checkout.

## What you can build with it

- **Block families:** attention, state-space, recurrent, FFN, MoE, and custom blocks.
- **Training objectives:** causal, masked-LM, MNTP, hybrid, distillation, and data2vec-style.
- **Architecture features:** GQA, DeBERTa relative attention, U-Net layouts, parallel
  residuals, recurrence, residual mixing, n-gram embeddings, character feature
  embeddings, and packed-sequence segment masks.
- **Optimizers:** Muon-style matrix optimization, AdamW, LAMB.
- **Interop:** safetensors import/export, Hugging Face export, quantization,
  evaluation exports, checkpointing, and profiling.

Useful modes beyond `-mode arch`:

```bash
# Race every config in a directory against the same data
-mode arch_race -configs /examples/ -train '/data/*.bin'

# Validate config fields and compatibility without initializing MLX
-mode validate -config /examples/plain_3L.json

# Export a checkpoint to Hugging Face, then verify it matches native inference
-mode export-hf  ...
-mode parity     ...
```

## When to use mixlab

Good fit:

- Rapid architecture iteration — edit JSON, train, compare.
- Mac-first workflows that later scale to CUDA with the same config.
- Comparing block families and objective variants on the same data.
- Teaching and research, where visible JSON configs make design choices explicit.

Not the right tool for:

- Production distributed training across many GPUs.
- Custom CUDA kernel development.
- Replacing a full training framework such as PyTorch or JAX.

mixlab is an architecture exploration tool, not a general-purpose training
framework. It trades generality for speed of iteration.

## Source, docs, and license

- **GitHub:** [mrothroc/mixlab](https://github.com/mrothroc/mixlab) — MIT licensed
- **Docs index:** [docs/README.md](https://github.com/mrothroc/mixlab/blob/main/docs/README.md)
- **CLI reference:** [docs/cli.md](https://github.com/mrothroc/mixlab/blob/main/docs/cli.md)
- **Recipes:** [docs/recipes.md](https://github.com/mrothroc/mixlab/blob/main/docs/recipes.md) — reproducible config → trained → Hugging Face runs
- **Docker details:** [docker/README.md](https://github.com/mrothroc/mixlab/blob/main/docker/README.md)

Images carry OCI labels for `version`, `revision`, and `source`, so you can trace
any tag back to the commit that built it:

```bash
docker inspect --format '{{json .Config.Labels}}' michaelrothrock/mixlab:latest
```

## Author

Built by [Michael Rothrock](https://michael.roth.rocks). Related work:

- [Trust Topology](https://michael.roth.rocks/research/trust-topology/) — engineering
  reliable systems from unreliable AI agents
- [claude-code-log-analyzer](https://github.com/mrothroc/claude-code-log-analyzer) —
  compute overlap ratios on your own Claude Code session logs
