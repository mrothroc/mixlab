# Docker builds for mixlab (CUDA)

mixlab runs natively on Apple Silicon (Metal/MLX). For NVIDIA GPUs, use Docker.

## Quick start (pre-built images)

```bash
# Pull the pre-built image with MLX + CUDA (sm_80, sm_86, sm_89)
docker pull michaelrothrock/mixlab:latest

# Smoke test
docker run --gpus all michaelrothrock/mixlab -mode smoke

# Train with your data
docker run --gpus all -v $(pwd)/data:/data michaelrothrock/mixlab \
    -mode arch -config /examples/plain_3L.json -train '/data/*.bin'
```

If you want to rebuild the app layer yourself (e.g. with custom code changes):

```bash
docker pull michaelrothrock/mixlab-cuda:latest
docker build -f docker/app.Dockerfile \
    --build-arg BASE_IMAGE=michaelrothrock/mixlab-cuda:latest \
    -t mixlab .
```

## Pre-built images on Docker Hub

| Image | Contents | Size |
|-------|----------|------|
| `michaelrothrock/mixlab-cuda-base` | Go + MLX + CUDA (sm_80 only) | ~6 GB |
| `michaelrothrock/mixlab-cuda` | + sm_86, sm_89 architectures | ~8 GB |
| `michaelrothrock/mixlab` | + mixlab binary, Python, example configs | ~9 GB |

## RunPod Serverless

mixlab ships with a separate RunPod serverless image that adds Python and
`scripts/handler.py` on top of the CLI image. To deploy on RunPod:

1. Create a serverless endpoint at [runpod.io](https://www.runpod.io/)
2. Set the container image to `michaelrothrock/mixlab:runpod`
3. The handler starts automatically — it accepts JSON jobs via the RunPod API

### Sending jobs

```bash
# Smoke test
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT/run \
    -H 'Content-Type: application/json' \
    -H 'Authorization: Bearer YOUR_API_KEY' \
    -d '{"input": {"mode": "smoke"}}'

# Train with inline config
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT/run \
    -H 'Content-Type: application/json' \
    -H 'Authorization: Bearer YOUR_API_KEY' \
    -d '{
  "input": {
    "setup": ["bash /scripts/download_example_data.sh --output /data/example"],
    "mode": "arch",
    "config_json": {
      "name": "my_experiment",
      "model_dim": 256, "vocab_size": 1024, "seq_len": 512,
      "blocks": [
        {"type": "plain", "heads": 8, "kv_heads": 4},
        {"type": "swiglu"}
      ],
      "training": {"steps": 1000, "lr": 3e-4, "matrix_lr": 0.02,
                   "muon_momentum": 0.99, "seed": 42, "batch_tokens": 4096}
    },
    "train": "/data/example/train_*.bin",
    "timeout": 600
  }
}'

# Check status
curl https://api.runpod.ai/v2/YOUR_ENDPOINT/status/JOB_ID \
    -H 'Authorization: Bearer YOUR_API_KEY'
```

### Job input fields

| Field | Description |
|-------|-------------|
| `mode` | Any mixlab CLI mode supported by the container, such as `smoke`, `arch`, `arch_race`, `count`, `eval`, `hiddenstats`, or `generate` |
| `config_json` | Inline JSON config (alternative to `config` file path) |
| `config` | Path to a config file inside the container (e.g. `/examples/plain_3L.json`) |
| `train` | Glob pattern for training data shards |
| `setup` | Array of shell commands to run before training (e.g. data download) |
| `post` | Array of shell commands to run after mixlab exits |
| `safetensors` | Path to export weights after training |
| `safetensors_load` | Path to load weights before training |
| `quantize` | `none`, `int8`, or `int6` |
| `output` | Output path for modes that write a file, such as `hiddenstats` |
| `checkpoint_dir` | Directory for periodic safetensors checkpoints |
| `checkpoint_every` | Save a checkpoint every N training steps |
| `max_tokens` | Maximum generated tokens for `generate` mode |
| `temperature` | Sampling temperature for `generate` mode |
| `timeout` | Max seconds (default 3600) |

Logs stream to the RunPod dashboard in real time.

## Build everything from scratch

If you don't want to use the pre-built images, build the full stack yourself.
This takes ~45 minutes but requires only Docker and an internet connection.

### Why three layers?

Compiling MLX from source with CUDA takes ~30 minutes and produces ~6GB of
build artifacts. Without layers, every code change would trigger a full MLX
rebuild. The layered approach separates what changes rarely from what changes
often:

- **Layer 1 (base):** Go + MLX + CUDA for one architecture. Rebuild only when
  upgrading Go or MLX versions. ~30 min.
- **Layer 2 (addarch):** Adds GPU architectures incrementally. Ninja reuses
  existing object files — only new kernels compile. ~10 min per architecture.
- **Layer 3 (app):** Compiles only the mixlab Go binary. **~2 min.** Rebuild
  on every code change.

For day-to-day development, you rebuild only layer 3.

### Layer 1: Base image (CUDA + Go + MLX)

Compiles MLX from source with CUDA backend for sm_80 (A100).

```bash
docker build -f docker/base.Dockerfile -t mixlab-cuda-base .
```

### Layer 2: Add GPU architectures

Add support for more GPU types. Each step only compiles the new kernels
(Ninja incremental build). Chain them:

```bash
# Add sm_86 (RTX 3090, A40)
docker build -f docker/addarch.Dockerfile \
    --build-arg BASE_IMAGE=mixlab-cuda-base \
    --build-arg ARCHS="80;86" \
    -t mixlab-cuda .

# Add sm_89 (RTX 4090, L40)
docker build -f docker/addarch.Dockerfile \
    --build-arg BASE_IMAGE=mixlab-cuda \
    --build-arg ARCHS="80;86;89" \
    -t mixlab-cuda .
```

### Layer 3: App image

Builds the mixlab Go binary (~2 min).

```bash
docker build -f docker/app.Dockerfile \
    --build-arg BASE_IMAGE=mixlab-cuda \
    -t mixlab .
```

### RunPod image (optional)

Adds Python + RunPod handler on top of the app image.

```bash
docker build -f docker/runpod.Dockerfile \
    --build-arg APP_IMAGE=mixlab \
    -t mixlab:runpod .
```

## Supported GPU architectures

| Architecture | GPUs | Status |
|-------------|------|--------|
| sm_80 | A100, A30 | Layer 1 (base) |
| sm_86 | RTX 3090, A40, A6000 | Layer 2 |
| sm_89 | RTX 4090, L40, L40S | Layer 2 |
| sm_90 | H100 | Add with addarch |

To add sm_90 (H100):
```bash
docker build -f docker/addarch.Dockerfile \
    --build-arg ARCHS="80;86;89;90" \
    -t mixlab-cuda .
```

## Memory requirements

Compiling CUDA kernels is memory-intensive. Each GPU architecture adds ~2GB
peak RAM during compilation at `-j4`. For 3 architectures, use a machine with
at least 16GB RAM, or reduce parallelism by editing the Dockerfile (`ninja -j2`).

## Preparing data inside Docker

```bash
# Tokenize a text corpus
docker run --gpus all -v $(pwd)/corpus:/corpus -v $(pwd)/data:/data mixlab \
    -mode prepare -input /corpus/text.txt -output /data -vocab-size 1024

# Then train
docker run --gpus all -v $(pwd)/data:/data mixlab \
    -mode arch -config /examples/plain_3L.json -train '/data/train_*.bin'
```
