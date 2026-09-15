# Docker builds for mixlab (CUDA)

mixlab runs natively on Apple Silicon (Metal/MLX). For NVIDIA GPUs, use Docker.
The current CUDA image is built from MLX v0.32.0 and requires NCCL support.
See [the MLX upgrade contract](../docs/mlx-0.32-upgrade.md) before rebuilding
the dependency layers.

## Quick start (pre-built images)

```bash
# Pull the pre-built image with MLX + CUDA (sm_80, sm_86, sm_89, sm_90)
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
    --build-arg MIXLAB_VERSION=dev \
    --build-arg VCS_REF="$(git rev-parse HEAD)" \
    -t mixlab .
```

### Image provenance

CLI and RunPod images carry OCI `version`, `revision`, and `source` labels:

```bash
docker image inspect michaelrothrock/mixlab:latest \
    --format '{{json .Config.Labels}}'
```

Cloud Build uses the trigger's release tag and commit for these labels. Main
branch builds use version `dev` plus the commit; `latest` is a mutable tag, not a
release identifier. For manual release builds, supply `_RELEASE_VERSION=vX.Y.Z`
and `_SOURCE_REVISION=<commit>` substitutions, or the corresponding Docker
build arguments `MIXLAB_VERSION` and `VCS_REF`. Direct builds default to `dev`
and `unknown` if the arguments are omitted. Record the image digest as well
when exact reproducibility matters.

## Pre-built images on Docker Hub

Each repository's Docker Hub page is generated from a file here, not edited in
the web form: `DOCKERHUB.md`, `DOCKERHUB-cuda.md`, and `DOCKERHUB-cuda-base.md`.
Cloud Build publishes them on every push to main. The first line of each file
carries the one-line description Docker Hub shows in search results:

```markdown
<!-- short: One line, 100 characters max. -->
```

It is required and stripped before the body is published. GPU architecture
claims in these files are pinned to `_ARCHS` by
`test_sync_dockerhub_description.py`, so they cannot drift from what the build
compiles.

| Image | Contents | Size |
|-------|----------|------|
| `michaelrothrock/mixlab-cuda-base` | Go + MLX + CUDA (sm_80 only) | ~6 GB |
| `michaelrothrock/mixlab-cuda` | + sm_86, sm_89, sm_90 architectures | ~8 GB |
| `michaelrothrock/mixlab` | + mixlab binary, Python with NumPy/tokenizers, example configs | ~9 GB |

## RunPod Serverless

mixlab ships with a separate RunPod serverless image that adds handler dependencies
and `scripts/handler.py` on top of the CLI image. Cloud Build publishes it to two
registries on every push to main:

- Artifact Registry: `us-central1-docker.pkg.dev/zapbox-cloud/parameter-golf/mixlab:runpod`
- Docker Hub: `michaelrothrock/mixlab:runpod`

To deploy a new endpoint:

1. Create a serverless endpoint at [runpod.io](https://www.runpod.io/)
2. Set the container image to `michaelrothrock/mixlab:runpod`
3. The handler starts automatically — it accepts JSON jobs via the RunPod API

### Updating the existing `mixlab` endpoint

The production `mixlab` endpoint pins the **Artifact Registry image by digest**, not
by tag, so a rebuild alone does not change what workers run. After Cloud Build
finishes, point the template at the new digest — workers then pull it on next start:

```bash
DIGEST=$(gcloud artifacts docker images describe \
    us-central1-docker.pkg.dev/zapbox-cloud/parameter-golf/mixlab:runpod \
    --format='value(image_summary.digest)')
```

Then update the template's `imageName` to `...parameter-golf/mixlab@${DIGEST}` and
cycle `workersMax` to force fresh workers. Because the reference is a digest, a
stale cached image cannot silently be served.

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
| `post` | Array of shell commands to run after mixlab exits. `$MIXLAB_CONFIG` is set to the config file path. |
| `safetensors` | Path to export weights after training |
| `safetensors_load` | Path to load weights before training |
| `resume` | Resume from a complete checkpoint directory, manifest, or companion file. Mutually exclusive with `safetensors_load` |
| `quantize` | `none`, `int8`, or `int6` |
| `output` | Output path for modes that write a file, such as `hiddenstats` |
| `checkpoint_dir` | Directory for periodic safetensors checkpoints |
| `checkpoint_every` | Save a checkpoint every N training steps |
| `max_tokens` | Maximum generated tokens for `generate` mode |
| `temperature` | Sampling temperature for `generate` mode. `0` selects deterministic greedy decoding |
| `env` | Environment variables for the mixlab process and all `setup`/`post` commands, e.g. `{"MIXLAB_TTT_MLP_DISABLE_CUDA_PRIMITIVE": "1"}`. Scoped to the job — it does not leak to later jobs on a warm worker |
| `timeout` | Positive finite wall-clock seconds per command (default 3600): applies independently to each setup command, the main process, and each post command. Not an idle-output timeout or a total-job budget |
| `timing` | Boolean; `true` forwards `-timing` (not a separate `true` argument) |
| `telemetry_out` | Optional path for periodic telemetry JSONL; use persistent storage for long runs |
| `stall_timeout` | Opt-in positive finite seconds without an increase in committed optimizer steps; single-process `arch` training only. Includes startup, validation, checkpointing and final export, so choose a threshold longer than their normal durations |
| `stall_dump_dir` | Required absolute persistent directory with `stall_timeout`; each run gets a unique subdirectory for native stacks and process diagnostics |

Both stdout and stderr stream to the RunPod dashboard while the command runs.
Stderr lines are labeled separately and remain separate in returned output.
The handler drains both pipes concurrently, including partial lines; silent
commands still time out. Timeout/error cleanup kills the command's process
group, including shell children, and reaps the direct child. Commands that
deliberately detach into a new session are outside that process group.
The returned capture retains the last 8 MiB of each stream per command,
prefixed with `[earlier output truncated by handler]` if older output was
dropped. Streaming continues after that limit; oversized console lines are
split into bounded chunks. RunPod can independently throttle dashboard logs.
Dashboard writes use one bounded background queue; a blocked or broken sink
cannot stop pipe draining, deadlines, or the watchdog. Dashboard lines can be
dropped under backpressure; returned stdout/stderr tails remain independent.
Use persistent files when complete long-run logs are required.

With `stall_timeout`, the handler watches a small atomic progress file on local
temporary storage, not stdout activity or GPU queries. Skipped optimizer updates
do not reset the deadline. On a stall it attempts a 30-second GDB all-thread dump,
captures `/proc` state and a bounded `nvidia-smi` snapshot, then kills/reaps the
trainer group and returns `error`, `exit_code`, output tails, and `diagnostics`
(the dump directory). Missing tools/ptrace permission are recorded, not treated
as successful stack capture. Diagnostic collection can add about 35 seconds to
the stall deadline. No automatic restart or checkpoint of a stuck GPU is attempted.
The hard `timeout` still applies independently; it does not promise a stack dump.

See [native stack capture on serverless workers](../docs/performance.md#native-stacks-on-runpod-serverless)
for investigating CPU-busy/GPU-idle hangs. Updating the binary alone does not
fix the wrapper: rebuild the RunPod image and update the endpoint's pinned
image digest so new workers run the updated scripts.

To continue an interrupted job, submit the same training config and data with
`"resume": "/runpod-volume/checkpoints"`. Resume restores training state, not
just weights; a missing or incomplete checkpoint produces an error rather than
silently starting fresh. The handler does not implement automatic resume.
The trainer validates checkpoint/config compatibility and restores the saved
learning-rate schedule. Extending `training.steps` does not restart the cosine
schedule; changing `lr_schedule_steps` is not a compatible resume.

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
- **Layer 3 (app):** Builds mixlab, installs preparation dependencies, and runs
  an embedded preparation smoke test. Rebuild on every code change; no MLX rebuild.

For day-to-day development, you rebuild only layer 3.

### Layer 1: Base image (CUDA + Go + MLX)

Compiles MLX from source with CUDA backend for sm_80 (A100).

```bash
docker build -f docker/base.Dockerfile -t mixlab-cuda-base:mlx-0.32.0 .
```

### Layer 2: Add GPU architectures

Add support for more GPU types. Each step only compiles the new kernels
(Ninja incremental build). Chain them:

```bash
# Add sm_86 (RTX 3090, A40)
docker build -f docker/addarch.Dockerfile \
    --build-arg BASE_IMAGE=mixlab-cuda-base:mlx-0.32.0 \
    --build-arg ARCHS="80;86" \
    -t mixlab-cuda:mlx-0.32.0 .

# Add sm_89 and sm_90 (RTX 4090/L40 and H100)
docker build -f docker/addarch.Dockerfile \
    --build-arg BASE_IMAGE=mixlab-cuda:mlx-0.32.0 \
    --build-arg ARCHS="80;86;89;90" \
    -t mixlab-cuda:mlx-0.32.0 .
```

### Layer 3: App image

Builds the mixlab Go binary and its preparation runtime.

```bash
docker build -f docker/app.Dockerfile \
    --build-arg BASE_IMAGE=mixlab-cuda:mlx-0.32.0 \
    -t mixlab:mlx-0.32.0 .
```

### RunPod image (optional)

Adds RunPod handler dependencies on top of the app image's Python environment.

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
| sm_90 | H100 | Layer 2 |

## Memory requirements

Compiling CUDA kernels is memory-intensive. Each GPU architecture adds ~2GB
peak RAM during compilation at `-j4`. For 4 architectures, use a machine with
at least 16GB RAM, or reduce parallelism by editing the Dockerfile (`ninja -j2`).

## Preparing data inside Docker

The CLI image includes Python 3.10+ and the NumPy/tokenizers dependencies from
`requirements-prepare.txt` in `/opt/mixlab/venv`, already on `PATH`. No source
checkout or RunPod handler is needed. Preparation runs on CPU, though the CUDA
binary still needs its driver libraries available in the container.

The image build checks embedded text and continuous-array preparation as a
non-root user, using a CPU-only build of the same CLI so the check needs no
NVIDIA driver. Failures stop the build. The check binary is not shipped.

Preparation reserves 10% for validation by default. Pass `-val-split 0` when
processing data that is already split and should be kept in full.

```bash
# Tokenize a text corpus
docker run --gpus all -v $(pwd)/corpus:/corpus -v $(pwd)/data:/data mixlab \
    -mode prepare -input /corpus/text.txt -output /data -vocab-size 1024

# Then train
docker run --gpus all -v $(pwd)/data:/data mixlab \
    -mode arch -config /examples/plain_3L.json -train '/data/train_*.bin'
```
