# MLX 0.32 Upgrade

Mixlab is tested against MLX `>=0.32.0 <0.33.0`, the range `Formula/mixlab.rb`
asserts at install time. Treat "or newer" as unverified rather than safe: MLX has
changed autodiff behavior in a patch release (0.32.1 made `take_along_axis` reject a
VJP with respect to its indices, breaking MoE and bf16 training until those indices
were wrapped in `stop_gradient`). Run the `-tags mlx` suite before widening the range. The CUDA images pin the exact upstream
release:

```text
tag:    v0.32.0
commit: 7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247
```

This upgrade is the R0 prerequisite for distributed training. It does not add
distributed configuration, process-group initialization, gradient
collectives, or coordinator behavior.

## Runtime Contract

The native bridge enforces MLX 0.32.0 as its minimum build version. `mixlab
-mode smoke` reports the linked runtime version and requires the communication
backend expected for the platform:

- macOS/Metal: TCP `ring`
- Linux/CUDA: `nccl`

The capability check uses MLX's availability API and does not initialize a
distributed process group.

## CUDA Image Build

MLX treats NCCL as optional and compiles a no-NCCL stub when the library is not
found. NVIDIA's CUDA development image provides CUDA-matched, held NCCL
development/runtime packages. Mixlab preserves that version, validates its
headers and library, and fails the image build unless CMake found `libnccl`.

An MLX upgrade must rebuild the base source layer. Running only
`addarch.Dockerfile` is insufficient because that layer deliberately reuses
the `/opt/mlx` source and object tree from its base image.

Build locally:

```bash
docker build -f docker/base.Dockerfile -t mixlab-cuda-base:mlx-0.32.0 .
docker build -f docker/addarch.Dockerfile \
  --build-arg BASE_IMAGE=mixlab-cuda-base:mlx-0.32.0 \
  --build-arg 'ARCHS=80;86;89;90' \
  -t mixlab-cuda:mlx-0.32.0 .
docker build -f docker/app.Dockerfile \
  --build-arg BASE_IMAGE=mixlab-cuda:mlx-0.32.0 \
  -t mixlab:mlx-0.32.0 .
```

Cloud Build must run the base build before the architecture build:

```bash
gcloud builds submit \
  --config=docker/cloudbuild-mlx-cuda-base.yaml \
  --timeout=14400s .
gcloud builds submit \
  --config=docker/cloudbuild-golf-mlx-cuda.yaml \
  --timeout=14400s .
```

For pre-promotion validation, build the application with versioned tags:

```bash
gcloud builds submit \
  --config=docker/cloudbuild-ci.yaml \
  --substitutions='_REGISTRY_PREFIX=us-central1-docker.pkg.dev/zapbox-cloud/parameter-golf,_MLX_BASE_IMAGE=us-central1-docker.pkg.dev/zapbox-cloud/parameter-golf/golf-mlx-cuda:mlx-0.32.0,_IMAGE_TAG=mlx-0.32.0-r0,_RUNPOD_TAG=mlx-0.32.0-r0-runpod' \
  .
```

The normal application trigger can promote the dependency by pointing
`_MLX_BASE_IMAGE` at the versioned `golf-mlx-cuda:mlx-0.32.0` image. Its
default `latest` and `runpod` output tags are unchanged.

## Acceptance

Metal:

```bash
go test ./...
go test -tags mlx ./gpu ./train -count=1
go run -tags mlx ./cmd/mixlab -mode smoke
```

CUDA:

```bash
mixlab -mode smoke
go test -tags mlx ./gpu ./train -count=1
```

CUDA release evidence must include the actual runtime output showing MLX
0.32.x and `MLX distributed backend available: nccl`. A successful image build
without a GPU is not enough.

Representative custom primitive coverage must include canonical Mamba-3,
Gated Delta, and TTT-MLP in addition to a plain-attention training smoke.
Checkpoint loading should also be checked in both Metal-to-CUDA and
CUDA-to-Metal directions.
