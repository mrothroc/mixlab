# mixlab-cuda

**Most people want [`michaelrothrock/mixlab`](https://hub.docker.com/r/michaelrothrock/mixlab)
instead.** That image has the mixlab binary in it and runs out of the box. This
one is a build layer: Go, CUDA, and MLX compiled for several GPU architectures,
with no mixlab binary.

Reach for it when you want to rebuild the application layer against your own
code changes without recompiling MLX, which takes roughly half an hour.

```bash
docker pull michaelrothrock/mixlab-cuda:latest

docker build -f docker/app.Dockerfile \
    --build-arg BASE_IMAGE=michaelrothrock/mixlab-cuda:latest \
    --build-arg MIXLAB_VERSION=dev \
    --build-arg VCS_REF="$(git rev-parse HEAD)" \
    -t mixlab .
```

## What is in it

Ubuntu 22.04 with the CUDA toolkit, Go, CMake, OpenBLAS/LAPACK, cuDNN, NCCL, and
MLX built from a pinned source release with the CUDA backend enabled. Around
8 GB.

The image describes its own MLX build rather than relying on this page staying
current:

```bash
docker run --rm michaelrothrock/mixlab-cuda:latest \
    env | grep MIXLAB_MLX
```

That prints the pinned MLX version and the exact upstream commit it was built
from. The add-architecture build refuses to run against a base whose MLX version
or commit disagrees, so the layers cannot silently drift apart.

## GPU architectures

This layer adds compute capabilities on top of the sm_80-only
[`mixlab-cuda-base`](https://hub.docker.com/r/michaelrothrock/mixlab-cuda-base).
The architectures are fixed at build time — they are compiled into `libmlx.so` —
so a tag supports exactly what it was built with and nothing more.

| Capability | Cards |
|------------|-------|
| sm_80 | A100, A30 |
| sm_86 | RTX 3090, A40, A6000, A5000, A4000, A4500 |
| sm_89 | RTX 4090, L4, L40, L40S |
| sm_90 | H100 |

To confirm what a specific tag actually carries, read it out of the image's own
build history rather than trusting a table:

```bash
docker history --no-trunc michaelrothrock/mixlab-cuda:latest \
    | grep -o 'MLX_CUDA_ARCHITECTURES="[^"]*"'
```

Need a capability that is not listed? Build it yourself — the layer is designed
for exactly that, and Ninja reuses the existing object files so only the new
architecture's kernels compile. See
[docker/README.md](https://github.com/mrothroc/mixlab/blob/main/docker/README.md).

## Layers

| Image | Contents | Size |
|-------|----------|------|
| [`mixlab-cuda-base`](https://hub.docker.com/r/michaelrothrock/mixlab-cuda-base) | Go + MLX + CUDA, sm_80 only | ~6 GB |
| **`mixlab-cuda`** | + additional GPU architectures | ~8 GB |
| [`mixlab`](https://hub.docker.com/r/michaelrothrock/mixlab) | + mixlab binary, Python runtime, example configs | ~9 GB |

Despite the name, `mixlab-cuda-base` is the *smaller, more primitive* of the two
dependency layers. This image is built on top of it.

## About mixlab

Explore ML architectures fast: define a model in JSON, train it on your Mac in
seconds, then ship the same config to a cloud GPU. One JSON file, two platforms,
no Python model code.

[GitHub](https://github.com/mrothroc/mixlab) · MIT licensed ·
built by [Michael Rothrock](https://michael.roth.rocks)
