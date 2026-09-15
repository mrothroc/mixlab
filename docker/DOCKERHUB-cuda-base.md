# mixlab-cuda-base

**Most people want [`michaelrothrock/mixlab`](https://hub.docker.com/r/michaelrothrock/mixlab)
instead.** That image has the mixlab binary in it and runs out of the box. This
one is the first of two dependency layers beneath it, and contains no mixlab
binary.

It carries Go, the CUDA toolkit, and MLX compiled for **sm_80 only** (A100, A30).
It exists so GPU architectures can be added incrementally instead of recompiling
MLX from scratch for every target — each additional architecture costs minutes
rather than the roughly half hour a full MLX CUDA build takes.

If you just want a prebuilt layer that already covers current datacenter and
workstation cards, use
[`mixlab-cuda`](https://hub.docker.com/r/michaelrothrock/mixlab-cuda).

## Adding architectures

Pass the **full** list each time, including ones already built. The build re-runs
cmake against the whole list, and Ninja reuses existing object files, so
previously-built architectures are cheap to re-link.

```bash
docker build -f docker/addarch.Dockerfile \
    --build-arg BASE_IMAGE=michaelrothrock/mixlab-cuda-base:latest \
    --build-arg ARCHS="80;86" \
    -t mixlab-cuda:local .
```

Chain it to go further — `base(80)` → `addarch(80;86)` → `addarch(80;86;89)`.

| Capability | Cards |
|------------|-------|
| sm_80 | A100, A30 |
| sm_86 | RTX 3090, A40, A6000, A5000, A4000, A4500 |
| sm_89 | RTX 4090, L4, L40, L40S |
| sm_90 | H100 |

Each concurrent architecture needs roughly 2 GB of RAM while compiling, so budget
about 4 GB per concurrent architecture at `-j4`. Building four at once wants a
machine with 16 GB or more and real disk headroom.

## What is in it

Ubuntu 22.04 with the CUDA toolkit, Go, CMake, OpenBLAS/LAPACK, cuDNN, and NCCL,
plus MLX built from a pinned source release with the CUDA backend enabled. Around
6 GB.

MLX is pinned to an immutable upstream commit, not just a tag, and the build
fails if the two disagree. The image reports both:

```bash
docker run --rm michaelrothrock/mixlab-cuda-base:latest \
    env | grep MIXLAB_MLX
```

The add-architecture build checks those values against its own expectations and
refuses to run against a mismatched base, so a stale layer cannot quietly produce
a broken image.

## Layers

| Image | Contents | Size |
|-------|----------|------|
| **`mixlab-cuda-base`** | Go + MLX + CUDA, sm_80 only | ~6 GB |
| [`mixlab-cuda`](https://hub.docker.com/r/michaelrothrock/mixlab-cuda) | + additional GPU architectures | ~8 GB |
| [`mixlab`](https://hub.docker.com/r/michaelrothrock/mixlab) | + mixlab binary, Python runtime, example configs | ~9 GB |

## About mixlab

Explore ML architectures fast: define a model in JSON, train it on your Mac in
seconds, then ship the same config to a cloud GPU. One JSON file, two platforms,
no Python model code.

[GitHub](https://github.com/mrothroc/mixlab) · MIT licensed ·
built by [Michael Rothrock](https://michael.roth.rocks)
