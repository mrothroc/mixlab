# mixlab — agent orientation

mixlab is a JSON-configurable training engine for transformer/SSM/hybrid models, lowered through a custom IR to MLX (Metal + CUDA backends).

## Build / test
```bash
CGO_ENABLED=1 go build -tags mlx -o mixlab ./cmd/mixlab/   # MLX-tagged build (production)
go build ./...                                              # stub build (no MLX)
golangci-lint run ./arch/... ./gpu/... ./train/... ./cmd/...
go test -tags mlx ./arch/... ./gpu ./train -count=1         # full MLX suite
CGO_ENABLED=0 go test ./arch/... ./data/... ./train/... -count=1   # CI command
```

## Layout
- [`arch/`](arch/CLAUDE.md) — IR builder, block emitters, registry, config schema
- [`gpu/`](gpu/CLAUDE.md) — IR dispatcher, MLX bridge, custom primitives, CUDA kernels
- [`train/`](train/CLAUDE.md) — trainer, optimizer, mamba3-aware code paths
- `data/`, `logprobs/`, `cmd/mixlab/` — supporting code; well-isolated, rarely touched
- `docker/` — Dockerfiles for runpod images; see also `docs/releasing.md`
- `examples/` — runnable JSON configs by feature
- `experiments/` — committed test configs (some need shards in `data/example/` which is gitignored)

## Reference docs
- [`docs/config-reference.md`](docs/config-reference.md) — full JSON schema, op list, training fields
- [`docs/canonical_mamba3.md`](docs/canonical_mamba3.md) — the canonical Mamba-3 block: paper alignment, 7-layer production stack, env vars
- [`docs/releasing.md`](docs/releasing.md) — release checklist (tag, formula, Homebrew, downstream)
- [`gpu/cuda_kernels/README.md`](gpu/cuda_kernels/README.md) — CUDA kernel build pipeline

## Conventions
- Run `gh run list --limit 1 --branch main` after every push to verify CI green. CI runs `CGO_ENABLED=0` and skips MLX-tagged tests.
- Tests with optional resources (data shards, GPU) must `t.Skip` not `t.Fatal` when resource missing.
- Don't push CUDA-kernel changes without smoke-testing on a CUDA host — GitHub CI doesn't have nvcc, so kernels aren't compiled there.
