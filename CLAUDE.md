# mixlab — agent orientation

mixlab is a JSON-configurable training engine for transformer/SSM/hybrid models, lowered through a custom IR to MLX (Metal + CUDA backends).

## Build / test
```bash
CGO_ENABLED=1 go build -tags mlx -o mixlab ./cmd/mixlab/   # MLX-tagged build (production)
go build ./...                                              # stub build (no MLX)
golangci-lint run ./arch/... ./cmd/mixlab ./data/... ./gpu/... ./train/...
go test -tags mlx ./arch/... ./gpu ./train -count=1         # full MLX suite (needs Metal/CUDA)
CGO_ENABLED=0 go test ./arch/... ./cmd/mixlab ./data/... ./train/... -count=1   # CI command (no MLX)
```
`mixlab -mode validate -config model.json` builds the IR from a config without touching data or the GPU — the fastest correctness check.

## Layout
- [`arch/`](arch/CLAUDE.md) — IR builder, block emitters, registry, config schema, objectives
- [`gpu/`](gpu/CLAUDE.md) — IR dispatcher, MLX bridge, custom primitives, CUDA kernels
- [`train/`](train/CLAUDE.md) — trainer, optimizer, objective batching, resume, generation
- [`data/`](data/CLAUDE.md) — shard loaders + binary formats, dataset manifest, tokenizer/nucleotide artifacts (a core subsystem since the sequence-modality work — no longer "rarely touched")
- `cmd/mixlab/` — CLI entrypoint; one `-mode` per task (see the mode list in `main.go` and `docs/cli.md`)
- `scripts/` — Python data prep (`prepare.py`, `prepare_records.py`): text/JSONL/FASTA → shards + manifest
- `docker/` — Dockerfiles for RunPod images; see also `docs/releasing.md`
- `examples/` — runnable JSON configs by feature; `experiments/` — committed test configs (some need gitignored `data/example/` shards)

## Reference docs
- [`docs/README.md`](docs/README.md) — **the doc index**: task → doc routing; start here for anything user-facing
- [`docs/feature-matrix.md`](docs/feature-matrix.md) — capability map across training / native runtime / HF export
- [`docs/config-reference.md`](docs/config-reference.md) — full JSON schema, op list, training fields (kept complete by CI doc guards)
- [`docs/canonical_mamba3.md`](docs/canonical_mamba3.md) — canonical Mamba-3 block: paper alignment, 7-layer production stack, env vars
- [`docs/releasing.md`](docs/releasing.md) — release checklist (tag, formula, Homebrew, RunPod)
- [`gpu/cuda_kernels/README.md`](gpu/cuda_kernels/README.md) — CUDA kernel build pipeline
- Deep dives: [`docs/data.md`](docs/data.md), [`docs/hf-export.md`](docs/hf-export.md), [`docs/grammar-constrained-generation.md`](docs/grammar-constrained-generation.md), [`docs/ttt-mlp-stateful-inference.md`](docs/ttt-mlp-stateful-inference.md)

## Conventions
- Run `gh run list --limit 1 --branch main` after every push to verify CI green. CI runs `CGO_ENABLED=0`, covers `./cmd/mixlab`, and skips MLX-tagged tests.
- **Docs are CI-enforced.** Doc-guard tests fail the build if a public CLI flag or JSON config field is undocumented, or a doc index has a dead local link. Add the doc entry in the same change as the flag/field.
- Tests with optional resources (data shards, GPU) must `t.Skip` not `t.Fatal` when the resource is missing.
- Keep each `.go` file ≤ 1000 lines (pre-commit hook enforces it). Split by extracting cohesive siblings.
- Don't push CUDA-kernel changes without smoke-testing on a CUDA host — GitHub CI has no nvcc, so kernels aren't compiled there.
- Roll a release only when asked; follow `docs/releasing.md` and verify each remote step (tag, release, formula) actually landed.
