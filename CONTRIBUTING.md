# Contributing to mixlab

## Reporting bugs

Open a GitHub issue with:
- The command you ran
- What you expected
- What actually happened
- Your platform (macOS/Linux, GPU type)

## Submitting PRs

1. Fork the repo
2. Create a feature branch
3. Run `make setup` to enable pre-commit hooks
4. Make your changes
5. Run `make test` and `make lint`
6. Submit a PR

## Code style

- `gofmt` for formatting
- `golangci-lint` for linting (config in `.golangci.yml`)
- All tests must pass: `make test`
- No Go file over 1000 lines

## Adding new block types

Prefer defining new architectures as custom JSON blocks first:
```json
{"type": "custom", "name": "my_block", "weights": [...], "ops": [...]}
```

If a custom block proves useful, propose a built-in block type via PR.
Built-in blocks use the registry API:
```go
arch.RegisterBlock("my_block", arch.BlockRegistration{...})
```

## Testing

```bash
make test      # all tests
make lint      # golangci-lint + file size check
make build     # build binary
```

`make test` and CI both build with `CGO_ENABLED=0`, so neither links MLX. Anything
touching `gpu/` needs the MLX-tagged suite on a machine with Metal or CUDA:

```bash
CGO_ENABLED=1 go test -tags mlx ./arch/... ./gpu ./train -count=1
```

## Local toolchain drift

Two dependencies are installed outside the repo and have each silently broken a
check after a routine `brew upgrade`. If something fails in a way that looks
unrelated to your change, suspect these before the code.

**MLX.** `Formula/mixlab.rb` asserts a tested range at install time, but that only
guards installs — upgrading MLX under an already-installed mixlab swaps the library
without rebuilding anything. MLX has changed autodiff behavior in a patch release
(0.32.1 broke MoE and bf16 training), so run the MLX-tagged suite after any MLX
upgrade. Homebrew keeps old versions, so you can A/B a suspected regression:

```bash
MLX=/opt/homebrew/Cellar/mlx/<version>
CGO_ENABLED=1 CGO_CFLAGS="-I$MLX/include" CGO_CXXFLAGS="-I$MLX/include -std=c++20" \
  CGO_LDFLAGS="-L$MLX/lib -Wl,-rpath,$MLX/lib" go test -tags mlx -c -o /tmp/x.test ./gpu
DYLD_LIBRARY_PATH="$MLX/lib" /tmp/x.test
```

Build *and* load the same version: the dylib's install name resolves through
`opt/mlx`, so `-rpath` alone does not redirect it, and a mixed build yields
`Symbol not found` rather than a clean result.

**Go and golangci-lint.** golangci-lint typechecks against the standard library of
whichever Go it runs under, and a newer Go can emit export data it cannot decode —
producing phantom `typecheck` errors on every import that drown out real findings.
The pre-commit hook pins lint to CI's Go version for this reason (override with
`LINT_GOTOOLCHAIN`); `make lint` does not, so prefer the hook, or pin explicitly:

```bash
GOTOOLCHAIN=go1.24.0 golangci-lint run ./...
```

A findings count that changes without a code change is the tell. Baseline against
`main` before assuming a finding is yours, and note that `max-same-issues` caps the
report — use `--max-same-issues=0 --max-issues-per-linter=0` to see the true count.
