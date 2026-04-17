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
