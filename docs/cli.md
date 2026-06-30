# CLI Usage

mixlab uses `-mode` to choose a workflow:

```bash
mixlab -mode MODE [flags]
```

Run `mixlab -h` for the mode list, or `mixlab -mode MODE -h` for grouped
mode-specific flags.

| Mode | Description | Details |
|------|-------------|---------|
| `arch` | Train a single architecture from a JSON config. The default mode. | [cli-train.md](cli-train.md) |
| `arch_race` | Train every JSON config in a directory and compare results. | [cli-train.md](cli-train.md) |
| `smoke` | Run diagnostic checks for MLX availability and GPU health. | This page |
| `prepare` | Tokenize raw text or JSONL into binary training shards. | [cli-prepare.md](cli-prepare.md) |
| `count` | Print parameter, size, block, FLOP, and IR op counts for a config. | [cli-eval.md](cli-eval.md) |
| `eval` | Load safetensors and evaluate validation loss or per-token exports. | [cli-eval.md](cli-eval.md) |
| `hiddenstats` | Export one batch of hidden states as float32 binary. | [cli-eval.md](cli-eval.md) |
| `generate` | Generate token IDs from a causal checkpoint. | [cli-generate.md](cli-generate.md) |
| `generate-diffusion` | Generate token IDs from a block-diffusion checkpoint. | [cli-generate.md](cli-generate.md) |
| `score-diffusion` | Score token-id sequences with native block-diffusion PLL. | [cli-eval.md](cli-eval.md) |
| `export-hf` | Export supported safetensors checkpoints to Hugging Face directories. | [cli-export.md](cli-export.md) |
| `parity` | Compare native Mixlab inference against a Hugging Face export. | [cli-export.md](cli-export.md) |

## Common Conventions

- `-config` points to the JSON architecture config for single-config modes.
- `-train` is a shard glob. It is used for training, eval, parity sampling,
  and hidden-state export.
- `-safetensors-load` loads a checkpoint before eval, generation, export, or
  resumed training.
- `-safetensors` writes a final checkpoint after training.
- `-output` is still supported for older scripts. New scripts should prefer
  mode-specific aliases where available: `-prepare-output-dir`, `-export-dir`,
  and `-hiddenstats-out`.
- The training flag `-eval` remains supported. New scripts can use
  `-eval-after-train` to avoid confusion with `-mode eval`.

## Smoke

Check GPU/backend availability:

```bash
mixlab -mode smoke
```

`smoke` has no required flags. It reports whether the MLX backend is available
and runs lightweight diagnostics.
