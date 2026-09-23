# CLI Usage

mixlab uses `-mode` to choose a workflow:

```bash
mixlab -mode MODE [flags]
```

Run `mixlab -h` for the mode list, or `mixlab -mode MODE -h` for grouped
mode-specific flags.

| Mode | Description | Details |
|------|-------------|---------|
| `validate` | Parse, validate, and build the IR for one config without initializing MLX. | This page |
| `arch` | Train a single architecture from a JSON config. The default mode. | [cli-train.md](cli-train.md) |
| `arch_race` | Train every JSON config in a directory and compare results. | [cli-train.md](cli-train.md) |
| `smoke` | Run diagnostic checks for MLX availability and GPU health. | This page |
| `prepare` | Tokenize raw text or JSONL into binary training shards. | [cli-prepare.md](cli-prepare.md) |
| `prepare-pairs` | Validate minimal-pair, invariance-pair, or annotated PLL-margin JSONL and optionally compile it to a compact pair shard. | [cli-prepare.md](cli-prepare.md) |
| `count` | Print parameter, size, block, FLOP, and IR op counts for a config. | [cli-eval.md](cli-eval.md) |
| `optimizer-report` | Write resolved optimizer groups and per-tensor assignments as JSON without a GPU. | This page |
| `eval` | Load safetensors and evaluate validation loss or per-token exports. | [cli-eval.md](cli-eval.md) |
| `hiddenstats` | Export one batch of hidden states as float32 binary. | [cli-eval.md](cli-eval.md) |
| `generate` | Generate token IDs from a causal checkpoint. | [cli-generate.md](cli-generate.md) |
| `generate-diffusion` | Generate token IDs from a block-diffusion checkpoint. | [cli-generate.md](cli-generate.md) |
| `score-diffusion` | Score token-id sequences with native block-diffusion PLL. | [cli-eval.md](cli-eval.md) |
| `score-electra` | Score token-id sequences with a native RTD detector head. | [cli-eval.md](cli-eval.md) |
| `score-ebm` | Score token-id sequences or pairs with native energy or scorer span-PLL ranking. | [cli-eval.md](cli-eval.md) |
| `export-hf` | Export supported safetensors checkpoints to Hugging Face directories. | [cli-export.md](cli-export.md) |
| `parity` | Compare native Mixlab inference against a Hugging Face export. | [cli-export.md](cli-export.md) |

## Common Conventions

- `-config` points to the JSON architecture config for single-config modes.
- `-train` is a shard glob. It is used for training, eval, parity sampling,
  and hidden-state export.
- `-safetensors-load` loads model weights before eval, generation, export, or a
  weights-only warm start. It does not restore optimizer or schedule state.
- `-resume` restores a complete periodic training bundle in `arch` mode. See
  [CLI: Training](cli-train.md#resume-and-extension).
- `-safetensors` writes a final checkpoint after training.
- `-output` is still supported for older scripts. New scripts should prefer
  mode-specific aliases where available: `-prepare-output-dir`, `-export-dir`,
  and `-hiddenstats-out`.
- The training flag `-eval` remains supported. New scripts can use
  `-eval-after-train` to avoid confusion with `-mode eval`.

## Version

`-version` prints the build identity and exits. It takes no `-mode` and runs
before any GPU probing, so it still answers on a machine where MLX is broken:

```bash
$ mixlab -version
mixlab v0.115.1 (e00e54f2efe2, 2026-09-18T13:28:21Z)
```

The version, commit, and build time come from the information the Go linker
stamps into the binary, so they describe the binary in hand rather than
whatever a package manager last recorded. A binary built from a modified tree
marks its commit `-dirty`, and a build without VCS stamping prints the module
version alone. An untagged build reports `(devel)` as its version.

This is the identity to quote in a bug report. Container images carry the same
value in their `org.opencontainers.image.version` label, but that label
describes the image while `-version` describes the executable inside it.

## Validate

Validate a config before allocating a GPU or loading training data:

```bash
mixlab -mode validate -config examples/plain_3L.json
```

This applies defaults, rejects unknown or incompatible fields, and builds the
native IR. A successful command exits zero and prints one summary line. Use
`count` when you also want parameter, memory, op, and FLOP estimates.

## Optimizer Report

Inspect the optimizer coverage before starting a run:

```bash
mixlab -mode optimizer-report -config examples/vit_pytorch_init.json > optimizer-groups.json
```

Only `-config` is required. JSON goes to stdout; warnings go to stderr. This
mode needs neither MLX nor training shards, allocates no weight arrays, and
does not initialize the GPU. It resolves config defaults through the same
optimizer builder used in training, without loading checkpoint state or applying
runtime-only optimizer overrides.

The `mixlab_optimizer_report_v1` document contains:

- `groups`: the four standard classes (including empty classes), followed by
  any specialized groups in resolution order. Each active group includes its
  optimizer-spec `index`, optimizer variant, configured LR/decay, tensor and
  parameter counts, and decay-eligible tensor count. Unused classes have no
  optimizer or rate fields.
- `tensors`: one row per unique stored weight or buffer, in checkpoint order.
  `index` disambiguates repeated names such as `wq`; `name`, `shape`, and
  `parameters` identify the tensor. Active rows include `group_index`, `group`,
  `optimizer`, `configured_lr`, `configured_weight_decay`, `decay_eligible`,
  and `effective_weight_decay`. Frozen/buffer rows have no optimizer assignment.
- `frozen_tensors` and `buffer_tensors`: disjoint inactive counts. Buffer rows
  also have `frozen:true` because optimizers never update them.
- `rate_basis: "configured_before_schedule"`: these are base group rates,
  not instantaneous rates after warmup, phases, schedule changes, or resume.
  Effective decay is the configured coefficient after tensor exclusions, not
  the realized update magnitude or a cautious-decay activation indicator.

Training also prints an `optimizer (configured rates, before schedule)` summary
at trainer creation. It reflects the final runtime spec, including optimizer
overrides; custom unnamed groups are labeled `group_<index>`. Counts do not
multiply shared weights by their number of uses. Zero LR is reported as zero,
not as frozen: optimizer state can still accumulate with zero LR.

See [Optimizer groups](config-reference.md#optimizer-groups) for current routing
rules and specialized SSM groups. This reporting feature does not change them.

## Smoke

Check GPU/backend availability:

```bash
mixlab -mode smoke
```

`smoke` has no required flags. It reports whether the MLX backend is available
and runs lightweight diagnostics.
