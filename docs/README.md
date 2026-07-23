# mixlab docs

Use this page to choose the shortest path for the task at hand. The root
[README](../README.md) remains the installation and first-run guide.

## Start By Task

| Task | Start here | Then use |
|------|------------|----------|
| Prepare text, JSONL, records, or FASTA | [Data preparation](data.md) | [Prepare CLI](cli-prepare.md) |
| Choose an architecture or objective | [Architecture guide](architecture.md) | [Model](config-model.md), [blocks](config-blocks.md), and [training](config-training.md) guides |
| Check a config without a GPU | `mixlab -mode validate -config model.json` | [Configuration reference](config-reference.md) |
| Count parameters and estimated compute | `mixlab -mode count -config model.json` | [Count and eval CLI](cli-eval.md) |
| Train, compare, or resume runs | [Training CLI](cli-train.md) | [Performance](performance.md) |
| Evaluate or score checkpoints | [Count and eval CLI](cli-eval.md) | [Feature matrix](feature-matrix.md) |
| Generate causal or diffusion samples | [Generation CLI](cli-generate.md) | [Grammar constraints](grammar-constrained-generation.md) |
| Export to Hugging Face | [HF export workflow](hf-export.md) | [HF support matrix](hf-export-support-matrix.md) |
| Reproduce a complete workflow | [Recipes](recipes.md) | [Example configs](../examples/README.md) |
| Add blocks or custom IR | [Architecture guide](architecture.md) | [Advanced config](config-advanced.md) and [custom ops](config-reference.md#custom-blocks) |

## Canonical References

These documents define the public user-facing contract:

| Document | Scope |
|----------|-------|
| [CLI index](cli.md) | Modes, common conventions, and links to mode-specific flags. |
| [Configuration reference](config-reference.md) | Public JSON fields, defaults, validation rules, and custom ops. |
| [Feature matrix](feature-matrix.md) | Lifecycle support across training, native runtime, and HF export. |
| [HF support matrix](hf-export-support-matrix.md) | Detailed Hugging Face export compatibility. |
| [Examples index](../examples/README.md) | Maintained starting configs and selection guidance. |
| [LLM index](../llms.txt) | Compact retrieval map and config-generation rules for automated clients. |

Run mode-specific help for the installed version:

```bash
mixlab -mode MODE -h
```

## Specialized Guides

- [Canonical Mamba-3](canonical_mamba3.md)
- [TTT-MLP stateful inference](ttt-mlp-stateful-inference.md)
- [Block-diffusion experiments](diffusion-experiments.md)
- [Grammar-constrained generation](grammar-constrained-generation.md)
- [Performance, memory, and profiling](performance.md)

## Maintainer And Design Records

These explain implementation decisions and release evidence; they are not the
primary user contract:

- [Block-diffusion design](block-diffusion-design.md)
- [Block-diffusion release verification](block-diffusion-release-verification.md)
- [Reference parity audit](reference-parity-audit-2026-06.md)
- [Release process](releasing.md)

Public config fields and CLI flags are checked against the documentation by the
Go test suite. Prefer updating the canonical references instead of adding a
second description with different defaults.
