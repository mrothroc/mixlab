# mixlab docs

This directory holds focused guides for using and extending mixlab.

| Document | Purpose |
|----------|---------|
| [cli.md](cli.md) | CLI mode overview and links to focused mode references. |
| [cli-train.md](cli-train.md) | `arch` and `arch_race` training flags. |
| [cli-prepare.md](cli-prepare.md) | `prepare` data/tokenizer flags. |
| [cli-eval.md](cli-eval.md) | `count`, `eval`, diffusion scoring, hidden-state export, and per-token eval exports. |
| [cli-generate.md](cli-generate.md) | Causal and block-diffusion generation flags. |
| [grammar-constrained-generation.md](grammar-constrained-generation.md) | Token-DFA and GBNF constrained causal decoding. |
| [cli-export.md](cli-export.md) | Hugging Face `export-hf` and `parity` flags. |
| [data.md](data.md) | Preparing example data, FineWeb-Edu, custom corpora, and tokenizer compatibility. |
| [architecture.md](architecture.md) | Model/block concepts, config customization, custom blocks, and extension points. |
| [recipes.md](recipes.md) | Registry of reproducible end-to-end recipes (config → trained → published HF model). |
| [config-model.md](config-model.md) | Short guide to top-level model fields and embedding channels. |
| [config-blocks.md](config-blocks.md) | Short guide to block families and common block patterns. |
| [config-training.md](config-training.md) | Short guide to training objectives, optimizers, and auxiliary losses. |
| [config-advanced.md](config-advanced.md) | Routing map for advanced graph, export, and runtime features. |
| [config-reference.md](config-reference.md) | Exhaustive JSON configuration reference with stable anchors. |
| [hf-export.md](hf-export.md) | Hugging Face export workflow and compatibility notes. |
| [hf-export-support-matrix.md](hf-export-support-matrix.md) | Supported HF export features by model surface. |
| [performance.md](performance.md) | MLX tuning, memory limits, step timing, and profiling. |
| [canonical_mamba3.md](canonical_mamba3.md) | Canonical Mamba-3 implementation notes. |
| [ttt-mlp-stateful-inference.md](ttt-mlp-stateful-inference.md) | Native TTT-MLP prefill/decode state, ownership, and Apple runtime measurements. |
| [diffusion-experiments.md](diffusion-experiments.md) | Workflow for causal, MLM, block-diffusion, hybrid-diffusion, and sampler sweeps. |
| [block-diffusion-design.md](block-diffusion-design.md) | Technical design for the block-diffusion walking skeleton. |
| [block-diffusion-release-verification.md](block-diffusion-release-verification.md) | Release verification notes for the block-diffusion walking skeleton. |
| [reference-parity-audit-2026-06.md](reference-parity-audit-2026-06.md) | Reference parity audit notes. |
| [sequence-modality-release-train.md](sequence-modality-release-train.md) | Review-gated roadmap from discrete sequences through encoder-free media inputs and codec-token generation. |
| [sequence-modality-r1-verification.md](sequence-modality-r1-verification.md) | Release 1 compatibility evidence and review gate. |
| [sequence-modality-r2-verification.md](sequence-modality-r2-verification.md) | Release 2 nucleotide vertical-slice evidence and review gate. |
| [releasing.md](releasing.md) | Release process notes. |

Start with the root [README](../README.md) for installation and a quickstart,
then use [cli.md](cli.md), [data.md](data.md), [config-model.md](config-model.md),
and [architecture.md](architecture.md) for day-to-day workflows.
