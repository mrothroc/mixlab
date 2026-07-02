# Config: Training

Training settings live under the top-level `training` object. This page
summarizes the major groups; use [config-reference.md](config-reference.md) for
exact defaults and validation rules.

## Basic Shape

```json
{
  "training": {
    "steps": 1000,
    "batch_tokens": 1024,
    "lr": 0.0003,
    "seed": 1,
    "objective": "causal"
  }
}
```

## Objectives

| `training.objective` | Use case |
|------|---------|
| `"causal"` | Standard next-token LM training. Default. |
| `"mlm"` | Masked language modeling on selected input positions. |
| `"mntp"` | Masked next-token prediction without target leakage. |
| `"hybrid"` | Per-batch mix of causal and MLM/MNTP secondary objective. |
| `"block_diffusion"` | Block-wise masked diffusion training. |
| `"multihead"` | Shared-trunk training with named scorer, denoiser, or auxiliary heads. |

Masked objectives use MLM fields such as `mlm_mask_prob`,
`mlm_mask_token_id`, replacement probabilities, and hybrid fields such as
`hybrid_clm_fraction` and `hybrid_secondary_objective`.

Multihead configs use `training.heads`, `export_head`, optional
`diffusion_head`, and optional auxiliary settings such as `training.rtd` and
`training.minimal_pair`. They are intended for recipes such as an MNTP/BERT-MLM
scorer head plus a native block-diffusion denoiser, ELECTRA-style RTD detector,
or native energy/ranking head over the same trunk.

## Optimizers

| `training.optimizer` | Behavior |
|------|----------|
| omitted / `"muon"` | Muon-style matrix optimization plus AdamW for other groups. |
| `"adamw"` | AdamW for all supported groups. |
| `"lamb"` | LAMB for all parameter classes. |

Optimizer-group overrides such as embedding, matrix, scalar, and head learning
rates/weight decay are documented in the full reference.

## Auxiliary Training Features

| Field | Purpose |
|------|---------|
| `distillation` | In-training fixed-teacher ensemble distillation for causal LM. |
| `data2vec` | EMA self-distillation for masked objective paths. |
| `rtd` | ELECTRA-style replaced-token detection auxiliary for multihead training. |
| `minimal_pair` | Clean/corrupt pair data for native multihead energy ranking. |
| `mtp` | Parameter-free multi-token prediction auxiliary loss. |
| `first_byte_mask` | First-byte masked loss path. |
| `attention_segment_mask` | Block-diagonal segment attention for packed training sequences. |
| `swa_start`, `swa_decay`, `swa_interval` | SWA/EMA averaged checkpoint accumulation. |

Compatibility is intentionally explicit. Some features are training-only and
are rejected by export paths until they have defined inference semantics.
Minimal-pair JSONL can be validated or compiled to `.mpair` shards with
`mixlab -mode prepare-pairs`; set `training.minimal_pair.source: "bin"` to use
the compiled artifact.

## Validation And Logging

Use `val_every`, early-stop fields, and target validation loss settings in the
config for repeatable runs. CLI flags such as `-log-every`, `-val-every`, and
`-eval-after-train` are run-level overrides; see [cli-train.md](cli-train.md).
