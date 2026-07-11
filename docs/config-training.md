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

LAMB caps its per-tensor trust ratio at `lamb_trust_ratio_cap` (`10.0` by
default) to avoid post-warmup amplification when an update norm becomes very
small. Set `lamb_trust_ratio_cap: 0` only when intentionally testing uncapped
LAMB behavior.

### Choosing an optimizer for architecture comparison

Muon only orthogonalizes 2D weight matrices; biases, norms, gains, embeddings, and
the output head stay on AdamW (hence the separate `matrix_lr` vs
`embed_lr`/`head_lr`/`scalar_lr` groups). Its benefit is therefore uneven across
architecture families, which matters when the optimizer is a variable in an
experiment rather than a fixed choice:

- **Within one family** (transformer vs transformer): prefer Muon, tuned once on a
  reference config and applied identically. It usually gives the best loss per unit
  of compute, so architecture differences surface sooner and the result transfers
  to how a real pretraining run would be optimized.
- **Across dissimilar families** (attention vs SSM/hybrid blocks such as
  `mamba3-canonical`, `gated_deltanet`, `hgrn2`, or a `parallel_group`): hold the
  optimizer fixed and use **AdamW as the neutral baseline**. Much of an SSM/recurrent
  block's parameters are gates and small projections that Muon treats differently, so
  Muon can favor one family for optimizer-interaction reasons rather than intrinsic
  architecture quality. Re-run the top one or two candidates under Muon afterward to
  confirm the ranking holds and to get the better absolute number.

Whichever you choose, apply the same optimizer and tuning protocol to every
candidate. An untuned optimizer — Muon especially, since its learning-rate scale
differs from AdamW's — will change rankings faster than the architecture differences
you are trying to measure.

## Auxiliary Training Features

| Field | Purpose |
|------|---------|
| `distillation` | In-training fixed-teacher ensemble distillation for causal and masked vocab objectives. |
| `data2vec` | EMA self-distillation for masked objective paths. |
| `rtd` | ELECTRA-style replaced-token detection auxiliary for multihead training. |
| `minimal_pair` | Clean/corrupt pair data for native energy ranking or MLM/MNTP span-PLL scorer regularization. |
| `invariance` | Structured two-view symmetric-KL consistency loss for masked vocab predictions. |
| `pll_margin` | Directional paired annotated-span PLL margin auxiliary for masked vocab predictions. |
| `word_structural_objective` | StructBERT-style local shuffle reconstruction for MLM/MNTP vocab heads. |
| `mtp` | Parameter-free multi-token prediction auxiliary loss. |
| `first_byte_mask` | First-byte masked loss path. |
| `attention_segment_mask` | Block-diagonal segment attention for packed training sequences. |
| `swa_start`, `swa_decay`, `swa_interval` | SWA/EMA averaged checkpoint accumulation. |

Compatibility is intentionally explicit. Some features are training-only and
are rejected by export paths until they have defined inference semantics.
`training.distillation` is training-only and can target causal, MLM, MNTP, and
compatible hybrid masked steps. Masked distillation runs teachers on the same
masked inputs as the student and reduces KL only over masked loss rows; set
`loss_weight_kl: 0` and `loss_weight_ce: 1` for a graph-identical no-KD
control.
Minimal-pair JSONL can be validated or compiled to `.mpair` shards with
`mixlab -mode prepare-pairs`; set `training.minimal_pair.source: "bin"` to use
the compiled artifact. The default `score_source: "energy_scalar"` trains a
native energy head. Set `score_source: "mlm_span_pll"` with
`energy_aggregation: "differing_span"` to add an exportable MLM/MNTP scorer
regularizer that ranks clean/corrupt pairs by masked-span pseudo-log-likelihood.

`word_structural_objective` is training-only and does not add weights or change
HF export/scoring. Omit it or set `enabled:false` for exact disabled parity.
`loss_weight:0` keeps the input shuffle active and only removes the auxiliary
loss contribution, which is useful as a corruption-only ablation.

`training.invariance` is a training-only, corpus-owned consistency objective for
MLM/MNTP models and multihead configs whose export head is MLM/MNTP. It samples
explicitly annotated view pairs, masks one annotated position in each view, and
adds `weight * 0.5 * (KL(P_a || P_b) + KL(P_b || P_a))` to the normal task loss.
The pair records must keep the annotated target token unchanged across both
views; Mixlab does not infer spans or learn a nuisance detector. For numerical
stability, probability tails below `1e-8` are floored and renormalized before
the divergence, bounding only collapsed-tail contributions. Use
`source: "file"` (the default) to auto-detect JSONL or a compiled pair binary,
`loss: "sym_kl"`, and `target: "masked_position"`. `batch_fraction` defaults
to `0.25`. `skip_token_ids` optionally excludes tokenizer special IDs from
annotated targets; `mlm_mask_token_id` is always excluded. Set `weight: 0` for an exact no-op: Mixlab loads no pair artifact,
declares no extra graph input, and produces the baseline graph and loss path.

```jsonc
"training": {
  "objective": "mlm",
  "mlm_mask_token_id": 103,
  "invariance": {
    "source": "file",
    "path": "data/invariance_pairs.bin",
    "loss": "sym_kl",
    "weight": 0.1,
    "batch_fraction": 0.25,
    "target": "masked_position",
    "skip_token_ids": [0, 1, 2]
  }
}
```

`training.pll_margin` is a training-only paired ranking objective for MLM,
MNTP, or a multihead masked export head. Each pair supplies preferred and
contrast views plus an unchanged annotated target span. Mixlab masks that span
in both views, computes stable span PLL values with `log_softmax`, and adds
`weight * (softplus(margin - (pll_pos - pll_neg)) + anchor_weight * -pll_pos)`.
The ordinary MLM/MNTP loss mask is cleared on selected pair rows: the contrast
view is input context only, while the explicit positive-view anchor prevents a
trivial decrease of both scores. `weight: 0` is an exact no-op with no pair-file
load, graph input, or additional RNG use. It is intentionally incompatible in
v1 with other batch-mutating pair auxiliaries.

```jsonc
"training": {
  "objective": "mntp",
  "mlm_mask_token_id": 103,
  "pll_margin": {
    "source": "file",
    "path": "data/pll_margin_pairs.bin",
    "margin": 1.0,
    "weight": 1.0,
    "anchor_weight": 0.5,
    "batch_fraction": 0.25,
    "target": "annotated_span",
    "skip_token_ids": [0, 1, 2]
  }
}
```

## Validation And Logging

Use `val_every`, early-stop fields, and target validation loss settings in the
config for repeatable runs. CLI flags such as `-log-every`, `-val-every`, and
`-eval-after-train` are run-level overrides; see [cli-train.md](cli-train.md).
