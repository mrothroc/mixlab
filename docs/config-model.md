# Config: Model Basics

This page is the short path through the top-level model fields. Use
[config-reference.md](config-reference.md) for exhaustive field-by-field details.

## Minimal Shape

```json
{
  "name": "plain-small",
  "model_dim": 128,
  "vocab_size": 1024,
  "seq_len": 128,
  "blocks": [
    {"type": "plain", "heads": 4},
    {"type": "swiglu"}
  ],
  "training": {
    "steps": 200,
    "lr": 0.0003,
    "batch_tokens": 1024
  }
}
```

## Core Fields

| Field | Purpose |
|------|---------|
| `model_dim` | Hidden size used by blocks and embeddings. |
| `vocab_size` | Token vocabulary size. Mixlab binary shards store token ids as `uint16`. |
| `seq_len` | Context length in tokens. |
| `mlp_mult` | FFN expansion multiplier used by FFN-style blocks and experts. |
| `blocks` | Ordered architecture stack. See [config-blocks.md](config-blocks.md). |
| `training` | Training objective, optimizer, schedule, and runtime settings. See [config-training.md](config-training.md). |

## Embedding Channels

The base token embedding is always present. Optional channels add into the
initial hidden state before block 0:

| Field | Channel |
|------|---------|
| `positional_embedding` | `"rope"` default rotary attention, `"learned_absolute"` GPT-2-style WPE, or `"none"`. |
| `char_vocab_size`, `char_dim`, `char_max_per_token` | Token-level fixed-slot byte/char features from `char_features.bin`. |
| `bigram_vocab_size`, `bigram_dim` | Hashed bigram embedding channel. |
| `trigram_vocab_size`, `trigram_dim` | Hashed trigram embedding channel. |
| `smear_embeddings` | Optional previous-token embedding smear before block execution. |

## Norm And Residual Defaults

Mixlab's default GPT-style stack is pre-norm RMSNorm. Use these top-level
fields to opt into other layouts:

| Field | Purpose |
|------|---------|
| `norm_type` | `"rmsnorm"` or `"layernorm"`. |
| `norm_affine` | Whether LayerNorm has learned scale/bias. |
| `norm_placement` | `"pre"`, `"post"`, or `"sandwich"` for supported sequential blocks. |
| `ffn_internal_norm` | Adds internal FFN normalization on supported FFN paths. |
| `block_scales` | Adds learned residual-branch scales. |
| `resid_mix` | Mixes current state with original embeddings on `plain` blocks. |
| `parallel_residual` | Runs supported attention/mixer plus FFN pairs in parallel. |

## Export Surface

Use `hf_export_format: "mixlab"` for the default custom Hugging Face export.
Use `hf_export_format: "gpt2"` only for strict GPT-2-compatible sequential
`plain` stacks. See [hf-export.md](hf-export.md) and
[hf-export-support-matrix.md](hf-export-support-matrix.md).
