# Feature Matrix

This page is a routing map, not a replacement for validation. Use
`mixlab -mode validate -config model.json` against the installed Mixlab version
before starting a run. Exact defaults and rejected combinations live in the
[configuration reference](config-reference.md); Hugging Face details live in
the [HF support matrix](hf-export-support-matrix.md).

## Workflows

| Workflow | Native support | Main command | Notes |
|----------|----------------|--------------|-------|
| Config validation | Yes, no GPU required | `mixlab -mode validate -config model.json` | Parses defaults, validates combinations, and builds IR. |
| Parameter/compute inspection | Yes, no GPU required | `mixlab -mode count -config model.json` | Reports unique/active parameters, memory, IR ops, and estimated FLOPs. |
| Text/JSONL preparation | Yes | `mixlab -mode prepare` | Produces tokenizer, shards, and a dataset manifest. |
| Per-record text framing | Yes | `mixlab -mode prepare -frame-per-record` | One BOS/EOS/PAD-framed record per sequence row. |
| FASTA preparation | Yes | `mixlab -mode prepare -input-format fasta` | Produces record-oriented nucleotide shards and vocabulary metadata. |
| Training and races | Yes | `mixlab -mode arch` / `arch_race` | Periodic checkpoints can be resumed with `-resume`. |
| Native evaluation | Yes | `mixlab -mode eval` | Supports loss plus optional aligned logprob, rank, uncertainty, and logits exports. |
| Causal generation | Yes | `mixlab -mode generate` | Supports batching, deterministic seeds, sequence vocabularies, and grammar constraints. |
| Diffusion generation/scoring | Yes | `generate-diffusion` / `score-diffusion` | Requires a block-diffusion objective or selected diffusion head. |
| RTD scoring | Yes | `mixlab -mode score-electra` | Scores `P(original)` from a native RTD detector head. |
| Energy/PLL scoring | Yes | `mixlab -mode score-ebm` | Supports native energy, differing-span PLL, dependent-window PLL, and full-sequence PLL. |
| Hugging Face export | Gated by model surface | `export-hf`, then `parity` | Unsupported features fail explicitly; see the HF support matrix. |

## Objectives And Heads

| Feature | Training | Native eval/scoring | Native generation | HF export | Starting point |
|---------|----------|---------------------|-------------------|-----------|----------------|
| Causal LM | Yes | Next-token eval | Causal generation | Causal LM | [plain_3L.json](../examples/plain_3L.json) |
| MLM | Yes | Masked eval and full-sequence PLL | No causal semantics | Masked LM on supported trunks | [mlm_tiny.json](../examples/mlm_tiny.json) |
| MNTP | Yes | Masked eval and same-position PLL scoring | No causal semantics | Masked LM on supported trunks | [distillation_mntp_tiny.json](../examples/distillation_mntp_tiny.json) |
| Causal + masked hybrid | Yes | Concrete-objective eval; causal validation | Causal path | Causal and masked surfaces when supported | [hybrid_tiny.json](../examples/hybrid_tiny.json) |
| Block diffusion | Yes | Native PLL scoring | Native diffusion generation | Pure diffusion is native-only | [block_diffusion_tiny.json](../examples/block_diffusion_tiny.json) |
| Multihead | Yes | Selected native scorer/detector/denoiser | Selected diffusion head | Exports only `export_head` | [multihead_mntp_diffusion_tiny.json](../examples/multihead_mntp_diffusion_tiny.json) |
| RTD detector | Multihead auxiliary | `score-electra` | No | Detector and generator are skipped | [multihead_mntp_rtd_tiny.json](../examples/multihead_mntp_rtd_tiny.json) |
| Native energy head | Multihead auxiliary | `score-ebm` | No | Energy head is skipped | [multihead_mntp_energy_tiny.json](../examples/multihead_mntp_energy_tiny.json) |
| MLM/MNTP span-PLL ranking | Multihead auxiliary | `score-ebm` | No | Scorer exports normally | [multihead_mntp_span_pll_ranking_tiny.json](../examples/multihead_mntp_span_pll_ranking_tiny.json) |

## Training-Time Features

| Feature | Runtime effect | Required artifact or restriction | Guide/example |
|---------|----------------|----------------------------------|---------------|
| Fixed-teacher distillation | Adds CE + teacher KL | Teacher configs/checkpoints; supported causal or masked objectives | [Causal](../examples/distillation_tiny.json), [masked](../examples/distillation_mntp_tiny.json) |
| Data2Vec | Adds EMA hidden-target loss | Masked objective path; correctness-first CPU EMA refresh | [data2vec_hybrid_tiny.json](../examples/data2vec_hybrid_tiny.json) |
| Minimal-pair energy/PLL | Adds paired ranking rows | JSONL or compiled pair artifact | [Training guide](config-training.md#auxiliary-training-features) |
| Invariance | Adds two-view symmetric KL | Annotated pair artifact; MLM/MNTP path | [invariance_mlm_tiny.json](../examples/invariance_mlm_tiny.json) |
| PLL margin | Adds annotated-span ranking loss | Annotated pair artifact; use conservative weight defaults | [pll_margin_mlm_tiny.json](../examples/pll_margin_mlm_tiny.json) |
| Word structural objective | Shuffles local spans and reconstructs originals | MLM/MNTP vocab head | [word_structural_mlm_tiny.json](../examples/word_structural_mlm_tiny.json) |
| Whole-word masking curriculum | Changes host-side mask selection | Compatible `tokenizer.json` beside shards | [mlm_wwm_curriculum_tiny.json](../examples/mlm_wwm_curriculum_tiny.json) |
| Segment attention masking | Blocks cross-segment attention | Boundary token or manifest-backed record segments | [packed_segment_mask_tiny.json](../examples/packed_segment_mask_tiny.json) |
| Character, bigram, trigram features | Adds token-derived embedding channels | Character features require `char_features.bin` | [char_features_plain.json](../examples/char_features_plain.json) |
| SWA/EMA checkpoints | Writes live and averaged weights | Configure start, decay, and interval | [swa_ema_tiny.json](../examples/swa_ema_tiny.json) |

Training-only features do not run during native generation or ordinary HF
inference unless the exported model surface explicitly requires their weights.
The config validator is authoritative for combinations.

## Architecture Families

| Family | Native training/inference | HF status |
|--------|---------------------------|-----------|
| `plain` attention and FFN/MoE blocks | Broad support | Broad but feature-gated support |
| DeBERTa and differential `plain` variants | Native causal/masked support | Supported combinations have parity coverage |
| `ttt_mlp` | Native causal training and stateful generation | Supported only for cache-safe TTT stacks |
| HGRN2, mLSTM, RetNet, RWKV, Mamba, Gated DeltaNet | Native correctness-first paths | Generally gated or unsupported |
| Parallel groups, recurrence, U-Net, custom blocks | Native under documented restrictions | Generally unsupported |

Use [config-blocks.md](config-blocks.md) to choose a block family and the
[HF support matrix](hf-export-support-matrix.md) before assuming exportability.
