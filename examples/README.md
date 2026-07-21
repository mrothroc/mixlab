# mixlab Example Configs

Example architecture configs from simplest to most advanced. All configs
support JSONC comments (`//`) for inline documentation.

## Quick start

```bash
# Build mixlab with MLX backend
make build

# Run any config
./mixlab -mode arch -config examples/<config>.json \
  -train "data/example/train_*.bin"
```

## Hugging Face export

After training a supported causal core model, export a Hugging Face custom-code directory with:

```bash
./mixlab -mode export-hf \
  -config examples/plain_3L.json \
  -safetensors-load runs/plain_3L/weights.safetensors \
  -export-dir runs/plain_3L/hf \
  -tokenizer-path data/example/tokenizer.json
```

See [../docs/hf-export.md](../docs/hf-export.md) and the [HF export support matrix](../docs/hf-export-support-matrix.md) for supported blocks, gated advanced features, tokenizer artifacts, parity-test setup, and the `trust_remote_code=True` security boundary.

## Grammar-constrained generation

The [`grammars/`](grammars/) directory contains a versioned token-DFA fixture
and a small GBNF fixture for causal constrained decoding. Grammar artifacts are
kept below the top-level example-config directory so config discovery does not
treat them as architectures. See
[Grammar-constrained generation](../docs/grammar-constrained-generation.md).

## SWA/EMA Averaged Weights

Configs can enable averaged weight tracking with `training.swa_start`, `training.swa_decay`, and `training.swa_interval`, or you can override those fields at the CLI:

```bash
./mixlab -mode arch \
  -config examples/swa_ema_tiny.json \
  -train "data/example/train_*.bin" \
  -safetensors runs/swa_ema_tiny/model.safetensors \
  -swa-start 100 \
  -swa-decay 0.99 \
  -swa-interval 5
```

When SWA/EMA weights are populated, Mixlab writes the live final weights to `model.final.safetensors` and the averaged weights to `model.swa.safetensors`. Use the `.swa.safetensors` file with `-mode export-hf -safetensors-load` when you want to publish or evaluate the averaged checkpoint.

## Configs at a glance

| Config | Architecture | Key features |
|--------|-------------|--------------|
| [plain_3L.json](plain_3L.json) | 3-layer transformer | Simplest config — attention + SwiGLU |
| [gqa_8h4kv.json](gqa_8h4kv.json) | GQA transformer | 8 query heads, 4 KV heads |
| [token_blend_plain.json](token_blend_plain.json) | Token blending | Learned adjacent-token gate |
| [bigram_plain.json](bigram_plain.json) | Bigram embedding | Hashed bigram context features |
| [char_features_plain.json](char_features_plain.json) | Char feature embedding | Tokenizer-level ByteLevel char feature channel |
| [packed_segment_mask_tiny.json](packed_segment_mask_tiny.json) | Packed segment masking | Block-diagonal plain attention from a boundary token |
| [nucleotide_dna_causal_tiny.json](nucleotide_dna_causal_tiny.json) | Base-level DNA causal model | Manifest-backed contig packing and deterministic reverse complements |
| [nucleotide_dna_mlm_tiny.json](nucleotide_dna_mlm_tiny.json) | Base-level DNA MLM | Bidirectional base reconstruction with contig isolation |
| [softcap_plain.json](softcap_plain.json) | Logit softcap | Bounded logits before loss |
| [qk_norm_tiny.json](qk_norm_tiny.json) | QK-norm attention | Learned per-head-dimension Q/K RMSNorm |
| [differential_attention_tiny.json](differential_attention_tiny.json) | Differential attention | DIFF Transformer two-softmax attention variant |
| [layernorm_sandwich_tiny.json](layernorm_sandwich_tiny.json) | LayerNorm sandwich stack | HF-exportable GPT-BERT-style norms and GEGLU |
| [gpt2_strict_small_2026.json](gpt2_strict_small_2026.json) | GPT-2 strict-small baseline | Learned absolute positions, GPT-2 FFN/norm/bias layout, native GPT-2 HF export |
| [mlm_tiny.json](mlm_tiny.json) | Bidirectional transformer | Masked language modeling objective |
| [mlm_wwm_curriculum_tiny.json](mlm_wwm_curriculum_tiny.json) | MLM masking curriculum | Whole-word masking followed by token masking |
| [word_structural_mlm_tiny.json](word_structural_mlm_tiny.json) | MLM + word-structural auxiliary | Local shuffle reconstruction on unmasked spans |
| [hybrid_tiny.json](hybrid_tiny.json) | Hybrid transformer | Causal plus masked-objective training, with per-batch default mixing and optional per-example mixing |
| [block_diffusion_tiny.json](block_diffusion_tiny.json) | Block diffusion | Block-wise masked-diffusion objective (`training.objective: "block_diffusion"`); train, then sample with `-mode generate-diffusion` |
| [hybrid_block_diffusion_tiny.json](hybrid_block_diffusion_tiny.json) | Hybrid block diffusion | Batch-level causal plus block-diffusion schedule |
| [multihead_mntp_diffusion_tiny.json](multihead_mntp_diffusion_tiny.json) | Multihead scorer + denoiser | Shared trunk with an MNTP/BERT-MLM scorer head and native block-diffusion denoiser head |
| [multihead_mntp_rtd_tiny.json](multihead_mntp_rtd_tiny.json) | Multihead scorer + RTD | Shared trunk with an MNTP/BERT-MLM scorer head and ELECTRA-style replaced-token detector |
| [multihead_mntp_rtd_dedicated_tiny.json](multihead_mntp_rtd_dedicated_tiny.json) | Multihead scorer + dedicated RTD | ELECTRA-style detector with a separate small MLM generator |
| [multihead_mntp_energy_tiny.json](multihead_mntp_energy_tiny.json) | Multihead scorer + energy ranking | Shared trunk with an MNTP/BERT-MLM scorer head and native minimal-pair energy head |
| [multihead_mntp_span_pll_ranking_tiny.json](multihead_mntp_span_pll_ranking_tiny.json) | Multihead scorer + span-PLL ranking | Exportable MNTP/BERT-MLM scorer regularized by minimal-pair span PLL |
| [invariance_mlm_tiny.json](invariance_mlm_tiny.json) | Structured masked-LM invariance | MLM with explicit two-view symmetric-KL consistency pairs |
| [pll_margin_mlm_tiny.json](pll_margin_mlm_tiny.json) | Directional masked-span PLL margin | MLM with explicit preferred/contrast annotated target-span pairs |
| [block_diffusion_sampler_sweep.json](block_diffusion_sampler_sweep.json) | Block diffusion sampler variant | Same tiny block-diffusion backbone with more aggressive sampler defaults |
| [distillation_tiny.json](distillation_tiny.json) | Teacher distillation | Causal LM with internal teacher ensemble loss |
| [distillation_mntp_tiny.json](distillation_mntp_tiny.json) | Masked teacher distillation | MNTP/BERT-MLM scorer distilled from masked teacher checkpoints |
| [data2vec_hybrid_tiny.json](data2vec_hybrid_tiny.json) | EMA representation distillation | Hybrid masked training with data2vec auxiliary loss |
| [swa_ema_tiny.json](swa_ema_tiny.json) | SWA/EMA averaging | Final and averaged checkpoint artifacts |
| [deberta_relative_tiny.json](deberta_relative_tiny.json) | Relative attention transformer | DeBERTa P2C/C2P relative position bias |
| [parallel_hybrid_branches_tiny.json](parallel_hybrid_branches_tiny.json) | Parallel attention + HGRN2 | Heterogeneous parallel group with a zero-initialized recurrent branch |
| [lamb_plain_tiny.json](lamb_plain_tiny.json) | LAMB optimizer | Whole-model LAMB optimizer on a tiny transformer |
| [moe_tiny.json](moe_tiny.json) | Sparse MoE transformer | Top-k routed SwiGLU feed-forward experts |
| [mamba_2L.json](mamba_2L.json) | Mamba SSM | Gated recurrence, no attention |
| [retnet_2L.json](retnet_2L.json) | RetNet | Exponential decay retention |
| [rwkv_2L.json](rwkv_2L.json) | RWKV | Linear attention with time decay |
| [hgrn2_2L.json](hgrn2_2L.json) | HGRN2 mixer | Matrix-state recurrence token mixer |
| [ttt_mlp_tiny.json](ttt_mlp_tiny.json) | TTT-MLP mixer | Native nonlinear per-sequence model-state updates |
| [mlstm_2L.json](mlstm_2L.json) | mLSTM mixer | Stabilized matrix-memory token mixer |
| [perceiver_2L.json](perceiver_2L.json) | Perceiver | Latent bottleneck cross-attention |
| [custom_geglu.json](custom_geglu.json) | Custom block | Gated feed-forward block defined in pure JSON |
| [unet_transformer.json](unet_transformer.json) | U-Net transformer | Skip connections, block scales, residual mixing |
| [recurrent_parallel.json](recurrent_parallel.json) | Recurrent parallel | Depth recurrence, parallel residuals, TTT |

## Which config should I use?

- **Learning mixlab**: Start with `plain_3L.json` — it trains in seconds.
- **Masked objectives**: Use `mlm_tiny.json` as the smallest bidirectional MLM starting point.
- **Nucleotide sequences**: Prepare FASTA with `-input-format fasta`, then use `nucleotide_dna_causal_tiny.json` or `nucleotide_dna_mlm_tiny.json`. Match `vocab_size` to the emitted `nucleotide_vocab.json` when enabling ambiguity symbols beyond the default `N`.
- **Whole-word MLM**: Use `mlm_wwm_curriculum_tiny.json` with shards prepared using `-wwm-compatible-tokenizer`. Mixlab derives word starts from the shard-adjacent `tokenizer.json` and changes only host-side mask selection.
- **Word-structural auxiliary**: Use `word_structural_mlm_tiny.json` when you want MLM training with local shuffled-span reconstruction through the same vocab head.
- **Hybrid objectives**: Use `hybrid_tiny.json` for GPT-BERT-style causal plus masked-objective training. Set `training.hybrid_mix_granularity: "example"` when you want mixed causal and masked sequences in the same batch.
- **Block diffusion**: Use `block_diffusion_tiny.json` for the v1 block-wise masked-diffusion objective. Use `hybrid_block_diffusion_tiny.json` to mix causal and block-diffusion batches, and `block_diffusion_sampler_sweep.json` for a sampler-settings variant. Train them like any other config, then sample with `-mode generate-diffusion`. Compare diffusion runs against causal and MLM baselines on the same backbone, using causal validation loss for apples-to-apples model comparison.
- **Dual-head scoring/denoising**: Use `multihead_mntp_diffusion_tiny.json` when you want one shared trunk trained with a masked scorer head plus a native block-diffusion denoiser head. `export-hf` exports the scorer head only; `generate-diffusion` and `score-diffusion` use the configured denoiser head.
- **RTD auxiliary training**: Use `multihead_mntp_rtd_tiny.json` for tied-generator ELECTRA-style training, or `multihead_mntp_rtd_dedicated_tiny.json` when you want a separate small MLM generator. `export-hf` exports the scorer head only; `score-electra` reads native detector scores.
- **Minimal-pair ranking**: Use `multihead_mntp_energy_tiny.json` when you want a native energy head to rank clean sequences below corrupted variants. Use `multihead_mntp_span_pll_ranking_tiny.json` when you want the exportable MLM/MNTP scorer itself regularized by span-masked pseudo-log-likelihood ranking. Set `energy_aggregation: "differing_span"` for local-edit pair data with explicit or derived edit spans. `score-ebm` reads native energies or scorer span-PLL scores depending on `training.minimal_pair.score_source`.
- **Directional paired span PLL**: Use `pll_margin_mlm_tiny.json` when an explicit pair artifact identifies an unchanged target span that should be more predictable in a preferred context than a contrast context. Compile the artifact with `-mode prepare-pairs`; the optional `scripts/make_distractor_margin_pairs.py` produces conservative corpus-only agreement-attractor pairs.
- **Internal distillation**: Use `distillation_tiny.json` for causal teacher distillation, or `distillation_mntp_tiny.json` for masked MNTP distillation. Teacher checkpoints must match `vocab_size`, `seq_len`, and masked configs must also match `mlm_mask_token_id` and tokenizer artifacts when present.
- **EMA representation distillation**: Use `data2vec_hybrid_tiny.json` for experimental online data2vec-style hidden-state targets on masked objective steps. The current implementation prioritizes correctness and uses CPU EMA weight refreshes.
- **Averaged checkpoints**: Use `swa_ema_tiny.json` to keep both live final and SWA/EMA averaged weights.
- **Relative attention**: Use `deberta_relative_tiny.json` for DeBERTa-style P2C/C2P position bias. Add `relative_attention_parameterization: "shared_qk_reuse"` when you want GPT-BERT-style shared relative embeddings instead of per-block position projection weights.
- **Differential attention**: Use `differential_attention_tiny.json` when you want to experiment with DIFF Transformer-style attention noise cancellation. Compare against both a same-seed ordinary-head baseline and a half-head baseline before attributing gains to the subtraction itself.
- **Parallel hybrid branches**: Use `parallel_hybrid_branches_tiny.json` when you want bidirectional attention and a recurrent side branch to read the same input and add their outputs together. The recurrent branch starts zero-gated with `residual_scale_init: 0.0`.
- **Nonlinear test-time-training layer**: Use `ttt_mlp_tiny.json` to exercise the causal TTT-MLP token mixer. Training resets its inner MLP per sequence; eligible native generation and Hugging Face stacks retain request-owned MLP, partial-gradient, and Q/K convolution state across prefill/decode.
- **GPT-2-compatible baseline**: Use `gpt2_strict_small_2026.json` when you need a strict GPT-2-style architecture and native `GPT2LMHeadModel` export instead of Mixlab custom-code export.
- **Character/byte features**: Use `char_features_plain.json` with data prepared using `-char-vocab-size 257`.
- **Packed document boundaries**: Use `packed_segment_mask_tiny.json` when your packed shards already contain a boundary token and you want block-diagonal attention inside each packed sequence.
- **Large-batch optimizer**: Use `lamb_plain_tiny.json` as a minimal whole-model LAMB starting point.
- **Sparse feed-forward experts**: Use `moe_tiny.json` for top-k routed MoE blocks with load balancing.
- **Exploring block types**: Try `mamba_2L.json`, `retnet_2L.json`, `rwkv_2L.json`, `hgrn2_2L.json`, `ttt_mlp_tiny.json`, or `mlstm_2L.json`.
- **Custom architectures**: See `custom_geglu.json` and [custom_geglu.md](custom_geglu.md).
- **Advanced features**: `unet_transformer.json` and `recurrent_parallel.json` cover U-Net skips, recurrence, parallel residuals, block scales, residual mixing, tied embeddings, and TTT.

## Companion documentation

- [plain_3L.md](plain_3L.md) — Plain transformer field reference
- [custom_geglu.md](custom_geglu.md) — Custom block JSON format reference
