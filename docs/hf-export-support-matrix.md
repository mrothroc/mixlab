# Hugging Face Export Support Matrix

This matrix is the source of truth for `mixlab -mode export-hf` support. Exported models are standard Hugging Face custom-code `AutoModel` and `AutoModelForCausalLM` directories, with `AutoModelForMaskedLM` added for MLM/MNTP/hybrid checkpoints. Supported features must preserve backbone hidden states, causal next-token inference logits, and masked-LM logits when that head is exported. Features that only affect training loss or require native recurrent scan state are rejected with explicit errors.

| Feature | Status | Notes |
|---------|--------|-------|
| `plain` attention | Supported | Adjacent-pair RoPE by default, optional `rope_convention: "half_rotation"`, partial `rope_dims`, GQA through `kv_heads`, `qk_norm`, `qk_gain`, XSA projection, sparse attention gates, `causal`/`bidirectional`/`none` masks, and causal `window_size`. |
| `plain.qk_norm` | Supported | Learned per-head-dimension Q/K RMSNorm scales before RoPE or DeBERTa relative score construction. |
| `plain.xsa` | Supported | Attention outputs are projected away from the corresponding value vector before the output projection. |
| `plain.sparse_attn_gate` | Supported | Per-head attention outputs are gated from the first token-state channels before head merge and output projection. |
| `plain.relative_attention=deberta_p2c_c2p` | Supported | DeBERTa/GPT-BERT C2P/P2C relative-bias operator semantics with Mixlab's per-block projected position tensors, `2 * bucket - 1` table rows, log-bucketed `q-k` relative positions, masks, and optional `qk_norm`/`qk_gain`. |
| `swiglu` | Supported | Bias-free SwiGLU FFN parity. |
| `geglu` | Supported | Bias-free GEGLU FFN parity. |
| `mlp` | Supported | `silu`, `gelu`, `relu`, and `leaky_relu_sq` activation variants. |
| `moe` | Supported | Sequential linear router with top-k token routing and `swiglu`, `geglu`, or `mlp` experts. Load-balancing auxiliary loss is training-only and not part of exported logits. |
| `char`, `bigram`, `trigram` embedding feature channels | Supported | `char_features.bin` is copied into the HF directory; n-gram IDs mirror Mixlab native token-id lookup semantics. |
| `tie_embeddings=true` | Supported | The exporter materializes `lm_head_weight` as the transpose of the input embedding table for broad Hugging Face consumer compatibility. |
| `AutoModel` backbone | Supported | `config.json` includes an `AutoModel` mapping to `MixlabModel`, which returns `last_hidden_state` before the LM head. |
| `AutoModelForMaskedLM` | Supported for masked-capable checkpoints | MLM, MNTP, and hybrid exports register `MixlabForMaskedLM`. It reuses the same exported `lm_head_weight`, computes unshifted masked-label cross entropy with `-100` ignored labels, and constructs plain attention blocks with bidirectional masks. |
| `training.data2vec` | Supported as stripped training-only state | The exporter ignores appended `data2vec_pred*` tensors and omits the training-only data2vec spec, exporting the student/base inference model. |
| `training.objective="causal"` | Supported | Exports a causal next-token graph. |
| `training.objective="hybrid"` | Supported | Exports both the causal evaluation graph and a masked-LM graph. Block-level `attention_mask` values are overridden by head: causal for `AutoModelForCausalLM`, bidirectional for `AutoModelForMaskedLM`. |
| `training.objective="mlm"` or `"mntp"` | Supported | Exports both heads. `AutoModelForMaskedLM` is the intended masked-eval path; `AutoModelForCausalLM` remains available for consumers that need next-token logits from the same checkpoint. |
| `hgrn2` | Gated | Matrix-state scan export is deferred until the PyTorch template has explicit recurrent-state parity coverage. |
| `mlstm` | Gated | Stabilized matrix-memory scan export is deferred until the PyTorch template has explicit recurrent-state parity coverage. |
| `gated_deltanet` | Gated | Chunked delta-rule recurrence uses native scan semantics not yet mirrored by the HF template. |
| `mamba`, `gated_linear_ssm`, `mamba3`, `mamba3-canonical` | Gated | Selective-scan and canonical Mamba-3 paths rely on native scan and backend-specific execution details. |
| `retnet` and `rwkv` | Gated | Recurrent/retention semantics need dedicated exported-state parity fixtures before support is enabled. |
| `custom` blocks | Unsupported | Arbitrary JSON custom op graphs cannot be safely converted into one static generated template. |
| `kv_source` | Gated | KV sharing export needs dedicated HF parity coverage before support is enabled. |
| `parallel_residual`, `block_scales`, `resid_mix`, `unet`, `backout`, recurrence weight sharing | Gated | These structural features alter weight layout or multi-stream graph semantics beyond the current exporter. |
| MTP, first-byte masked loss, distillation, eval-time TTT | Training/eval-only | Rejected because they do not change the exported causal forward logits in a representable HF way or require separate runtime policy. |

Unsupported and gated features fail before an incomplete directory is written. This is intentional: export should either produce parity-tested logits or return a field-specific error.
