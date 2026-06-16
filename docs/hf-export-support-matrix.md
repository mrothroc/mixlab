# Hugging Face Export Support Matrix

This matrix is the source of truth for `mixlab -mode export-hf` support. Exported models are standard Hugging Face custom-code `AutoModel` and `AutoModelForCausalLM` directories, so supported features must preserve backbone hidden states and causal next-token inference logits. Features that only affect training loss or require native recurrent scan state are rejected with explicit errors.

| Feature | Status | Notes |
|---------|--------|-------|
| `plain` attention | Supported | Adjacent-pair RoPE by default, optional `rope_convention: "half_rotation"`, partial `rope_dims`, GQA through `kv_heads`, `qk_norm`, `qk_gain`, `causal`/`bidirectional`/`none` masks, and causal `window_size`. |
| `plain.qk_norm` | Supported | Learned per-head-dimension Q/K RMSNorm scales before RoPE or DeBERTa relative score construction. |
| `plain.relative_attention=deberta_p2c_c2p` | Supported | DeBERTa C2P/P2C relative bias, projected position key/query tensors, clipping window, masks, and optional `qk_norm`/`qk_gain`. |
| `swiglu` | Supported | Bias-free SwiGLU FFN parity. |
| `geglu` | Supported | Bias-free GEGLU FFN parity. |
| `mlp` | Supported | `silu`, `gelu`, `relu`, and `leaky_relu_sq` activation variants. |
| `moe` | Supported | Sequential linear router with top-k token routing and `swiglu`, `geglu`, or `mlp` experts. Load-balancing auxiliary loss is training-only and not part of exported logits. |
| `char`, `bigram`, `trigram` embedding feature channels | Supported | `char_features.bin` is copied into the HF directory; n-gram IDs mirror Mixlab native token-id lookup semantics. |
| `tie_embeddings=true` | Supported | The exporter materializes `lm_head_weight` as the transpose of the input embedding table for broad Hugging Face consumer compatibility. |
| `AutoModel` backbone | Supported | `config.json` includes an `AutoModel` mapping to `MixlabModel`, which returns `last_hidden_state` before the LM head. |
| `training.data2vec` | Supported as stripped training-only state | The exporter ignores appended `data2vec_pred*` tensors and omits the training-only data2vec spec, exporting the student/base inference model. |
| `training.objective="causal"` | Supported | Exports a causal next-token graph. |
| `training.objective="hybrid"` | Supported for inference | Exports the causal evaluation graph. Block-level `attention_mask` values are overridden to causal in exported config because hybrid masked steps are training-only. |
| `training.objective="mlm"` or `"mntp"` | Training-only | Rejected because masked-objective programs are not `AutoModelForCausalLM` inference graphs. |
| `hgrn2` | Gated | Matrix-state scan export is deferred until the PyTorch template has explicit recurrent-state parity coverage. |
| `mlstm` | Gated | Stabilized matrix-memory scan export is deferred until the PyTorch template has explicit recurrent-state parity coverage. |
| `gated_deltanet` | Gated | Chunked delta-rule recurrence uses native scan semantics not yet mirrored by the HF template. |
| `mamba`, `gated_linear_ssm`, `mamba3`, `mamba3-canonical` | Gated | Selective-scan and canonical Mamba-3 paths rely on native scan and backend-specific execution details. |
| `retnet` and `rwkv` | Gated | Recurrent/retention semantics need dedicated exported-state parity fixtures before support is enabled. |
| `custom` blocks | Unsupported | Arbitrary JSON custom op graphs cannot be safely converted into one static generated template. |
| `kv_source`, XSA, sparse attention gates | Gated | These attention sub-features need dedicated HF parity coverage before export. |
| `parallel_residual`, `block_scales`, `resid_mix`, `unet`, `backout`, recurrence weight sharing | Gated | These structural features alter weight layout or multi-stream graph semantics beyond the current exporter. |
| MTP, first-byte masked loss, distillation, eval-time TTT | Training/eval-only | Rejected because they do not change the exported causal forward logits in a representable HF way or require separate runtime policy. |

Unsupported and gated features fail before an incomplete directory is written. This is intentional: export should either produce parity-tested logits or return a field-specific error.
