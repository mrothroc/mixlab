# gated_linear_attention

GatedDeltaNet-style linear-attention example inspired by the current competition #3 architecture.

Key features:
- 544-wide model with `vocab_size: 8192`, `seq_len: 1024`, and 10 recurrent custom blocks
- Time-varying gated recurrence via `scan_tv`, plus auxiliary `outer` and `matrix_scan` state updates inside each custom block
- ReLU^2 channel-mixing MLPs via `activation: "leaky_relu_sq"` and `leaky_slope: 0.0`
- `logit_softcap: 30.0` and hashed bigram embeddings with `bigram_vocab_size: 3072`, `bigram_dim: 112`

Note:
- mixlab's `kv_source` feature only applies to `plain` attention blocks today, so the stride-2 KV-sharing aspect is not encoded directly in this custom-block example.

Train with:

```bash
./mixlab -mode arch -config examples/gated_linear_attention.json -train 'benchmarks/data/shakespeare_char/train_*.bin'
```
