# gated_linear_attention

GatedDeltaNet-style linear attention with time-varying gated recurrence. Uses
`scan_tv` for the main sequence mixing, with auxiliary `outer` and
`matrix_scan` state updates — all defined as custom block ops in JSON.

Key features:
- 544-wide model with `vocab_size: 8192`, `seq_len: 1024`, and 10 custom blocks
- Time-varying gated recurrence via `scan_tv`, plus auxiliary `outer` and `matrix_scan` state updates
- ReLU^2 channel-mixing MLPs via `activation: "leaky_relu_sq"` and `leaky_slope: 0.0`
- `logit_softcap: 30.0` and hashed bigram embeddings with `bigram_vocab_size: 3072`, `bigram_dim: 112`

Note: mixlab's `kv_source` feature only applies to `plain` attention blocks
today, so stride-2 KV sharing is not encoded in this custom-block example.

Train with example data (quick test):

```bash
mixlab -mode arch -config examples/gated_linear_attention.json \
    -train 'benchmarks/data/shakespeare_char/train_*.bin'
```

Train on FineWeb-Edu (real-scale):

```bash
# Prepare data first (see README.md "Training data" section)
python3 scripts/download_fineweb.py --output data/fineweb_sp8192 --vocab-size 8192

mixlab -mode arch -config examples/gated_linear_attention.json \
    -train 'data/fineweb_sp8192/train_*.bin'
```
