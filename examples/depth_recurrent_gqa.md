# depth_recurrent_gqa

Depth-recurrent grouped-query transformer. Applies the same attention + FFN
block repeatedly, using weight sharing to keep parameter count low while
maintaining full execution depth — a Universal Transformer style design.

Key features:
- 544-wide model with `vocab_size: 8192` and `seq_len: 1024`
- GQA with `heads: 8` and `kv_heads: 4`
- Learnable QK gain enabled on each attention block via numeric `qk_gain`
- `parallel_residual: true` so each attention/FFN pair shares the same pre-norm input
- Full depth recurrence via `weight_group`, giving 10 executed blocks from 1 shared attention block and 1 shared FFN block

Train with example data (quick test):

```bash
mixlab -mode arch -config examples/depth_recurrent_gqa.json \
    -train 'benchmarks/data/shakespeare_char/train_*.bin'
```

Train on FineWeb-Edu (real-scale):

```bash
# Prepare data first (see README.md "Training data" section)
python3 scripts/download_fineweb.py --output data/fineweb_sp8192 --vocab-size 8192

mixlab -mode arch -config examples/depth_recurrent_gqa.json \
    -train 'data/fineweb_sp8192/train_*.bin'
```
