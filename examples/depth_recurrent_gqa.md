# depth_recurrent_gqa

Depth-recurrent grouped-query transformer inspired by the current competition #2 architecture.

Key features:
- 544-wide model with `vocab_size: 8192` and `seq_len: 1024`
- GQA with `heads: 8` and `kv_heads: 4`
- Learnable QK gain enabled on each attention block via numeric `qk_gain`
- `parallel_residual: true` so each attention/FFN pair shares the same pre-norm input
- Full depth recurrence via `weight_group`, giving 10 executed blocks from 1 shared attention block and 1 shared FFN block

Note:
- The competition entry used a LeakyReLU^2 FFN, but mixlab's current `parallel_residual` implementation requires `(plain, swiglu)` pairs, so this example uses shared `swiglu` blocks to stay runnable.

Train with:

```bash
./mixlab -mode arch -config examples/depth_recurrent_gqa.json -train 'benchmarks/data/shakespeare_char/train_*.bin'
```
