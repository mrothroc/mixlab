# depth_recurrent -- Shared-Depth Transformer Demo

This example uses depth recurrence: the same transformer block is applied multiple times, similar to a Universal Transformer unrolled across depth.

`weight_group` ties block weights by name. All `plain` blocks with `weight_group: "attn"` share one attention parameter set, and all `swiglu` blocks with `weight_group: "ffn"` share one FFN parameter set.

The stack still executes 6 blocks total, but it only learns one attention block and one SwiGLU block, reused 3 times each.

Compared with a non-shared 3-pair transformer, this keeps the same compute pattern while using about 1/3 of the block parameters. Confirm that with:

```bash
./mixlab -mode count -config examples/depth_recurrent.json
```

The count output should show `unique / with sharing expanded`, where the expanded count matches the full 6-block execution.

Train it with:

```bash
./mixlab -mode arch -config examples/depth_recurrent.json -train 'data/example/train_*.bin'
```
