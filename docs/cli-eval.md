# CLI: Count, Eval, And Hidden States

These modes inspect existing configs or checkpoints. They do not update model
weights.

## `count`

Print model size and IR accounting:

```bash
./mixlab -mode count -config examples/plain_3L.json
```

| Flag | Description |
|------|-------------|
| `-config` | Required. JSON architecture config. |

## `eval`

Evaluate a checkpoint:

```bash
./mixlab -mode eval \
  -config examples/plain_3L.json \
  -safetensors-load weights.safetensors \
  -train 'data/example/val_*.bin'
```

| Flag | Description |
|------|-------------|
| `-config` | Required. JSON architecture config. |
| `-safetensors-load` | Required. Checkpoint to evaluate. |
| `-train` | Required. Shard glob used as the eval token stream. |
| `-lut-dir` | Directory containing BPB lookup tables. Default: `data`. |

Per-token export flags can be combined in a single eval pass:

| Flag | Output |
|------|--------|
| `-logprobs-out PATH` | Binary file of per-token NLLs: `logprobs.Record{TokenID, NLL}`. |
| `-ranks-out PATH` | Binary file of target ranks: `ranks.Record{TokenID, Rank}`. Rank is 0-indexed; rank 0 means the target was the model's argmax. |
| `-uncertainty-out PATH` | Binary file of candidate uncertainty: `uncertainty.Record{TokenID, Top1Prob, Entropy, Margin}`. |
| `-logits-out PATH` | Binary file of full-vocab outputs: `logits.Record{TokenID, Values[vocab]}`. |
| `-logits-dtype MODE` | On-disk dtype for `-logits-out`: `float16` or `float32`. |
| `-logits-form MODE` | Encoding for `-logits-out`: `raw` or `logprobs`. |

When multiple export flags are supplied, the records are aligned
position-by-position and derived from one GPU pass over the validation shard.

### Reading ranks

```python
import struct, numpy as np
header = struct.Struct("<IIII")  # magic, version, vocab, total_tokens
record = np.dtype([("token_id", "<u2"), ("rank", "<u2")])
with open("ranks.bin", "rb") as f:
    _, _, vocab, n = header.unpack(f.read(16))
    arr = np.fromfile(f, dtype=record, count=n)
hit_at_1  = (arr["rank"] == 0).mean()
hit_at_5  = (arr["rank"] <  5).mean()
hit_at_10 = (arr["rank"] < 10).mean()
mrr       = (1.0 / (arr["rank"].astype(np.float64) + 1.0)).mean()
print(f"Hit@1={hit_at_1:.4f} Hit@5={hit_at_5:.4f} Hit@10={hit_at_10:.4f} MRR={mrr:.4f}")
```

### Reading uncertainty

```python
import struct, numpy as np
header = struct.Struct("<IIII")  # magic, version, vocab, total_tokens
record = np.dtype([("token_id", "<u2"), ("top1_prob", "<f4"),
                   ("entropy", "<f4"), ("margin", "<f4")])
with open("uncertainty.bin", "rb") as f:
    _, _, vocab, n = header.unpack(f.read(16))
    arr = np.fromfile(f, dtype=record, count=n)
mean_top1 = arr["top1_prob"].mean()
mean_entropy = arr["entropy"].mean()
mean_margin = arr["margin"].mean()
threshold = np.quantile(arr["top1_prob"], 0.10)
abstain_mask = arr["top1_prob"] < threshold
print(f"mean top1_prob={mean_top1:.4f} entropy={mean_entropy:.4f} margin={mean_margin:.4f}")
print(f"abstaining on bottom 10% by top1_prob: threshold={threshold:.4f} positions={abstain_mask.sum()}")
```

### Reading logits

The logits header carries dtype and form, so one reader handles float16,
float32, raw logits, and log-probabilities.

```python
import struct, numpy as np

header = struct.Struct("<IIIIBB2s")  # magic, version, vocab, total, dtype, form, reserved
DTYPE = {0: np.float16, 1: np.float32}
FORM  = {0: "raw",      1: "logprobs"}

with open("logits.bin", "rb") as f:
    magic, version, vocab, n, dtype_id, form_id, _ = header.unpack(f.read(20))
    if magic != 0x4C4F4754 or version != 1:
        raise ValueError(f"unexpected logits header: magic={magic:#x} version={version}")
    dt = DTYPE[dtype_id]
    record = np.dtype([("token_id", "<u2"), ("logits", dt, (vocab,))])
    arr = np.fromfile(f, dtype=record, count=n)
print(f"loaded {n} positions @ vocab={vocab} dtype={dt.__name__} form={FORM[form_id]}")

top1 = arr["logits"].argmax(axis=1)

if FORM[form_id] == "raw":
    row = arr["logits"].astype(np.float32)
    logZ = np.log(np.exp(row - row.max(axis=1, keepdims=True)).sum(axis=1)) + row.max(axis=1)
    nll  = logZ - row[np.arange(n), arr["token_id"]]
    print(f"mean NLL = {nll.mean():.6f}")
```

When `-logits-out` is combined with `-logprobs-out`, recovered NLLs match the
values in `logprobs.bin` to float32/float16 tolerance.

## `hiddenstats`

Export one batch of hidden states:

```bash
./mixlab -mode hiddenstats \
  -config examples/plain_3L.json \
  -safetensors-load weights.safetensors \
  -train 'data/example/val_*.bin' \
  -hiddenstats-out hidden.bin
```

| Flag | Description |
|------|-------------|
| `-config` | Required. JSON architecture config. |
| `-safetensors-load` | Required. Checkpoint to inspect. |
| `-train` | Required. Shard glob used for the input batch. |
| `-hiddenstats-out` | Output float32 binary file. Preferred alias for legacy `-output`. |
| `-output` | Legacy output path. |
