# CLI usage

mixlab uses `-mode` to select the command to run.

| Mode | Description |
|------|-------------|
| `arch` | Train a single architecture from a JSON config. The default mode. |
| `arch_race` | Train every JSON config in a directory and compare results. |
| `smoke` | Run diagnostic checks for MLX availability and GPU health. |
| `prepare` | Tokenize raw text or JSONL into binary training shards. |
| `count` | Print parameter, size, block, FLOP, and IR op counts for a config. |
| `eval` | Load safetensors and evaluate validation loss. |
| `export-hf` | Export supported safetensors checkpoints to Hugging Face model directories. |
| `parity` | Compare native Mixlab inference against a Hugging Face export. |
| `hiddenstats` | Export one batch of hidden states as float32 binary. |
| `generate` | Generate token IDs from a safetensors checkpoint (causal next-token). |
| `generate-diffusion` | Generate token IDs from a `block_diffusion` checkpoint using block-wise masked diffusion sampling. |

## arch

Train one config:

```bash
./mixlab -mode arch -config examples/plain_3L.json -train 'data/example/train_*.bin'
```

Common flags:

| Flag | Description |
|------|-------------|
| `-eval` | Run full validation BPB evaluation after training. |
| `-safetensors FILE` | Export weights to safetensors after training. |
| `-safetensors-load FILE` | Load weights before training, resume, or eval-only. |
| `-quantize MODE` | Weight quantization: `none`, `int8`, or `int6`. |
| `-quant-method METHOD` | Quantization method, including `sdclip`. |
| `-lut-dir DIR` | Directory for BPB lookup tables. Default: `data`. |
| `-checkpoint-dir DIR` | Directory for periodic safetensors checkpoints. |
| `-checkpoint-every N` | Save a checkpoint every `N` training steps. `0` disables. |
| `-timing` | Print data/GPU/validation/log timing at progress intervals. |

## arch_race

Train every `.json` config in a directory and print a ranked summary:

```bash
./mixlab -mode arch_race -configs examples/ -train 'data/example/train_*.bin'
```

## count

Print model size and IR accounting:

```bash
./mixlab -mode count -config examples/plain_3L.json
```

## eval

Evaluate a checkpoint:

```bash
./mixlab -mode eval -config examples/plain_3L.json \
  -safetensors-load weights.st -train 'data/example/train_*.bin'
```

Per-token export flags can be combined in a single eval pass:

| Flag | Output |
|------|--------|
| `-logprobs-out PATH` | Binary file of per-token NLLs: `logprobs.Record{TokenID, NLL}`. |
| `-ranks-out PATH` | Binary file of target ranks: `ranks.Record{TokenID, Rank}`. Rank is 0-indexed; rank 0 means the target was the model's argmax. |
| `-uncertainty-out PATH` | Binary file of candidate uncertainty: `uncertainty.Record{TokenID, Top1Prob, Entropy, Margin}`. |
| `-logits-out PATH` | Binary file of full-vocab outputs: `logits.Record{TokenID, Values[vocab]}`. Use `-logits-dtype float16` or `float32`, and `-logits-form raw` or `logprobs`. |

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

## export-hf

Export a supported Mixlab checkpoint as a Hugging Face model directory:

```bash
./mixlab -mode export-hf \
  -config examples/plain_3L.json \
  -safetensors-load runs/plain_3L/weights.safetensors \
  -output runs/plain_3L/hf \
  -tokenizer-path data/example/tokenizer.json
```

| Flag | Description |
|------|-------------|
| `-config` | JSON config used to train the checkpoint. |
| `-safetensors-load` | Safetensors checkpoint to export. |
| `-output` | Destination Hugging Face model directory. |
| `-tokenizer-path` | Tokenizer JSON to bundle with the export. |

The default export format is Mixlab custom-code Hugging Face export. Configs
with `hf_export_format: "gpt2"` export as native `GPT2LMHeadModel` when they
meet the strict GPT-2 compatibility rules. See [hf-export.md](hf-export.md)
for supported features and load examples.

## parity

Compare a Hugging Face export against native Mixlab inference:

```bash
./mixlab -mode parity \
  -config examples/plain_3L.json \
  -safetensors-load runs/plain_3L/weights.safetensors \
  -hf runs/plain_3L/hf \
  -train 'data/example/val_*.bin'
```

| Flag | Description |
|------|-------------|
| `-config` | JSON config used for native inference. |
| `-safetensors-load` | Native safetensors checkpoint to compare. |
| `-hf` | Hugging Face export directory to load. |
| `-train` | Shard glob used as the comparison token stream. |
| `-threshold` | Maximum allowed native-vs-HF mean NLL difference. Default: `0.05`. |
| `-max-logit-diff` | Maximum allowed absolute logit difference on sampled rows. `<=0` disables the logit gate. Default: `0.001`. |
| `-parity-logit-tokens` | Number of token pairs to sample for logit comparison, rounded up to full eval batches. `0` uses one batch. |
| `-parity-python` | Python interpreter for the HF checker. Defaults to `HF_PARITY_PYTHON` or `python3`. |

The Python checker needs the packages in `requirements-hf.txt`. Use `parity`
after `export-hf` when changing export templates, weight mapping, tokenizer
metadata, or supported block features.

## hiddenstats

```bash
./mixlab -mode hiddenstats -config examples/plain_3L.json \
  -safetensors-load weights.st -train 'data/example/train_*.bin' \
  -output hidden.bin
```

## generate

```bash
./mixlab -mode generate -config examples/plain_3L.json \
  -safetensors-load weights.st -prompt token_ids:0,1,2
```

| Flag | Description |
|------|-------------|
| `-max-tokens` | Maximum generated tokens. Default: `256`. |
| `-temperature` | Sampling temperature. Default: `0.8`. |
| `-top-k` | Top-k sampling cutoff. `0` disables the cutoff. |
| `-prompt` | Prompt token IDs in `token_ids:0,1,2` form. |

`generate` is causal next-token generation. It does not consume
`training.diffusion.steps_per_block`, `confidence_threshold`, or `commit_floor`;
those drive `generate-diffusion` instead.

## generate-diffusion

Generate from a `training.objective: "block_diffusion"` checkpoint or a hybrid
checkpoint with `training.hybrid_secondary_objective: "block_diffusion"`.
Starting from the prompt, mixlab appends a block of mask tokens, then runs up to
`steps_per_block` denoising passes per block, committing positions whose
predicted probability clears `confidence_threshold` (and at least
`commit_floor` positions per pass so every block completes) until the requested
number of tokens is produced or `seq_len` is reached.

```bash
./mixlab -mode generate-diffusion -config examples/block_diffusion_tiny.json \
  -safetensors-load weights.st -prompt token_ids:0,1,2 -max-tokens 16
```

| Flag | Description |
|------|-------------|
| `-config` | Required. Must set `training.objective: "block_diffusion"` or hybrid with `hybrid_secondary_objective: "block_diffusion"`. |
| `-safetensors-load` | Required. Trained block-diffusion weights. |
| `-max-tokens` | Maximum generated tokens (capped at `seq_len - prompt`). Default: `256`. |
| `-prompt` | Prompt token IDs in `token_ids:0,1,2` form. |
| `-diffusion-steps-per-block` | Override `training.diffusion.steps_per_block`. `0` uses the config. |
| `-diffusion-confidence-threshold` | Override `training.diffusion.confidence_threshold` when explicitly set. |
| `-diffusion-commit-floor` | Override `training.diffusion.commit_floor`. `0` uses the config. |
| `-diffusion-temperature` | Diffusion sampling temperature. `0` keeps deterministic argmax. |
| `-diffusion-top-k` | Diffusion top-k cutoff when `-diffusion-temperature > 0`. `0` disables the cutoff. |
| `-diffusion-trace-out` | Write sampler telemetry JSONL, one denoising pass per line. |

Block size and sampler behavior come from `training.diffusion`
(`block_size`, `steps_per_block`, `confidence_threshold`, `commit_floor`).
By default, sampling is deterministic argmax over unresolved positions;
`-temperature` and `-top-k` still apply only to causal `generate`. Use the
diffusion-specific temperature/top-k flags for stochastic diffusion sampling.
Output uses the same `generated token_ids:...` format as `generate`.

## prepare

`prepare` is documented in [data.md](data.md).
