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

For `training.objective: "classification"`, `eval` consumes labeled
validation shards and reports cross-entropy loss, accuracy, multiclass MCC,
macro-F1, and binary AUROC. BPB and per-token LM exports do not apply to this
objective.

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

## `score-diffusion`

Score already-tokenized candidate sequences with the native block-diffusion
forward. Multihead checkpoints score with the configured `training.diffusion_head`.

```bash
./mixlab -mode score-diffusion \
  -config examples/hybrid_block_diffusion_tiny.json \
  -safetensors-load runs/hybrid-block-diffusion.safetensors \
  -score-in candidates.jsonl \
  -score-out scores.jsonl
```

| Flag | Description |
|------|-------------|
| `-config` | Required. Must be a pure `block_diffusion` config, a hybrid config whose secondary objective is `block_diffusion`, or a multihead config with a block-diffusion `diffusion_head`. |
| `-safetensors-load` | Required. Checkpoint to score with. |
| `-score-in` | Required. JSONL input with `{"id": "...", "tokens": [...]}` records. Tokens are raw token IDs; mixlab does not tokenize text in this mode. |
| `-score-out` | Required. JSONL output with `logprob_sum`, `logprob_mean`, and `per_token` block-causal PLL values. |
| `-score-mode` | Scoring mode. V1 supports only `block_causal`, the default. |
| `-score-skip-first` | Globally skip the first N tokens when summing scores. Per-record `score_from` overrides this value. |
| `-score-position-batch` | Number of masked positions per forward. `0` auto-selects a value targeting about 256 MiB of logits output. |

Input JSONL:

```json
{"id":"case_0","tokens":[1,815,22,4],"score_from":1}
```

Output JSONL:

```json
{"id":"case_0","n_tokens":3,"score_from":1,"logprob_sum":-7.42,"logprob_mean":-2.47,"per_token":[-2.1,-2.8,-2.52]}
```

`score-diffusion` masks each scored position exactly once and runs the
prefix-plus-block attention graph used by block-diffusion training. The result
is a deterministic block-causal pseudo-log-likelihood for forced-choice ranking,
not a true normalized sequence likelihood.

## `score-electra`

Score already-tokenized candidate sequences with a native multihead RTD
detector. The mode reads `head_<rtd-head>_logits` and reports per-token
`log P(original)` from `log_sigmoid(detector_logit)`.

```bash
./mixlab -mode score-electra \
  -config examples/multihead_mntp_rtd_tiny.json \
  -safetensors-load runs/mntp-rtd.safetensors \
  -score-in candidates.jsonl \
  -score-out electra_scores.jsonl
```

| Flag | Description |
|------|-------------|
| `-config` | Required. Must be a multihead config with `training.rtd` and exactly one `objective: "rtd"` detector head. |
| `-safetensors-load` | Required. Checkpoint to score with. |
| `-score-in` | Required. JSONL input with `{"id": "...", "tokens": [...]}` records. Tokens are raw token IDs. |
| `-score-out` | Required. JSONL output with `logprob_sum`, `logprob_mean`, and scored `per_token` values. |
| `-score-skip-first` | Globally skip the first N tokens when summing scores. Per-record `score_from` overrides this value. |
| `-score-batch` | Sequence rows per detector forward. `0` uses a conservative default. |

`score-electra` does not run generator corruption. It evaluates the supplied
sequence as-is and is intended as a native detector-based ranking signal, not a
normalized language-model likelihood.

## `score-ebm`

Score already-tokenized sequences or clean/corrupt pairs with either a native
multihead energy head, an MLM/MNTP scorer span-PLL minimal-pair configuration,
or native full-sequence masked PLL for MLM/MNTP scorer configs. Energy mode is
lower-is-better; scorer PLL modes are higher-is-better.

```bash
./mixlab -mode score-ebm \
  -config examples/multihead_mntp_energy_tiny.json \
  -safetensors-load runs/mntp-energy.safetensors \
  -score-in pairs.jsonl \
  -score-out ebm_scores.jsonl
```

Input JSONL can contain single-sequence rows:

```json
{"id":"case_0","tokens":[1,10,11,2]}
```

or pair rows:

```json
{"id":"pair_0","clean":[1,10,11,2],"corrupt":[1,10,19,2],"family":"agreement"}
```

With `-sequence-vocab data/genome/nucleotide_vocab.json`, nucleotide records
may use `sequence`, or `clean_sequence` and `corrupt_sequence`, instead of token
arrays. Output repeats the canonical decoded strings alongside native scores.

For configs with `training.minimal_pair.energy_aggregation: "differing_span"`,
single-sequence rows may include `span:[start,end]`, and pair rows may include
`clean_span:[start,end]` and `corrupt_span:[start,end]`. When pair spans are
omitted, Mixlab derives the differing spans by token alignment.

Pair outputs include `energy_clean`/`energy_corrupt` for native energy configs
or `score_clean`/`score_corrupt` for `score_source: "mlm_span_pll"` configs.
`margin` is positive when the clean side wins in either mode. The mode appends a
`__summary__` row with aggregate and per-family pair accuracy when pair rows
were scored.

Use `-score-pll-aggregation full_seq` to compute full-sequence masked PLL for
MLM/MNTP scorer configs. This masks each scored token position in turn and sums
`log softmax(logits)[gold_token]`. Explicit `differing_span` scores only the
alignment-derived or supplied differing span, and `dependent_window` scores that
span plus `-score-pll-window K` context tokens on each side. These explicit PLL
aggregations are deterministic score-time alternatives; `config` preserves the
legacy native scoring path for existing `mlm_span_pll` configs.
`training.mlm_mask_token_id` is always skipped. Pass tokenizer special IDs such as
`<unk>`, BOS, EOS, and PAD through `-score-pll-skip-token-ids` when they should
be excluded from the PLL sum. Length-normalized PLL variants are intentionally
not included in v1 because they are an audit/experiment concern rather than a
default scoring policy.

`-score-pll-attribution-dump path.jsonl` writes one audit record per pair for
PLL scorer modes. Each record contains clean/corrupt token IDs, sequence-aligned
per-token log-prob arrays, derived differing spans, skipped token IDs, and the
same aggregate scores/margin/correct fields written to `-score-out`.

| Flag | Description |
|------|-------------|
| `-config` | Required. Must be a multihead config with either an `objective: "energy"` head or `training.minimal_pair.score_source: "mlm_span_pll"`, or a single-objective `mlm`/`mntp` config for full-sequence PLL. |
| `-safetensors-load` | Required. Checkpoint to score with. |
| `-score-in` | Required. JSONL input with `tokens` rows or `clean`/`corrupt` pair rows. |
| `-score-out` | Required. JSONL output path. |
| `-score-batch` | Even sequence rows per energy forward. `0` uses a conservative default. |
| `-score-position-batch` | Masked positions per explicit PLL forward. `0` auto-selects a memory-bounded value. |
| `-score-pll-aggregation` | `config`, `full_seq`, `differing_span`, or `dependent_window`. `config` preserves current behavior: native energy uses energy scoring, multihead `mlm_span_pll` uses the configured span-PLL scoring path, and single-objective `mlm`/`mntp` uses full-sequence PLL. |
| `-score-pll-window` | Context window for `dependent_window`; `0` is equivalent to `differing_span`. |
| `-score-pll-attribution-dump` | Optional JSONL path for per-pair PLL attribution records. |
| `-score-pll-skip-token-ids` | Comma-separated token IDs to skip in PLL aggregation, in addition to the mask token. |
| `-score-emit-token-energy` | Include per-token energy arrays for differing-span native energy configs. Not supported for scorer span-PLL mode. |
| `-sequence-vocab` | Optional versioned nucleotide vocabulary enabling sequence-string input/output. |

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
