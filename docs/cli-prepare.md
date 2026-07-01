# CLI: Data Preparation

`prepare` tokenizes raw text or JSONL and writes binary shards that Mixlab can
train and evaluate against. See [data.md](data.md) for corpus examples and
tokenizer compatibility notes.

```bash
./mixlab -mode prepare \
  -input data/raw/corpus.jsonl \
  -prepare-output-dir data/example \
  -vocab-size 16384 \
  -text-field text
```

| Flag | Description |
|------|-------------|
| `-input` | Required. Input text file, JSONL file, or directory. |
| `-prepare-output-dir` | Output directory for shards and tokenizer artifacts. Preferred alias for legacy `-output`. |
| `-output` | Legacy output directory. |
| `-vocab-size` | BPE vocabulary size when training a tokenizer. Default: `1024`. |
| `-val-split` | Fraction of tokens reserved for validation. Default: `0.1`. |
| `-tokenizer-path` | Existing `tokenizer.json` to reuse instead of training a tokenizer. |
| `-text-field` | JSONL field that contains text. Default: `text`. |
| `-char-vocab-size` | Write tokenizer-level `char_features.bin` with this char vocab size. `0` disables. |
| `-char-max-per-token` | Fixed char feature slots per token when char features are enabled. Default: `16`. |
| `-minimal-pair-out` | Optional JSONL path for corpus-derived clean/corrupt minimal-pair records. |
| `-minimal-pair-corruptions` | Comma-separated corruption families. Default: `agreement,attractor,word_order`. |
| `-minimal-pair-max-pairs` | Maximum generated minimal pairs. `0` lets `prepare` choose from input size. |
| `-minimal-pair-seed` | Deterministic seed for minimal-pair generation. Default: `1234`. |

When `-char-vocab-size` is enabled, `prepare` writes a reusable
tokenizer-level `char_features.bin` next to `tokenizer.json`. Configs with
`char_vocab_size > 0` expect that artifact during training, eval, and
generation.

When `-minimal-pair-out` is set, `prepare` writes JSONL records shaped as
`{"id":"...","clean":[...],"corrupt":[...],"family":"..."}`. The generator
uses only the input corpus text and tokenizer, applies broad stochastic
corruptions, filters duplicates, and prints accepted/rejected counts by family.
Use that file from `training.minimal_pair.path` with a multihead energy head.
