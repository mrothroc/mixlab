# Data preparation

mixlab trains on binary token shards. Use `prepare` for your own data, or the
provided scripts for example and FineWeb-Edu data.

## Example data

The root README quickstart uses `scripts/download_example_data.sh` to download
about 5 MB of public-domain Project Gutenberg text. This is enough to verify
setup and see loss curves, but too small for real architecture experiments.

```bash
bash scripts/download_example_data.sh
mixlab -mode arch -config examples/plain_3L.json \
    -train 'data/example/train_*.bin'
```

## FineWeb-Edu

For serious architecture exploration, use
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), a
curated and deduplicated web-text corpus. The download script fetches data from
Hugging Face, trains a BPE tokenizer, and writes binary shards.

```bash
pip install numpy tokenizers datasets

# Quick test (~30 seconds, streams 5000 documents)
python3 scripts/download_fineweb.py --output data/fineweb_sp1024 \
    --vocab-size 1024 --max-docs 5000

# Full dataset: SP-1024, small vocab for fast iteration
python3 scripts/download_fineweb.py --output data/fineweb_sp1024 --vocab-size 1024

# Full dataset: SP-8192, larger vocab for stronger BPB
python3 scripts/download_fineweb.py --output data/fineweb_sp8192 --vocab-size 8192
```

`--max-docs` streams only a subset, which is useful before committing to the
full download. Without it, the first run downloads about 20 GB from Hugging
Face. Tokenization takes 30-60 minutes. The output includes binary shards,
`tokenizer.json`, and BPB lookup tables.

Train on prepared data:

```bash
mixlab -mode arch -config examples/plain_3L.json \
    -train 'data/fineweb_sp1024/train_*.bin'

mixlab -mode arch_race -configs examples/ \
    -train 'data/fineweb_sp8192/train_*.bin'
```

## Bring your own data

mixlab can tokenize UTF-8 text files, directories of text files, or JSONL:

```bash
# Single text file
mixlab -mode prepare -input corpus.txt -output data/my_data -vocab-size 1024

# Directory of text files
mixlab -mode prepare -input texts/ -output data/my_data -vocab-size 4096

# JSONL
mixlab -mode prepare -input data.jsonl -output data/my_data \
    -vocab-size 8192 -text-field content
```

Or use a pre-trained tokenizer:

```bash
mixlab -mode prepare -input corpus.txt -output data/my_data \
    -tokenizer-path path/to/tokenizer.json
```

`prepare` requires Python 3 with `numpy` and `tokenizers`.

```bash
pip install numpy tokenizers
```

Tokens are stored as uint16, so `vocab-size` must be 65,535 or less.

Common flags:

| Flag | Description |
|------|-------------|
| `-input` | Input text file, JSONL file, or directory. |
| `-output` | Output directory for binary shards. |
| `-vocab-size` | BPE vocabulary size. Default: `1024`. |
| `-val-split` | Fraction of tokens reserved for validation. Default: `0.1`. |
| `-tokenizer-path` | Path to a pre-trained `tokenizer.json`. |
| `-text-field` | JSON field name for text in JSONL input. Default: `text`. |
| `-char-vocab-size` | Generate tokenizer-level byte/character feature lookup when enabled. |
| `-char-max-per-token` | Maximum byte/character slots per token for char features. |

## Data/config compatibility

The `vocab_size` in your JSON config must match the tokenizer used to create
the `.bin` shards. This is the most common source of bad training behavior.

```jsonc
// Config says vocab_size: 1024 -> train on SP-1024 shards
{"vocab_size": 1024}

// Config says vocab_size: 8192 -> train on SP-8192 shards
{"vocab_size": 8192}
```

If a config enables tokenizer-adjacent features such as character feature
embeddings, keep the generated artifacts next to `tokenizer.json` or the train
shards so training and generation can find the same lookup tables.

## Packed streams

For packed training streams that already contain a reliable document or segment
marker, use training-time block-diagonal attention:

```jsonc
{
  "training": {
    "attention_segment_mask": "boundary_token",
    "attention_segment_boundary_token_id": 1
  }
}
```

mixlab derives segment IDs from the unmasked input tokens and masks `plain`
self-attention so tokens attend only within their segment. Causal,
bidirectional, and hybrid masks still apply inside each segment. This is a
training feature for packed shards; prompt-time segmentation is out of scope in
v1.
