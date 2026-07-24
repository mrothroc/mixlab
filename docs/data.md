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
mixlab -mode prepare -input corpus.txt -prepare-output-dir data/my_data -vocab-size 1024

# Directory of text files
mixlab -mode prepare -input texts/ -prepare-output-dir data/my_data -vocab-size 4096

# JSONL
mixlab -mode prepare -input data.jsonl -prepare-output-dir data/my_data \
    -vocab-size 8192 -text-field content
```

Or use a pre-trained tokenizer:

```bash
mixlab -mode prepare -input corpus.txt -prepare-output-dir data/my_data \
    -tokenizer-path path/to/tokenizer.json
```

`prepare` requires Python 3 with `numpy` and `tokenizers`.

```bash
pip install numpy tokenizers
```

Tokens are stored as uint16, so `vocab-size` must be 65,535 or less.

## Nucleotide FASTA

Base-level DNA and RNA data use the same discrete embedding/model path as text.
The default preparation mode preserves each FASTA contig as a record:

```bash
mixlab -mode prepare -input genome.fasta -input-format fasta \
  -prepare-output-dir data/genome -nucleotide-alphabet dna \
  -nucleotide-ambiguous-symbols N,R,Y
```

The emitted `nucleotide_vocab.json` records every symbol, ID, complement, and
invalid-symbol decision. The record-oriented loader frames and packs contig
chunks at runtime. Causal targets at EOS and all cross-contig attention are
masked, while MLM position selection considers biological symbols only.

Recurrent and SSM causal models cannot consume the record packer's
block-diagonal attention mask. Prepare a continuous nucleotide stream instead:

```bash
mixlab -mode prepare -input genome.fasta -input-format fasta \
  -prepare-output-dir data/genome-stream \
  -nucleotide-framing stream \
  -nucleotide-stream-separator eos
```

This writes `mixlab_token_shard_v1` with
`sequence_layout: "continuous_stream"`. Configure fixed causal rows explicitly:

```json
{
  "seq_len": 1024,
  "training": {
    "objective": "causal",
    "example_framing": {"content_len": 1022, "bos_id": 1, "eos_id": 2},
    "reverse_complement_prob": 0.5
  }
}
```

The validation split is still made by contig before either split is flattened.
EOS is inserted between contigs by default; use
`-nucleotide-stream-separator none` only when direct cross-contig transitions
are intentional. Runtime framing resets recurrent state at every row and masks
the final row target. No attention segment mask is emitted or required.

Common flags:

| Flag | Description |
|------|-------------|
| `-input` | Input text file, JSONL file, or directory. |
| `-prepare-output-dir` | Output directory for binary shards. Preferred alias for legacy `-output`. |
| `-vocab-size` | BPE vocabulary size. Default: `1024`. |
| `-val-split` | Fraction of tokens reserved for validation. Default: `0.1`. |
| `-tokenizer-path` | Path to a pre-trained `tokenizer.json`. |
| `-text-field` | JSON field name for text in JSONL input. Default: `text`. |
| `-label-field` | JSONL integer-label field for sequence classification. |
| `-label-file` | FASTA sibling `id<TAB>label` TSV for sequence classification. |
| `-char-vocab-size` | Generate tokenizer-level byte/character feature lookup when enabled. |
| `-char-max-per-token` | Maximum byte/character slots per token for char features. |

## Dataset manifest

`prepare` writes `mixlab.dataset.json` beside the generated shards. This is a
versioned description of the sequence representation; it does not replace or
modify the binary shard format. A prepared text dataset currently looks like:

```json
{
  "format": "mixlab.dataset",
  "version": 1,
  "representation": "discrete_tokens",
  "modality": "text",
  "vocab_size": 16384,
  "token_dtype": "uint16",
  "shard_format": "mixlab_token_shard_v1",
  "special_token_ids": {"[PAD]": 0, "[MASK]": 4},
  "artifacts": {"tokenizer": "tokenizer.json"},
  "splits": {
    "train": {"pattern": "train_*.bin", "tokens": 1000000, "shards": 1},
    "val": {"pattern": "val_*.bin", "tokens": 100000, "shards": 1}
  }
}
```

Paths and patterns are relative to the manifest directory. When a manifest is
present, Mixlab validates its schema and requires its `vocab_size` to match the
model before constructing a trainer. Existing shard directories without a
manifest remain supported for backward compatibility.

Discrete datasets support `shard_format: "mixlab_token_shard_v1"` for flat
token streams, including `sequence_layout: "continuous_stream"` nucleotide
data, `"mixlab_sequence_shard_v1"` for record-oriented sequences, and
`"mixlab_labeled_sequence_shard_v1"` for atomic record-plus-label
classification data.
Nucleotide split entries additionally report `sequences`, and the manifest
points to `artifacts.vocabulary: "nucleotide_vocab.json"`. The `modality` field
describes the sequence domain without changing the backbone; later releases
will introduce separate
versioned representations for continuous features.

Text datasets prepared with `-frame-per-record` also use
`mixlab_sequence_shard_v1`, with `sequence_layout: "one_record_per_row"` and a
required `record_seq_len`. Each shard record contains content tokens only. The
loader adds BOS/EOS and trailing PAD at runtime, places every record at position
zero in its own row, and masks every PAD target from causal loss. This is the
appropriate layout when source records are complete examples and must not be
split or packed together. Record mode splits validation data by record rather
than by token offset.

Labeled sequence manifests additionally declare:

```json
{
  "shard_format": "mixlab_labeled_sequence_shard_v1",
  "sequence_layout": "one_record_per_row",
  "record_seq_len": 502,
  "task": {
    "type": "single_label_classification",
    "num_labels": 2
  },
  "splits": {
    "train": {"sequences": 900, "class_counts": {"0": 450, "1": 450}},
    "val": {"sequences": 100, "class_counts": {"0": 50, "1": 50}}
  }
}
```

The loader keeps each label aligned with its variable-length token record,
frames the record as `[BOS] content [EOS] [PAD]...`, and emits a validity mask.
For `plain` attention, PAD occupies a separate segment so bidirectional
classification cannot use padding as context.

DNA datasets also carry the complement table used by top-level
`rc_equivariant: true`. For packed MLM, biological positions reverse only
inside their original segment. For one-record classification, content bases
reverse inside the framed record while BOS, EOS, PAD, and MASK remain fixed
and self-complementary. RNA datasets reject this option.

## Data/config compatibility

The `vocab_size` in your JSON config must match the tokenizer used to create
the `.bin` shards. Prepared datasets record the tokenizer's actual emitted
vocabulary size in `mixlab.dataset.json`, which can be lower than a requested
maximum when a small corpus does not contain enough merge candidates.

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

Packed streams and per-record framing solve different problems. Segment masks
preserve attention boundaries inside densely packed rows, while per-record
framing preserves the generation-time condition that every row starts at BOS
position zero. `training.example_framing` is a third mode for fixed chunks cut
from an otherwise continuous raw token stream; it does not preserve source
record boundaries.
