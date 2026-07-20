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
| `-input-format` | Input representation: `text` (default) or `fasta`. |
| `-prepare-output-dir` | Output directory for shards and tokenizer artifacts. Preferred alias for legacy `-output`. |
| `-output` | Legacy output directory. |
| `-vocab-size` | BPE vocabulary size when training a tokenizer. Default: `1024`. |
| `-val-split` | Fraction of tokens reserved for validation, or fraction of records with `-frame-per-record`. Default: `0.1`. |
| `-tokenizer-path` | Existing `tokenizer.json` to reuse instead of training a tokenizer. |
| `-wwm-compatible-tokenizer` | Train or validate tokenizer metadata suitable for whole-word MLM. The built-in path uses prefix-space ByteLevel BPE and reserves `[PAD]`, `[CLS]`, `[SEP]`, `[UNK]`, and `[MASK]` as ids `0..4`. |
| `-text-field` | JSONL field that contains text. Default: `text`. |
| `-frame-per-record` | Preserve each input text/JSONL record and train it as one independently framed row. |
| `-record-seq-len` | Required with `-frame-per-record`. Fixed row length including BOS/EOS and PAD. |
| `-record-pad-id` | Required PAD token ID for per-record rows. |
| `-record-bos-id` | Required BOS token ID for per-record rows. |
| `-record-eos-id` | Required EOS token ID for per-record rows. |
| `-record-overflow` | Policy when a tokenized record exceeds `record-seq-len - 2`: `error` (default), `drop`, or `truncate`. |
| `-char-vocab-size` | Write tokenizer-level `char_features.bin` with this char vocab size. `0` disables. |
| `-char-max-per-token` | Fixed char feature slots per token when char features are enabled. Default: `16`. |
| `-minimal-pair-out` | Optional JSONL path for corpus-derived clean/corrupt minimal-pair records. |
| `-minimal-pair-corruptions` | Comma-separated corruption families. Default: `agreement,attractor,word_order`. |
| `-minimal-pair-weights` | Optional family weights as JSON (`{"agreement":1}`) or comma-separated `family=value` entries. Missing enabled families default to `1`. |
| `-minimal-pair-morphology` | Morphology source for generated pairs. V1 supports `induced`, using corpus-token frequency tables plus built-in broad alternations. |
| `-minimal-pair-max-pairs` | Maximum generated minimal pairs. `0` lets `prepare` choose from input size. |
| `-minimal-pair-seed` | Deterministic seed for minimal-pair generation. Default: `1234`. |
| `-minimal-pair-report-out` | Optional JSON report with accepted/rejected counts, rejection reasons, weights, attempts, and seed. |
| `-minimal-pair-sample-out` | Optional audit JSONL containing clean/corrupt text and token IDs for a bounded sample of accepted pairs. |
| `-minimal-pair-sample-count` | Maximum audit samples to write. Default: `20`. |
| `-nucleotide-alphabet` | FASTA alphabet: `dna` (default) or `rna`. |
| `-nucleotide-ambiguous-symbols` | Comma-separated IUPAC ambiguity symbols to include. Default: `N`; complementary partners are added automatically. |
| `-nucleotide-invalid-symbol-policy` | FASTA invalid-symbol handling: `error` (default), `map_to_n`, or `skip`. `map_to_n` requires `N` in the vocabulary. |

## FASTA nucleotide data

FASTA preparation uses a fixed base-level vocabulary instead of training a
text tokenizer:

```bash
./mixlab -mode prepare \
  -input reference.fasta \
  -input-format fasta \
  -prepare-output-dir data/reference-dna \
  -nucleotide-alphabet dna \
  -nucleotide-ambiguous-symbols N,R,Y
```

The first IDs are always `<PAD>=0`, `<BOS>=1`, `<EOS>=2`, and `<MASK>=3`,
followed by `A,C,G,T` for DNA or `A,C,G,U` for RNA and then enabled ambiguity
symbols in canonical IUPAC order. Preparation writes `nucleotide_vocab.json`
and record-oriented `mixlab_sequence_shard_v1` shards. Contigs remain separate
records; the runtime packs framed contig chunks into fixed rows and supplies
block-diagonal segment IDs and loss masks automatically. Do not also set
`training.attention_segment_mask` for these datasets.

Text tokenizer, whole-word, char-feature, and minimal-pair preparation flags
are rejected with `-input-format fasta` rather than being silently ignored.

## Per-record causal examples

Use per-record framing when every source record is a complete generative
example, such as a molecule string, sentence, code snippet, or biological
sequence:

```bash
./mixlab -mode prepare \
  -input examples.jsonl \
  -text-field text \
  -tokenizer-path tokenizer.json \
  -prepare-output-dir data/examples \
  -frame-per-record \
  -record-seq-len 64 \
  -record-pad-id 0 \
  -record-bos-id 1 \
  -record-eos-id 2 \
  -record-overflow drop
```

Preparation writes variable-length records without BOS, EOS, or padding. At
load time, each record becomes `[BOS] content [EOS] [PAD]...`; causal loss is
active through the EOS target and zero for every PAD target. Records are
shuffled as units and never share or cross a row. Tokenizer-added special
tokens are disabled while encoding records so framing tokens are not doubled.

The manifest records `sequence_layout: "one_record_per_row"`, the required
`record_seq_len`, semantic token IDs, and per-split record counts, dropped and
truncated counts, and mean/max tokenized lengths. V1 supports causal training
only. Prefer `error` or `drop` when record completeness matters; `truncate` is
an explicit lossy policy.

When `-char-vocab-size` is enabled, `prepare` writes a reusable
tokenizer-level `char_features.bin` next to `tokenizer.json`. Configs with
`char_vocab_size > 0` expect that artifact during training, eval, and
generation.

When `-tokenizer-path` is used, prepare writes an exact copy of the supplied
`tokenizer.json` next to the generated shards. Add
`-wwm-compatible-tokenizer` to validate that it exposes a supported word-start
convention. Native SentencePiece `.model` files must first be converted to a
Hugging Face fast `tokenizer.json`.

Every successful text preparation also writes `mixlab.dataset.json`. The
manifest declares the discrete-token representation, text modality, actual
tokenizer vocabulary size, special tokens, tokenizer artifact, and exact split
token/shard counts. It is additive: older shard directories without a manifest
remain readable. See [data.md](data.md#dataset-manifest) for the schema.

When `-minimal-pair-out` is set, `prepare` writes JSONL records shaped as
`{"id":"...","clean":[...],"corrupt":[...],"family":"..."}`. The generator
uses only the input corpus text and tokenizer, applies broad stochastic
corruptions, applies deterministic per-family sampling weights, filters
duplicates, and reports accepted/rejected counts by family. The available
families are `agreement`, `attractor`, `word_order`, `npi_licensor`,
`quantifier_scope`, and `filler_gap`. Use `-minimal-pair-report-out` for a
machine-readable balance/rejection report and `-minimal-pair-sample-out` for
human inspection before training. Use the generated JSONL from
`training.minimal_pair.path` with a native energy head or scorer span-PLL
minimal-pair regularizer.

## `prepare-pairs`

`prepare-pairs` validates explicit minimal-pair, structured invariance-pair,
or annotated PLL-margin JSONL files, then can compile them to compact binary
shards for faster startup. It does not tokenize raw text; records must already
contain token IDs.

```bash
./mixlab -mode prepare-pairs \
  -config examples/multihead_mntp_energy_tiny.json \
  -pair-in data/pairs.train.jsonl \
  -pair-out data/pairs.train.mpair
```

You can omit `-pair-out` to run validation only. With `-config`, Mixlab uses
the config's `vocab_size` and `seq_len` as validation limits; if `-pair-in` is
omitted and the config has a JSONL `training.minimal_pair`,
`training.invariance`, or `training.pll_margin` source, Mixlab uses that path.
Without `-config`, pass
`-vocab-size`; use `-pair-max-len` to enforce a maximum view length.

| Flag | Description |
|------|-------------|
| `-pair-in` | Minimal-pair, invariance-pair, or PLL-margin JSONL input. Required unless `-config` supplies a JSONL pair path. |
| `-pair-out` | Optional compiled binary output. Omit for validation-only runs. |
| `-config` | Optional config used to infer `vocab_size`, `seq_len`, and pair input path. |
| `-vocab-size` | Token-id upper bound when no config is supplied. |
| `-pair-max-len` | Maximum pair-view length. `0` uses config `seq_len` when a config is supplied; otherwise no length cap. |

Compiled shards are used with:

```jsonc
"training": {
  "minimal_pair": {
    "source": "bin",
    "path": "data/pairs.train.mpair"
  }
}
```

Invariance records use `view_a`, `view_a_pos`, `view_b`, and `view_b_pos`.
Both positions are required and must point to the same token ID, which Mixlab
masks before comparing their vocabulary distributions. A compiled invariance
artifact can be referenced with `training.invariance.source: "file"` (automatic
format detection) or `"bin"`.

PLL-margin records use `view_pos`, `target_pos_positions`, `view_neg`,
`target_neg_positions`, and `target_ids`. Both position lists are non-empty,
strictly increasing, and must select the same token-id span supplied in
`target_ids`. Mixlab masks the complete annotated span in both views, ranks its
log pseudo-likelihood higher in `view_pos`, and anchors its prediction in that
preferred view. `training.pll_margin.source: "file"` auto-detects JSONL or its
compiled binary artifact. The target is general-purpose; the optional
`scripts/make_distractor_margin_pairs.py` tool is one conservative corpus-only
producer for agreement-attractor experiments.

When `energy_aggregation` is `"differing_span"`, pair JSONL may include
`clean_span` and `corrupt_span` half-open ranges. If spans are omitted, Mixlab
derives them by token alignment while validating or loading the pair data.
