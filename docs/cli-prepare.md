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
| `-minimal-pair-weights` | Optional family weights as JSON (`{"agreement":1}`) or comma-separated `family=value` entries. Missing enabled families default to `1`. |
| `-minimal-pair-morphology` | Morphology source for generated pairs. V1 supports `induced`, using corpus-token frequency tables plus built-in broad alternations. |
| `-minimal-pair-max-pairs` | Maximum generated minimal pairs. `0` lets `prepare` choose from input size. |
| `-minimal-pair-seed` | Deterministic seed for minimal-pair generation. Default: `1234`. |
| `-minimal-pair-report-out` | Optional JSON report with accepted/rejected counts, rejection reasons, weights, attempts, and seed. |
| `-minimal-pair-sample-out` | Optional audit JSONL containing clean/corrupt text and token IDs for a bounded sample of accepted pairs. |
| `-minimal-pair-sample-count` | Maximum audit samples to write. Default: `20`. |

When `-char-vocab-size` is enabled, `prepare` writes a reusable
tokenizer-level `char_features.bin` next to `tokenizer.json`. Configs with
`char_vocab_size > 0` expect that artifact during training, eval, and
generation.

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
