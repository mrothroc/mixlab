# Continuous sequence input

Mixlab can train native sequence classifiers directly from fixed-shape
continuous feature records. The public boundary is modality-neutral:

```text
float32 [N,T,F] -> linear_frames -> hidden [B,T,D] -> ordinary backbone
```

Audio decoding, resampling, image patch extraction, and numeric-track
construction remain outside Mixlab. This release consumes prepared arrays; it
does not add modality-specific preprocessing.

## Prepare data

Write features as a NumPy `.npy` array with shape `[N,T,F]`, or an `.npz`
archive containing an array named `features`. Variable-length inputs may put an
integer `lengths: [N]` array in the `.npz`, or provide the same values through
`-length-file lengths.tsv`. Labels use the existing
`-label-file` surface with exact row indexes:

```text
0	3
1	0
2	3
```

Every row `0..N-1` must appear exactly once, labels must be contiguous
non-negative IDs, and at least two classes are required.

```bash
mixlab -mode prepare \
  -input features.npy \
  -input-format continuous \
  -label-file labels.tsv \
  -length-file lengths.tsv \
  -continuous-modality waveform \
  -val-split 0.1 \
  -prepare-output-dir data/waveform
```

`prepare` converts numeric inputs to little-endian float32, rejects non-finite
values, validates lengths in `[1,T]`, stratifies labeled records
deterministically, and writes `mixlab_continuous_sequence_shard_v2` shards.
Each shard stores labels, valid lengths, and frames atomically. Legacy binary
v1 shards and manifests remain readable and are treated as fully valid. The adjacent
`mixlab.dataset.json` records:

- `representation: "continuous_frames"`
- `feature_dtype: "float32"`
- `feature_dim: F`
- `record_seq_len: T`
- classification label count and per-split class counts

The model/data `feature_dim`, `seq_len`, and `num_labels` contract is checked
before trainer construction.

## Configure a model

```json
{
  "name": "continuous_mamba3_classifier",
  "model_dim": 32,
  "seq_len": 16000,
  "positional_embedding": "none",
  "input_adapter": {
    "kind": "linear_frames",
    "feature_dim": 1,
    "bias": true,
    "norm": "none"
  },
  "blocks": [
    {
      "type": "mamba3-canonical",
      "inner_dim": 32,
      "state_size": 8,
      "n_groups": 4,
      "scan_chunk_size": 64
    },
    {"type": "swiglu"}
  ],
  "training": {
    "objective": "classification",
    "classification": {"num_labels": 10, "pooling": "last"},
    "batch_tokens": 16000
  }
}
```

`batch_tokens` continues to mean `B*T`, so `16000` is one 16k-timestep
example per batch. `feature_dim` is the per-timestep channel width, not the
number of records.

For skewed record lengths, set `training.length_buckets` to compile and reuse
several shorter batch shapes. Use `training.batch_size` instead of
`training.batch_tokens` when every bucket must contain the same number of
records as a reference data loader. See
[Length-bucketed classification](config-training.md#length-bucketed-classification).

For a linear time-invariant FFT-convolution baseline, replace the canonical
Mamba block with:

```json
{"type":"s4d","state_size":64,"init":"s4d-lin"}
```

See [S4D](s4d.md) and
[`continuous_s4d_classification_tiny.json`](../examples/continuous_s4d_classification_tiny.json).
For reference-style normalization and output projection, use
`norm_type: "batchnorm"` with `output_transform: "glu"` as shown in
[`continuous_s4d_batchnorm_reference_tiny.json`](../examples/continuous_s4d_batchnorm_reference_tiny.json).
That mode requires fixed unpadded records; partial/padded classification
batches fail rather than contaminating running statistics.

The adapter uses the top-level `positional_embedding` policy. Set `"none"` for
raw signals whose sequence axis already carries order, or
`"learned_absolute"` for a learned table. There is no adapter-specific
position field.

Keep `input_adapter.norm: "none"` for low-dimensional raw signals when
magnitude carries information. The optional LayerNorm is applied after the
linear projection. At `feature_dim: 1`, its zero-bias initialization cancels
the input magnitude exactly and gives the backbone only the input sign. Mixlab
therefore warns about `norm: "layernorm"` for `feature_dim <= 4`. LayerNorm is
still available for inputs where per-timestep scale invariance is deliberate.

## Current boundary

Continuous v1 supports native `training.objective: "classification"` with
existing `mean` or `last` pooling. It deliberately rejects:

- language-model, masked, diffusion, and multihead objectives
- token-derived char, n-gram, smear, framing, and reverse-complement features
- token generation
- mixed token and continuous inputs in one example

Omitting `input_adapter` preserves the existing token graph and weight layout
exactly.

Fixed-shape native classifiers with S4D-only backbones are the exception to the
earlier HF boundary: `export-hf` writes a tokenizer-free custom-code
`AutoModelForSequenceClassification` directory accepting float
`input_values: [B,T,F]`. It preserves `linear_frames`, S4D, pooling, and trained
classifier weights. BatchNorm, padded records, mixed backbones, and stateful
generation remain native-only. See [Hugging Face export](hf-export.md#continuous-s4d-classifiers).
