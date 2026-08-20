# Discrete codebook input

Mixlab can classify synchronized tokens from residual-vector-quantized codecs
such as EnCodec, DAC, SpeechTokenizer, Mimi, and WavTokenizer. The public input
shape is integer `[B,T,Q]`: `T` timesteps and `Q` codebooks per timestep.

```text
int32 [B,T,Q] -> shared offset embedding -> codebook fusion -> [B,T,D] -> backbone
```

Codec encoding and decoding remain outside Mixlab. Prepare frozen codec output
offline as `.npy` `[N,T,Q]`, or `.npz` with `codebook_tokens` and optional
`lengths`. Code ID `0` is a normal embedding row; it is not padding.

## Prepare data

The label TSV must map every zero-based array row to one non-negative class.
An optional length TSV uses the same format and supplies each row's valid
timestep count. An `.npz` may instead contain an integer `lengths: [N]` array.

```bash
mixlab -mode prepare \
  -input codec_tokens.npy \
  -input-format codebooks \
  -codebook-vocab-size 1024 \
  -codebook-modality audio \
  -label-file labels.tsv \
  -length-file lengths.tsv \
  -prepare-output-dir data/audio_codes
```

`-codebook-vocab-size` is the exclusive upper bound for every code ID. Prepare
rejects negative or out-of-range IDs and reports the source row, timestep,
codebook, and offending value.

The output uses `mixlab_codebook_sequence_shard_v1`. Each record stores its
label, valid length, and contiguous int32 `[T,Q]` codes atomically. The adjacent
manifest records `T`, `Q`, codebook vocabulary size, modality, label count, and
split class counts. Model/manifest mismatches fail before trainer creation.

## Configure the adapter

```json
{
  "model_dim": 256,
  "seq_len": 128,
  "positional_embedding": "none",
  "input_adapter": {
    "kind": "discrete_codebooks",
    "num_codebooks": 2,
    "codebook_vocab_size": 1024,
    "fusion": "attention_mlp",
    "fusion_hidden_dim": 256,
    "norm": "none"
  },
  "blocks": [
    {"type": "gated_deltanet", "heads": 4},
    {"type": "swiglu"}
  ],
  "training": {
    "objective": "classification",
    "classification": {"num_labels": 10, "pooling": "mean", "bias": false},
    "batch_tokens": 1024
  }
}
```

`attention_mlp` is the default fusion. It matches the DASB reference: one
shared `[Q*V,D]` table indexed by `code + codebook*V`, then
`Linear(D,H) -> ReLU -> Linear(H,1,bias=false) -> softmax(Q)` and a weighted
sum over codebooks. `fusion: "mean"` is a parameter-free ablation.
`fusion_hidden_dim` defaults to `model_dim` and is valid only for
`attention_mlp`. `norm` accepts `none` or `layernorm`.

The embedding uses PyTorch `nn.Embedding` normal initialization. Attention
linears and the first-linear bias use PyTorch fan-in uniform initialization.
Set `training.classification.bias: false` for a DASB-style bias-free output
linear; classifier bias otherwise defaults to `true`.

### Linear probes

For a representation-only baseline, classification may use an empty block
stack:

```jsonc
{
  "final_norm": false,
  "input_adapter": {
    "kind": "discrete_codebooks",
    "num_codebooks": 2,
    "codebook_vocab_size": 1024,
    "fusion": "attention_mlp",
    "fusion_hidden_dim": 1024,
    "norm": "none"
  },
  "blocks": [],
  "training": {
    "objective": "classification",
    "classification": {"num_labels": 18, "pooling": "mean", "bias": false}
  }
}
```

This runs adapter, optional final norm, temporal pooling, and classifier only.
The shape above with `model_dim: 1024` has exactly `3,166,208` trainable
parameters. Set `pooling` explicitly for zero-block probes; the general
classification default is `last` when no bidirectional mixer establishes a
mean-pooling default. See
[`discrete_codebooks_linear_probe_tiny.json`](../examples/discrete_codebooks_linear_probe_tiny.json)
for a runnable small recipe.

For skewed valid lengths, set `training.length_buckets` to slice each batch to
the smallest configured width that fits its records. Set `training.batch_size`
instead of `training.batch_tokens` to hold the record count constant across
buckets. See
[Length-bucketed classification](config-training.md#length-bucketed-classification).

V1 supports native single-label classification only. Empty block stacks are
accepted only with `linear_frames` or `discrete_codebooks` classification;
language-model objectives and token-embedding classifiers still require at
least one block. Token feature channels,
LM objectives, framing, segment masks, reverse-complement features, generation,
and Hugging Face export are rejected explicitly. Valid lengths affect temporal
classification pooling; they do not mask ID `0` or the codebook fusion.

Reference parity is pinned by
`scripts/gen_discrete_codebooks_reference_fixture.py`, which records the exact
SpeechBrain DASB source commit and PyTorch runtime in its generated fixture.
Run it with `--check` to re-verify the checked-in values without rewriting
them, or `--output <path>` to regenerate. The fixture values are bit-identical
across interpreter versions, so `--check` compares them while ignoring the
recorded runtime comment.
