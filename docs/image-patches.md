# Image Patch Input

`linear_patches` consumes prepared float32 patch records. It adds fixed image
geometry, optional learned 2-D coordinates, and training-loader crop/flip to
the existing continuous projection. Image decoding, patch extraction, and
normalization remain external; no pretrained encoder is needed.

## Configuration

```json
"input_adapter": {
  "kind": "linear_patches",
  "image": {"height": 32, "width": 32, "channels": 3},
  "patch": 4,
  "bias": true,
  "norm": "none",
  "coords": "learned_xy",
  "augment": {"hflip": true, "random_crop_pad": 4}
}
```

Input arrays are `[N,T,F]`: patches in raster order, pixels within each patch
in row-major order with channels last. `T=(height/patch)*(width/patch)` and
`F=patch*patch*channels`. Both image dimensions must be divisible by the positive
square patch size. Set `seq_len=T`; `feature_dim` is derived, or must agree if
provided. Every record must contain the whole image. Short records, padding
inside images, and shape-changing length buckets/schedules are rejected.

`coords` defaults to `none`, a geometry-checked `linear_frames` adapter with
identical model weight layout and computation. `learned_xy` adds
`input_adapter_coord_x: [width/patch,D]` and
`input_adapter_coord_y: [height/patch,D]`, after projection and optional adapter
LayerNorm. Its parameter delta is `(width/patch + height/patch)*D`.

Coordinates apply only to patches. Optional CLS is then prepended, followed by
the model's positional embeddings and embedding dropout. Set top-level
`positional_embedding: "none"` to use XY alone. Other positional modes remain
explicitly additive; learned absolute positions must include CLS when enabled.
Final normalization follows the ordinary model configuration.

Native classification supports the same mixer choices as continuous frames.
Mean/last pooling can compare attention and recurrent mixers; CLS still requires
its supported bidirectional attention stack. See the
[CLS image example](../examples/linear_patches_classifier.json).

## Preparation

```bash
mixlab -mode prepare -input image_patches.npy -input-format continuous \
  -label-file labels.tsv -continuous-modality image \
  -config examples/linear_patches_classifier.json \
  -prepare-output-dir data/images
```

Supplying a `linear_patches` config checks grid length, feature width, and
full-record lengths before writing shards. Existing prepare calls without
`-config` remain valid. Training/evaluation independently validate geometry.
Existing continuous shards need no conversion if their layout matches. Shape
checks cannot detect incorrectly ordered pixels; use the convention above.

## Augmentation

Only the training loader applies `augment`, in this order:

1. `random_crop_pad: k` chooses independent uniform crop offsets in `[0,2k]`
   after conceptual padding on every edge; output size remains `H*W`.
2. `hflip: true` flips horizontally with probability 0.5, including both patch
   column order and pixel column order inside each patch.

`random_crop_pad` defaults to zero, `hflip` to false. Padding defaults to zero
in stored feature space. For arrays normalized as `(pixel-mean)/std`, use
`augment.pad_value: [-mean_r/std_r, -mean_g/std_g, -mean_b/std_b]` to represent
black padding applied before normalization. Supply finite numeric values, one
per channel. This equivalence assumes fixed per-channel affine normalization.

Transforms use an independent RNG keyed by seed and record occurrence. They
do not change shuffle order; repeated occurrences get fresh random choices,
and resumable checkpoints reproduce them through loader replay. Cached shards
are never mutated. One scratch record is reused without per-record allocation.
Validation/evaluation never augment, including when augmentation is configured.
The final last-training-batch diagnostic can still reflect its augmented input.

Mixup, CutMix, policy-based augmentation, nonsquare patches, and video are out
of scope. No multi-epoch copies of image shards are necessary for crop/flip.

## Export

Supported classifiers export tokenizer-free to
`AutoModelForSequenceClassification`, accepting already-patchified
`input_values: [B,T,F]`. Export retains projection, coordinates, positions,
CLS if used, and classifier weights. It does not augment, decode, normalize, or
patchify raw images. Existing HF backbone restrictions still apply. Native
checkpoints include XY weights in deterministic order; enabling XY is not a
checkpoint-compatible toggle on an already-trained `coords: none` model.
