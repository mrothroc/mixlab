# Dense Grid Regression

Dense regression maps fixed-size multi-channel grids to continuous output grids.
It is a native task, separate from token models and sequence classification.
Start with [`grid_regression_tiny.json`](../examples/grid_regression_tiny.json).

## Configuration

Use `input_adapter: {"kind":"grid","channels":2,"height":8,"width":8}`,
`dense_regression: {"output":"network.prediction","target_channels":1}` and
`training: {"objective":"dense_regression","batch_size":2,"optimizer":"adamw"}`.
Do not set `model_dim`, `vocab_size`, `seq_len` or `batch_tokens`.
`batch_size` counts records, not pixels. `metric_scale` is optional, positive,
and defaults to 1; it scales reported RMSE only, never predictions or gradients.

V1 supports one named `custom` graph. Its `x` input is `[B,IH,IW,C]` (NHWC).
Declared weights may use `B`, `C`, `IH`, `IW` or literal integer shape strings.
The selected `dense_regression.output` must have shape
`[B,IH,IW,target_channels]`. There is no implicit embedding, normalization,
residual, final activation or output head.

Spatial custom operations:

| Op | Inputs | Parameters and semantics |
|---|---|---|
| `conv2d` | input, kernel, optional bias | Kernel `[Cout,K,K,Cin]`; integer `kernel`, `stride` (default 1), `padding` (default 0). |
| `conv_transpose2d` | input, kernel, optional bias | Kernel `[Cin,K,K,Cout]`; same parameters; `output_padding` must be 0. |
| `max_pool2d` | input | `kernel=stride`, padding 0; floor output sizes; first maximum receives gradient on ties. |
| `concat` | two inputs | `axis:3`; matching batch/height/width. |
| `relu`, `stop_gradient` | one input | No parameters. |
| `add`, `sub`, `mul` | two inputs | Equal shapes, no parameters. |
| `transpose` | one input | Complete `axes` permutation. |
| `slice` | one input | Explicit `start`, `end`, positive `step`, and `axis`; nonempty in-bounds range. |

Convolutions use native MLX operations, not expanded patch matrices. V1 uses
square kernels, groups=1 and dilation=1. Convolution output size is
`floor((input+2*padding-kernel)/stride)+1`; transposed output size is
`(input-1)*stride-2*padding+kernel`.

Custom weights accept `init: {"kind":...}`: `zero`, `one`, `normal`, `uniform`,
or `pytorch_conv_uniform`. Normal uses positive `scale` as standard deviation;
uniform uses `[-scale,scale]`. PyTorch convolution uniform uses
`1/sqrt(fan_in)` as its bound: `Cin*K*K` for conv, `Cout*K*K` for transposed
conv. A bias explicitly references its kernel with `init.weight`; this must
be the kernel in the same convolution operation. Omitted init keeps existing
custom initialization. Equal seeds ensure Mixlab repeatability, not identical
PyTorch random samples.

All declared weights retain stable `network.<weight>` logical names and
`w<index>_<logical-name>` checkpoint keys. Unreachable operations are pruned;
unreachable weights remain initialized/checkpointed but receive no gradients,
optimizer moments, or decay. Count mode reports total and selected-output
trainable parameters. Forward FLOPs are analytical; backward FLOPs are not
estimated for spatial graphs.

## Prepare

Provide normalized, finite float32/float16 NumPy arrays and an explicit source
JSON manifest. Paths are relative to that manifest:

```json
{
  "dtype": "float32",
  "records_per_shard": 64,
  "splits": {
    "train": {"inputs":"train_x.npy","targets":"train_y.npy","masks":"train_mask.npy"},
    "val": {"inputs":"val_x.npy","targets":"val_y.npy","masks":"val_mask.npy"}
  }
}
```

Inputs are `[N,C,H,W]`, targets `[N,Ct,H,W]`, masks `[N,1,H,W]` containing
exactly 0 or 1. Optional `ids` names a JSON string array with one unique,
nonempty ID per record. Without it, IDs are `<split>_<index>`. Input-only
prediction splits omit both targets and masks; use a separate input-only
manifest because all splits in one dataset share geometry.

```bash
mixlab -mode prepare -input-format grid -input source.json \
  -prepare-output-dir data/grids
mixlab -mode arch -config examples/grid_regression_tiny.json \
  -train data/grids/mixlab.dataset.json -checkpoint-dir runs/grid \
  -safetensors runs/grid/final.safetensors
mixlab -mode eval -config examples/grid_regression_tiny.json \
  -train data/grids/mixlab.dataset.json \
  -safetensors-load runs/grid/final.safetensors
```

Grid prepare never random-splits records or fits normalization on validation.
Normalize outside the encoder using training-only statistics. Float32 storage
preserves all finite bit patterns, including signed zero. Opt-in float16 stores
the float32-to-float16 rounded value, not the original exact value; overflow
is rejected. Review quantization error before using it for a large dataset.
The reader keeps a record index, one open shard, one record scratch buffer and
reusable batch storage; it does not cache whole shards. Training shuffles all
record indices each epoch and includes the last partial batch.

Optional source `normalization` metadata has `fit_split:"train"`,
`input_offset`, `input_scale`, `target_offset`, and `target_scale` arrays (one
value per channel; scales positive). It records statistics already applied by
the caller, never transforms the arrays. One shared object applies to every
split; validation-fitted statistics are rejected. Each split may also name a
`groups` JSON string-array file. Both are preserved in manifest
`grid_provenance` for leakage auditing; Mixlab does not infer a split from groups.

## Loss, Validation And Limits

Loss is pooled masked MSE: `sum(mask*(prediction-target)^2)/sum(mask)`, with
the mask broadcast to output channels. Padded batch rows contribute zero.
An empty training mask produces zero loss and zero current gradients, but
still executes an optimizer step. Existing moments/decay can therefore move
weights. A wholly empty validation mask is an error, not a perfect score.

Validation visits the complete `val` split. It accumulates masked and unmasked
SSE/count in float64, excluding padded examples, before computing RMSE.
`best.safetensors` in `-checkpoint-dir` minimizes masked validation RMSE.
Early stopping and `target_val_loss` use masked RMSE for this task.
They are checked on initial validation as well: a model that already meets a
stopping condition is saved without applying an optimizer update.
Logs show records/s and pixels/s, excluding first-step compilation from the
compute rate. The telemetry JSON includes these rates in `extra`.

AdamW and LAMB are supported. Rank-4 kernels use matrix LR/decay overrides;
vector biases use scalar settings. For Adam equivalence choose AdamW, zero
global/group decay, beta1 0.9, beta2 0.999, epsilon 1e-8.
Use FP32 compute: the existing BF16 custom-block restriction still applies.
DDP, accumulation, SWA, token objectives, sequence schedules, built-in sequence
blocks, HF export and token generation are not supported for grid tasks.

## Augmentation And Staged Training

`training.grid_augmentation: {"dihedral":true}` chooses uniformly among eight
joint D4 transforms per record occurrence. All input channels, targets, and
validity masks receive the same transform. Geometry must be square. Indices
0..3 rotate counterclockwise by 0/90/180/270 degrees; indices 4..7 mirror columns
first, then rotate. Selection is keyed by seed, epoch, occurrence, and record ID,
independent of shuffle RNG or worker timing. Omission or `dihedral:false` leaves
batches unchanged. Validation and prediction never augment. Scratch memory is
one record's largest NHWC tensor, reused between records.

Use `training.init_from` or `-safetensors-load` for weights-only warm starts,
never both. Config-relative paths resolve beside the config file; CLI-relative
paths resolve from the working directory. New grid checkpoints record a
`logical_weights` mapping from stable names such as `network.U1.weight` to
physical safetensor keys. Named loading accepts reordered inventories only
through that explicit mapping. Older files without it require exact physical
index/name matching; no heuristic prefix stripping occurs.

The default load is exact. `training.init_allow_missing: ["network.W.*"]`
explicitly permits new target weights for a model extension, initialized from
the current config/seed. This requires a metadata-bearing grid checkpoint.
Unexpected tensors, incomplete/ambiguous metadata, non-finite loaded values,
and all shape mismatches fail, including for allow-missing names. Load reports
list loaded/new names; errors identify missing/unexpected names.

`training.freeze: ["network.U1.*"]` freezes matching logical weights. Patterns
use Go `path.Match`: `*`, `?`, character classes (`[a-z]`, `[^a-z]`), and backslash
escapes; patterns match the entire name. Dots are ordinary characters, `/` is
a separator. Every pattern must match, overlaps are deduplicated, and freezing
all weights reachable from the selected output fails. The resolved plan drives
autograd, optimizer state/update, clipping, reporting, and resume checks. Frozen
weights stay unchanged even with decay enabled and remain in checkpoints.

The tiny [stage-one](../examples/grid_two_stage_1.json) and
[stage-two](../examples/grid_two_stage_2.json) examples share a complete weight
inventory. Stage one selects `network.output1`, pruning the refinement graph
and retaining its initialized weights. Stage two selects `network.output2`,
loads all stage-one weights, freezes U1, and concatenates its detached output
with raw input for refinement. These are workflow examples, not full reference
architectures or benchmark recipes.

```bash
mixlab -mode arch -config examples/grid_two_stage_1.json \
  -train data/grids/mixlab.dataset.json -safetensors stage1.safetensors
mixlab -mode arch -config examples/grid_two_stage_2.json \
  -train data/grids/mixlab.dataset.json -safetensors-load stage1.safetensors \
  -checkpoint-dir checkpoints/stage2 -checkpoint-every 10
```

## Resume

Periodic checkpoints now use the common `mixlab_resume_v1` bundle: a
`step_XXXXXX.st` model, optimizer-state safetensors, and a `.resume.json`
manifest published last. Keep all companion files together. Resume restores
the optimizer counters/moments, original LR schedule, record order/epoch/cursor,
augmentation replay inputs, resolved trainable set, best masked RMSE, and
early-stop state. Partial final batches are replayed exactly.

```bash
mixlab -mode arch -config examples/grid_two_stage_2.json \
  -train data/grids/mixlab.dataset.json -resume checkpoints/stage2 \
  -checkpoint-dir checkpoints/stage2 -checkpoint-every 10
```

Remove `init_from` and `init_allow_missing` from a warm-start config before
resuming; those one-time settings are excluded from resume compatibility hashes.
Do not pass `-safetensors-load`. Config, precision, geometry, freeze policy,
validation cadence, and dataset identity must match. Training/validation shard
contents and the manifest are hashed once per run with checkpointing/resume;
keep them immutable during training. Paths are part of dataset identity, so
moving the dataset is not a supported exact-resume operation. Restoring the
shuffle RNG replays epoch permutations, not data reads or optimizer steps.

Raising `training.steps` continues the saved schedule and then holds its original
terminal LR; it does not restart or stretch the schedule. A saved early-stop
decision remains stopped. Resume does not add an extra initial validation or
reset best-checkpoint selection. Reuse the checkpoint directory to retain the
earlier best-model artifact. A stage/output/freeze transition requires a warm
start, not resume. Old weights-only grid checkpoints remain valid warm starts
but cannot provide optimizer/loader state for resume.

The best-model file stores its selection score and config/dataset identity in
the same atomic safetensors write as its weights. A resume from an older periodic
checkpoint cannot replace that artifact with a worse replayed result. Its metric
and early-stop state still replay from the periodic checkpoint. An existing best
file without matching selection metadata is rejected on resume; use a fresh
checkpoint directory rather than silently overwriting an unrelated artifact.

## Prediction

```bash
mixlab -mode predict-grid -config model.json -safetensors-load weights.safetensors \
  -grid-in data/predict/mixlab.dataset.json -grid-split predict -grid-out predictions
```

`-grid-split` defaults to `predict`. `-grid-out` must not exist. A successful
run atomically publishes one float32 `[Ct,H,W]` `.npy` per record and
`predictions.json` mapping original IDs to numeric filenames. IDs cannot escape
the output directory. Failed runs clean their staging directory; no existing
predictions are overwritten. Prediction builds only the selected graph, with
no targets, loss or optimizer state. Output is in model units; apply any
dataset-specific inverse normalization (offset and scale) outside Mixlab.
Prediction and standalone evaluation use the same strict logical-name checkpoint
loading as training: declaration order may differ, but missing, unexpected,
shape-mismatched or non-finite tensors are rejected, including unused weights.
Allow-missing warm-start patterns never apply to inference.

## Binary Format

`mixlab_grid_shard_v1` uses little endian. The first 1024 bytes are 256 uint32s:
magic `20260930`, version `1`, dtype (`1` float32, `2` float16), C, H, W, Ct,
record count, ID JSON byte length, then zero reserved words. Next is a UTF-8
JSON array of record IDs. Each fixed-stride record contains CHW input floats,
CHW target floats, and `ceil(H*W/8)` mask bytes, low bit first; unused high
bits must be zero. Ct=0 omits targets/mask. Exact file length, finite payloads,
dimensions, IDs and manifest geometry are validated. ID metadata is capped at
16 MiB, a shard at one million records, and one record at 512 MiB. Combined
CPU input/target/mask batch buffers are capped at 1 GiB. The usual MLX memory
and cache limits also apply to training and prediction.

## PyTorch State Export

`export-torch-state` is a CPU-only, grid-only weight/layout exporter. It does not
export Python model code or require MLX. Existing grid HF rejection is unchanged.

```bash
mixlab -mode export-torch-state -config model.json -safetensors-load weights.st \
  -export-map map.json -export-dir exported-grid
```

`-export-map` is strict JSON, with exactly one disposition for every declared
logical weight, including frozen/unreachable ones. Example for a 1x2x3x3 PyTorch
convolution kernel and scalar bias:

```json
{
  "format": "mixlab.torch_state_map.v1",
  "mappings": [
    {"source":"network.conv.weight","target":"conv.weight",
     "transform":"transpose","axes":[0,3,1,2],"shape":[1,2,3,3]},
    {"source":"network.conv.bias","target":"conv.bias",
     "transform":"identity","shape":[1]}
  ]
}
```

Only `identity` and `transpose` are accepted; `axes` follows NumPy permutation
semantics, and `shape` is the required destination shape. Conv2d OHWI -> OIHW
and ConvTranspose2d IHWO -> IOHW both use `[0,3,1,2]`. An optional `excluded`
array contains `{ "source":"logical.name", "reason":"explanation" }`.
Exclusions are explicit, recorded, and inappropriate for faithful full-model
reference loading. Duplicate/unknown sources or destinations, unmapped weights,
invalid axes/shapes and executable transforms are errors. Optional `provenance`
is a string-to-string attribution map, never executable configuration.

The output directory must be new. It is published atomically only after writing
standard contiguous `model.safetensors`, `export.json` (selection, layout,
source-file hashes and mapping), `export-map.json`, a native `config.json`, and
`infer.py`. One-time initialization paths are removed from the packaged config.
Keep source config/map/checkpoint files immutable while exporting.

The helper requires NumPy, PyTorch and safetensors. It calls
`safetensors.torch.load_file` and `model.load_state_dict(..., strict=True)`.
The caller explicitly supplies trusted Python code and a model factory:

```bash
python exported-grid/infer.py --model-file trusted_model.py --factory build_model \
  --kwargs-json '{"channels":2}' --native-output network.prediction \
  --input normalized_inputs.npy --output predictions.npy --batch-size 2
```

Use the exact `selected_output` from `export.json` for `--native-output`. For a
tuple/list model result, also supply the corresponding `--output-index` (0-based).
This explicitly acknowledges the native stage and selects the reference output;
the helper cannot infer semantic correspondence between arbitrary model APIs.
For the reference RadioWNet factory, use `--factory RadioWNet`, kwargs
`{"inputs":2,"phase":"secondU"}`, and `--output-index 1` for refinement.
Provide the reference's dependencies or your own trusted factory wrapper.
The export map never imports model code. Do not run untrusted Python modules.

Input/output arrays are NCHW. Inputs must already be normalized, finite, floating
arrays matching the fixed geometry. Batch inference uses memory-mapped arrays,
does not shuffle/augment, and never overwrites an existing output. Outputs default
to model units. To invert normalization `z=(y-offset)/scale`, use
`--target-offset ... --target-scale ...` for `y=z*scale+offset`, with one value or
one per output channel. `metric_scale` alone is **not** an inverse transform.
No activation/clamping is added. Pickle `.pt` is deliberately not emitted; an
explicit offline conversion is `torch.save(load_file("model.safetensors"), "model.pt")`.

See the [two-/three-channel staged reference recipes](../examples/grid_unet_reference/README.md)
for architecture attribution, maps, licenses, and deliberate training differences.
General segmentation losses, variable-resolution/tiling, distributed grid training
and HF model export remain outside this workflow.

## Numerical Verification

`scripts/generate_grid_reference.py` regenerates the small PyTorch operator
fixtures. Real MLX tests cover unequal channel counts, transposed layouts,
pool ties and gradients. `scripts/generate_grid_model_reference.py` derives
the full reference graph directly from pinned source and emits large fixtures
only into a caller-selected temporary directory. Run the gated full-resolution
test with `GRID_REFERENCE_DIR=<dir> go test -tags mlx ./train
-run TestGridPinnedModelForwardAndBenchmark -count=1 -v`. These are numerical
and performance checks, not a claim of reproducing any dataset's final score.

Generate a separate fixture with `--phase secondU` for refinement training, then
run `GRID_REFINEMENT_REFERENCE_DIR=<dir> go test -tags mlx ./train
-run TestGridPinnedRefinementFreeze -count=1 -v`. This checks the actual reference
forward plus a native optimizer step with nonzero decay: U1 must stay identical,
have no optimizer moments, and refinement weights must update. Routine tiny
tests additionally compare interrupted/resumed AdamW and LAMB training exactly,
including augmentation, partial batches, weights, moments, and schedule state.

The small architecture/count fixtures in `arch/testdata/grid_reference` derive
from the pinned RadioWNet implementation and retain both MIT license notices.
Generate `grid-source.json`, prepare it with `-input-format grid`, then use
`GRID_BENCH_MANIFEST=<manifest> go test ./data -run '^$' -bench GridShardIO
-benchmem` to measure shard reads/layout conversion separately from GPU work.
Repeated reads normally use the OS page cache; this is not cold-disk throughput.

Full export parity: generate both phases for channels 2 and 3 with `--samples 4`
into `<root>/<channels>/<phase>`. With PyTorch/safetensors on the Python path, run
`GRID_EXPORT_REFERENCE_DIR=<root> GRID_EXPORT_REFERENCE_SOURCE=<pinned-modules.py>
go test -tags mlx ./train -run TestGridTorchExportReferenceParity -count=1 -v`.
It verifies strict real-reference loading for both outputs, before and after
training updates, and writes machine-readable `*-export-report.json` files beside
the temporary fixtures. The stage-two warm start uses trained stage-one tensors.
Routine tests cover every intermediate shape, mapping errors, both layouts,
normalization offsets, bounded batch inference and no-overwrite publication.
