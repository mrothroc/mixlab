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

### Per-record vectors

Set `input_adapter.vector_dim: S` to add a float32 vector per record. Omitted or
`0` keeps the existing map-only contract; positive widths are grid-only. The
custom graph alias `v` has shape `[B,S]` and binds to runtime input
`grid_vectors`. An unused `v` is pruned from the executable graph, but the
configured width must still match the dataset. Counting and weight export need
no dataset. No weights are added just by declaring a vector input.

Vectors are caller-defined features, not tokens or image channels. Mixlab does
not normalize them; undeclared columns are never augmented. For spatial points,
declare `input_adapter.vector_points: [[row_column,col_column], ...]` so D4 moves
them with the map (see [augmentation](#augmentation-and-staged-training)). Generic
non-geometric vectors may accompany D4 without such a declaration. Directions
and displacement vectors are not supported by this point-coordinate contract.
[`grid_vectors_tiny.json`](../examples/grid_vectors_tiny.json) demonstrates a
one-pixel map plus a per-record scalar, using reshape, addition and convolution.

### Muon for convolution kernels

Dense regression accepts `adamw`, `lamb`, `muon`, `muon_eq_r`, and `normuon`.
AdamW/LAMB routing is unchanged. Muon-family optimizers use `matrix_lr` and
`matrix_weight_decay` for convolution kernels, orthogonalizing an optimizer-only
`[out_channels, kernel_height * kernel_width * in_channels]` view. Ordinary
`[O,H,W,I]` kernels flatten directly; transposed `[I,H,W,O]` kernels first permute
to `[O,H,W,I]`. Model storage, prediction, export, and checkpoint tensor names do
not change. The same view controls shape scaling and variant normalization.

`training.muon_exclude: ["stem", "output"]` matches exact convolution **node
output names**, not weight names or wildcard patterns. Those kernels use AdamW
in `extra:grid_muon_excluded` with `scalar_lr` and `matrix_weight_decay`, still
respecting decay eligibility. Biases and GroupNorm affine parameters keep their
scalar AdamW policy. There are no automatic first/last-layer exclusions.
Unknown or duplicate names, non-convolution nodes, and conflicting exclusions or
layouts for a shared kernel are errors. All consumers of a shared kernel must
agree. Muon kernels must be direct declared weights, not computed activations.

Inspect routing before training:

```bash
mixlab -mode optimizer-report -config examples/grid_muon_tiny.json
```

Each active Muon tensor reports `muon_matrix_shape`; grid kernels also report
`kernel_layout` (`ohwi` or `ihwo`). Frozen weights have no optimizer assignment or
state. `muon_momentum`, `muon_backend_steps`, `muon_nesterov`, and
`newton_schulz_variant` retain their existing semantics. Resume restores momentum
and variant state and rejects an incompatible optimizer view/configuration.
[`grid_muon_tiny.json`](../examples/grid_muon_tiny.json) demonstrates a small
encoder/decoder with excluded input/output kernels. These are experimental
optimizer choices, not an accuracy or speed guarantee.

Spatial custom operations:

| Op | Inputs | Parameters and semantics |
|---|---|---|
| `conv2d` | input, kernel, optional bias | Kernel `[Cout,Kh,Kw,Cin]`; `kernel`, `stride` (default 1), `padding` (default 0), `dilation` (default 1) each accept an integer or `[H,W]`. Kernel/stride/dilation must be positive, padding nonnegative. |
| `conv_transpose2d` | input, kernel, optional bias | Square kernel `[Cin,K,K,Cout]`; integer `kernel`, `stride`, `padding`; `output_padding` must be 0. No dilation or pairs. |
| `max_pool2d` | input | `kernel=stride`, padding 0; floor output sizes; first maximum receives gradient on ties. |
| `pad` | one rank-4 NHWC input | Required `axis1:[before,after]`, `axis2:[before,after]` and `mode:"zero"`, `"edge"` or `"circular"`; no batch/channel padding. |
| `group_norm` | input, scale, bias | NHWC activation, affine vectors `[C]`; required positive integer `groups` dividing C, optional finite positive `eps` (default `1e-5`). |
| `concat` | two inputs | `axis:3`; matching batch/height/width. |
| `relu`, `stop_gradient` | one input | No parameters. |
| `add`, `sub`, `mul` | two inputs | Equal shapes, no parameters. |
| `transpose` | one input | Complete `axes` permutation. |
| `reshape` | one input | `shape` array of positive integer strings, rank 2..4; leading `"B"` is allowed. Preserves leading batch size and total element count; no inferred `-1` dimension. |
| `slice` | one input | Explicit `start`, `end`, positive `step`, and `axis`; nonempty in-bounds range. |

Convolutions use native MLX operations; groups remain 1. An integer applies equally
to both spatial axes. A pair is ordered height, width; legacy scalar configs and
weight layouts are unchanged. `conv2d` output size on each axis is
`floor((input+2*padding-dilation*(kernel-1)-1)/stride)+1`; transposed output size is
`(input-1)*stride-2*padding+kernel`.
`dilation` is not accepted on `conv_transpose2d` or `max_pool2d`.
Increasing dilation does not change weights, initialization fan-in or MACs per
output pixel; it changes the sampled locations and potentially output size.
For a stride-1 odd kernel, `padding=dilation*(kernel-1)/2` preserves spatial size.
For example, `kernel:[1,3], dilation:[1,8], padding:[0,8]` preserves both axes.
Initialization fan-in is `Cin*Kh*Kw`, and Muon views the OHWI kernel as
`[Cout,Kh*Kw*Cin]`, including rectangular kernels.

`pad` amounts must be nonnegative integers; both pairs and the mode are required.
`zero` fills with zeros, `edge` repeats the border, and `circular` wraps the axis.
Each circular amount must be no larger than that original input axis. Zero amounts
are valid in every mode. Gradients from repeated edge/wrap indices accumulate
into the original input; padding introduces no weights. Use explicit `pad` before
`conv2d` for asymmetric before/after padding; convolution's `padding:[H,W]` remains
symmetric on each axis.

[`grid_rectangular_tiny.json`](../examples/grid_rectangular_tiny.json) uses a
1x9 convolution with circular width padding followed by a 9x1 convolution with
edge height padding, on two-channel 8x8 inputs.

[`grid_aspp_tiny.json`](../examples/grid_aspp_tiny.json) combines four parallel
3x3 branches at dilation 1/6/12/18 on a 32x32 grid, channel concatenation and a
1x1 output projection. Train it with a two-channel 32x32 grid manifest using the
same prepare/train workflow below. It is a runnable operator example, not a
benchmark reproduction.

### Polar branches

[`grid_polar_tiny.json`](../examples/grid_polar_tiny.json) resamples fixed inputs,
applies radial and circular-angular convolutions, maps features back to the full
Cartesian resolution, and fuses them with the original map. Declare
`input_adapter.vector_dim` and supply vectors in every dataset split.

| Op | Inputs | Required parameters | Output |
|---|---|---|---|
| `cartesian_to_polar` | map, `v` | `n_rays`, `n_r`, `dr`, `tx:[row_column,col_column]` | `[B,K,J,C]` |
| `polar_to_cartesian` | polar features, `v` | `n_rays`, `dr`, `height`, `width`, `tx` | `[B,H,W,C]` |

`tx` identifies two distinct columns in the unnormalized per-record vector.
Coordinates must be finite, in-map, and are measured at integer pixel centres.
Native train/validation/prediction errors name the record; raw GPU calls name the
batch row. Geometry vectors must be direct external inputs, not learned values.
Dihedral augmentation is allowed only when every polar op's ordered `tx` pair
is declared in `input_adapter.vector_points`. Otherwise graph validation names
the offending op and pair. This applies to both sampling and back-mapping.
Mixlab does not require `n_rays` to be divisible by four: augmentation is valid
independently of whether the chosen ray set is D4-equivariant.

Ray zero points along positive column; angles increase toward positive row.
Ray `k` has angle `2*pi*k/K`; sample `j` has radius `j*dr`. Cartesian sampling
is bilinear with border clamping. Back-mapping is inverse bilinear sampling,
with angular wraparound and `atan2(0,0)=0`, not splatting. `height`/`width` must
equal the full configured map size; downsample after back-mapping if needed.
The inverse requires `J >= ceil(hypot(H-1,W-1)/dr)+2`, including for fractional
transmitters. `dr` must remain finite/positive when represented as float32.

Cartesian-to-polar is not differentiable. Every declared weight in its map
ancestry is treated as potentially trainable, including frozen weights and
weights reached through another branch. Such graphs are rejected unless that
dependency is explicitly cut with `stop_gradient`. Input-only transforms are
allowed. An explicit stop intentionally prevents learning through that path;
it is not a way to train a Cartesian encoder through the sampler.
Polar-to-Cartesian differentiates into its polar input; neither op differentiates
into `v`. Neither op adds weights or changes checkpoint naming.

The implementation computes coordinates on-device for each batch and uses MLX
gathers/autodiff; there is no per-record coordinate cache. FLOPs include seven
interpolation operations per output value, not coordinate trig/index/scatter
costs. Large polar grids may consume much more memory than the Cartesian input;
the tiny example is not a full-resolution throughput or capacity claim.

PyTorch export remains weights plus a caller-supplied model. `export.json`
includes `polar_geometry` validation metadata; `infer.py --vectors ...` validates
coordinates before invoking the model. The reference test twin uses
`grid_sample(align_corners=True)` with border clamping, and a duplicated first
angular row for inverse wraparound. The forward ops and their parameters remain
in `config.json`; no Python model is generated.

Run `go test -tags mlx ./gpu ./train -run GridPolar -count=1` with NumPy, PyTorch
and safetensors available for train/resume/predict/export checks. Run
`bash scripts/test_grid_polar_native.sh` for the full independent double-precision
coordinate/forward/shared-plan adjoint matrix on the active GPU. The latter
separates coordinate error from adjoint accumulation error and reports, rather
than gates on, end-to-end white-noise error.

#### Measuring capacity and throughput

CUDA images tile FP32 convolution weight-gradient patch matrices larger than
2 GiB along the batch axis, targeting 256 MiB per tile (at least one record). This is an internal
operator reduction: the training batch, optimizer update and model stay unchanged.
Polar programs also default to CUDA graph limits of 20 ops, 100 MiB and 64 cached
graphs. Explicit `MLX_MAX_OPS_PER_BUFFER`, `MLX_MAX_MB_PER_BUFFER` and
`MLX_CUDA_GRAPH_CACHE_SIZE` overrides still win, but can increase memory pressure.
These are not hard VRAM caps. The dependency patch requires rebuilding the CUDA
base and downstream images; see [image rebuilds](../docker/README.md#why-three-layers).
Metal keeps its existing runtime. Neither backend caches record-dependent geometry.

Run each configuration in a separate process so allocator peaks are not carried
over from a previous model. The opt-in synthetic training benchmark runs five
warmup and twenty measured updates with AdamW, masked MSE and explicit strict
FP32. It does not reduce the configured batch size to fit memory:

```bash
MIXLAB_GRID_BENCH_CONFIG=/path/to/config.json \
  go test -tags mlx ./train -run '^TestGridTrainingBenchmark$' -count=1 -v
```

Set `MIXLAB_GRID_BENCH_BASELINE=/path/to/baseline.json` to match all equal-name,
equal-shape initial tensors across architecture arms, including biases. A widened
kernel retains its own fan-in-correct initialization. Synthetic inputs, targets and masks are
deterministic; coordinates vary each step and have no record-plan cache.
Each measured interval includes batch submission, forward, backward, optimizer
update and synchronized loss collection. It excludes host fixture construction.
The `GRID_BENCHMARK` JSON line reports samples, median, config hashes and MLX
memory. Peak memory includes initialization and warmup but excludes host buffers.
An allocation failure is a capacity failure, not a throughput result.
Set `MIXLAB_GRID_BENCH_STEPS=100` for a sustained-capacity check (total updates,
including the five warmups; accepted range 6-1000). `GRID_MEMORY` JSON lines
record initialization, trainer setup and each completed update, outside the
timed interval. They include MLX counters and, when available, CUDA pool/graph
reservations and device free memory. The final report records the effective
graph-policy environment. For total process GPU memory, sample the benchmark
PID separately with `nvidia-smi`; device free memory also reflects other users.
Record the measured peak separately from the configured allocator limit: on
Metal, that setting guides reclamation and scheduling rather than imposing a
hard process-memory cap, so the peak can exceed it.
MLX counters also need not include allocations owned by CUDA graphs or backend
libraries; they are not a measurement of total device-process memory.

For isolated native back-map measurements:

```bash
bash scripts/test_grid_polar_native.sh --benchmark 512
bash scripts/test_grid_polar_native.sh --benchmark 1024
```

These use B=16, H=W=256, C=32, J=363 and an 8 GiB MLX limit with a 256 MiB
allocator cache. Each forward or gradient-only VJP includes fresh coordinate
construction and synchronization. The VJP is measured directly, not estimated
by subtracting forward time; it is not the full model's backward time. Peaks are
reset for each phase and include its warmups and live inputs. Use
`--benchmark --smoke` for a small harness check, not a capacity claim. Record
device, runtime revision, precision and memory limits alongside results before
comparing machines. The CNN-only baseline has no back-map measurement.

### GroupNorm

[`grid_group_norm_tiny.json`](../examples/grid_group_norm_tiny.json) uses
convolution, GroupNorm, ReLU and an output projection, with batch size one.
GroupNorm normalizes height, width and the channels within each group separately
for each sample. It uses biased variance (`correction=0`) and epsilon inside the
square root, matching [PyTorch GroupNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html).
It is not channel-only normalization. Training and evaluation use the same
statistics; there are no running buffers or dependence on other batch records.

Declare affine weights explicitly with `init: {"kind":"one"}` for scale and
`init: {"kind":"zero"}` for bias. No normalization or weights are inserted
automatically. For example:

```json
{"op":"group_norm", "inputs":["hidden","norm.weight","norm.bias"],
 "output":"normalized", "params":{"groups":2,"eps":0.00001}}
```

The implementation composes native MLX reductions and affine operations with
autodiff on Metal/CUDA. Forward FLOPs are an analytical estimate (nine per
element plus two per sample/group), not a measured kernel instruction count.
State-dictionary export maps scale and bias unchanged with `identity` transforms.
The consuming PyTorch model must instantiate `nn.GroupNorm(groups, C, eps=eps)`;
export does not generate that model. Only convolution kernels need layout
transposes. Regenerate forward/all-gradient fixtures with
`python3 scripts/generate_grid_group_norm_reference.py`; run real training/export
checks with `go test -tags mlx ./gpu ./train -run GridGroupNorm -count=1`.
Random fixtures use float32 PyTorch; constant/near-constant fixtures use float64
PyTorch on float32-representable inputs and affine weights as a higher-precision
oracle. This avoids testing against cancellation in the reference's float32
fused affine kernel. All comparisons retain the same spatial tolerances.

Native convolution does not guarantee small temporary allocations: MLX may
materialize patch tensors during backward. Measure peak training memory and
steady-state throughput for the actual dilation/stride/shape/backend, rather
than inferring them from parameter count. See [memory diagnostics](performance.md).

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

For vector-enabled models, add `"vectors":"train_v.npy"` (and corresponding
files for every other split). Each vector array must be finite float32 `[N,S]`,
with the same N/order as that split's maps and the same positive S across all
splits. Float16 vector arrays are rejected even with `"dtype":"float16"` map
storage. Prepare records `grid.vector_dim` and, per split,
`grid_provenance.vectors` entries containing the source basename, SHA-256 file
hash and width. Input-only prediction datasets also require their vector file.

Vectors travel with record IDs through shuffle, prefetch, validation and resume.
Partial batches duplicate the first real vector into unused slots while keeping
their loss masks zero; padded records are never included in output or metrics.
This avoids inventing zero-valued coordinates for later coordinate consumers.
Resume identity covers both the manifest and all train/validation shard bytes,
so changing only vectors invalidates replay just as changing maps does.

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
The reader keeps a record index, bounded open shards and decode scratch, and
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
Use `compute_dtype: "float32"`: the existing BF16 custom-block restriction still
applies. On a GPU, float32 storage defaults to allowing reduced-precision
arithmetic in eligible kernels; it is not a promise of strict FP32 multiplication.
DDP, accumulation, SWA, token objectives, sequence schedules, built-in sequence
blocks, HF export and token generation are not supported for grid tasks.

### GPU Float32 Precision

MLX gates reduced-precision float32 arithmetic with one switch on both GPU
backends. On CUDA it permits TF32 in eligible matmuls and convolutions. On Apple
GPUs with neural accelerators (M5-class GPUs on macOS 26.2 or later) it routes
float32 matmuls and attention through the accelerators; earlier Apple GPUs have no
reduced-precision path, so the setting changes nothing there.

Use `training.allow_tf32: false` for strict-FP32 parity experiments. Omitted
preserves MLX's speed default, but honors an existing `MLX_ENABLE_TF32=0` and, on
CUDA, `NVIDIA_TF32_OVERRIDE=0`. Explicit config must agree with `MLX_ENABLE_TF32`;
on CUDA, explicit true conflicts with `NVIDIA_TF32_OVERRIDE=0`.
`NVIDIA_TF32_OVERRIDE=1` does not force TF32 when Mixlab disables it, and Metal
ignores it. Environment values must be `0` or `1`. Without a GPU backend, explicit
values fail at execution, not config parsing.

On CUDA, a strict-FP32 convolution whose cuDNN execution plans are unavailable
uses MLX's GEMM convolution fallback, including input and weight gradients. It
does **not** enable TF32. A `[mlx conv] fallback=gemm` stderr message reports
the backend operation, input/weight/output shapes, stride, padding, kernel/input
dilation and reason once per signature (up to 256 distinct signatures). This
path can be slower and needs an unfolded-input workspace. Successful cuDNN
plans and Metal execution are unchanged.
Strict CUDA setup also prints a `[grid convolution context]` source-name map
once per run/prediction setup so backend shape signatures can be attributed to
named graph operations.

CUDA grid training's existing preflight executes and restores probe steps before
normal training. Failures include precision and named convolution candidates;
these names let you correlate lazy backend shape diagnostics with the source
graph, rather than guessing which operation failed. Keep the preflight enabled
for acceptance runs. The fix is in the pinned MLX CUDA library: rebuild the base
and architecture-tier images, not just the Mixlab binary. No opt-in flag is needed.

Training and `predict-grid` banners report dtype, backend, and the permission. This is permission for
eligible kernels, not a claim that every operation uses reduced precision. The
switch is locked for the process lifetime, including native eval/prediction.
Use a fresh process for each precision variant. This policy does not configure
an independently exported Hugging Face/PyTorch runtime.

`export-torch-state` reports `storage=float32` and the config's requested
`allow_tf32` value (or `unset`). Export only converts data, so this notice is not
an effective GPU policy and does not initialize a GPU. The exported inference
script currently runs PyTorch on CPU; other consumers must configure their own
compute precision.

Runs with a final safetensors path write `<path>.run.json`; otherwise runs with
a checkpoint directory write `<checkpoint-dir>/run.json`. Runs without an
artifact destination report precision in the banner only. Single-device and
DDP resume manifests also record the resolved policy. Exact resume rejects a
policy/backend change, including an environment-only change, or a legacy
manifest without precision provenance: a checkpoint from before this policy
cannot show whether reduced precision was allowed. To change policy
intentionally, or to continue a legacy run, start a new weights-only run; do
not edit the resume manifest. Matching policy is necessary but does not
guarantee bitwise results across GPUs or library versions.

Error and speed tradeoffs depend on shapes, values, and kernels. Our RTX 4090
tied-convolution fixture measured worst per-tensor normalized gradient errors
around `5e-3` to `9e-3` with TF32, versus `6e-7` to `1e-5` with it disabled,
checked against PyTorch float64. The two-channel stage-1 graph ran at 123-125
records/s with TF32 versus 96-102 with strict FP32 (batch 16, warmed-up compute
only). These are workload-specific observations, not error bounds or
full-dataset training guarantees. Strict FP32 does not guarantee bit-identical
CUDA and Metal results.

To reproduce, run the numerical probes in fresh processes against a float64
oracle (Metal runs them by default; CUDA needs the opt-in), optionally check the
dumps with PyTorch, then time each setting in its own process on CUDA:

```bash
MIXLAB_TEST_CUDA_PRECISION=1 MIXLAB_TF32_PROBE_OUT=/tmp/tf32-probes \
  go test -tags mlx ./gpu -run '^TestGPUPrecisionSubprocess$' -count=1 -v
python3 scripts/validate_tf32_probe.py /tmp/tf32-probes
MIXLAB_TEST_GRID_TF32=1 go test -tags mlx ./train -run '^TestGridTF32Throughput$' -count=1 -v
MIXLAB_TEST_GRID_TF32=0 go test -tags mlx ./train -run '^TestGridTF32Throughput$' -count=1 -v
```

For a standalone CUDA-library check without Go or model data, run
`bash docker/test_mlx_conv_precision.sh` on a CUDA development image with MLX.
It tests ordinary and transposed convolutions, cached-plan reuse, batch sizes
1 and 16, and large encoder/decoder shapes against scalar float64 contractions. Large tensors use
deterministic spatial samples, not an exhaustive model-level comparison. Neither
this check nor the small IR probes replace a trained-model fidelity check.

Strict FP32 does not promise identical results across CUDA, Metal, or PyTorch:
convolution algorithms and reduction order can differ. For accuracy reports,
compare each runtime directly with the same float64 reference, and record the
effective policy, batch size, GPU driver, cuDNN version, and MLX revision. A
cross-runtime difference alone does not identify which runtime is inaccurate.

## Piecewise-Constant Learning Rates

Grid training supports the shared `training.phases` schedule without restarting
the optimizer. For example, add these fields to the training object:

```json
{
  "phases": [{"steps":6,"lr":0.0001}, {"steps":4,"lr":0.00001}],
  "warmdown_steps": 0
}
```

The grid log numbers completed updates from 1: updates 1 through 6 use `0.0001`,
and 7 through 10 use `0.00001`. Phase lengths sum to the total number of updates;
`training.steps` is replaced by the phase total. `training.lr` remains the
reference denominator: effective group LR is
`group_lr * scheduled_phase_lr / training.lr` when the base is positive.
Default group rates inherit the base and therefore follow the phase LR exactly.
Explicit `matrix_lr` (including the `grid_kernel` group), scalar, or per-weight
overrides are scaled by the same ratio, so a config with phases and a nonzero
override must set `training.lr` explicitly (validation rejects it otherwise).
Setting it to the first phase LR makes those overrides apply as written in that
phase. With base LR zero, inherited rates remain zero and overrides are
not schedule-scaled. Startup/resume and phase-transition logs show effective
active-group rates; the optimizer startup summary shows configured rates.
AdamW/LAMB moments and step counters carry across the transition. Full-state
`-resume` restores them along with the phase position and data sampler; do not
use a weights-only warm start to continue a schedule.

Omit `warmdown_steps` or set it to zero for strictly constant phases. If positive,
the common scheduler applies a linear warmdown only within the last phase.
`warmup_steps`, `warmup_ratio`, and `hold_steps` are ignored for phase schedules;
`lr_schedule_steps` cannot be combined with phases. All phase lengths and rates
must be positive. See [training phases](config-reference.md#training-phases).

To reproduce an epoch-based step decay, convert epochs to updates using
`ceil(training_records / batch_size)`, including the last partial batch. For
40,080 records at batch 15, a 30-epoch phase is 80,160 updates and a subsequent
20-epoch phase is 53,440 updates. Shipped recipe budgets are illustrative;
choose the phase lengths explicitly for your dataset and experiment.

## Augmentation And Staged Training

`training.grid_augmentation: {"dihedral":true}` chooses uniformly among eight
joint D4 transforms per record occurrence. All input channels, targets, and
validity masks receive the same transform. Geometry must be square. Indices
0..3 rotate counterclockwise by 0/90/180/270 degrees; indices 4..7 mirror columns
first, then rotate. Selection is keyed by seed, epoch, occurrence, and record ID,
independent of shuffle RNG or worker timing. Omission or `dihedral:false` leaves
batches unchanged. Validation and prediction never augment. Scratch memory is
one record's largest NHWC tensor, reused between records.

For per-record pixel coordinates, declare ordered column pairs on the grid
adapter, for example `"vector_dim":5, "vector_points":[[2,0],[4,1]]`.
These pairs identify row/column point coordinates, not directions or displacement
vectors. Each column may appear only once; indices must be integers within the
vector width. Explicit empty/null declarations and malformed pairs are errors.
Omission keeps vectors unchanged, including the existing rejection of polar
graphs with D4 enabled and no point declaration.

Real rows' declared points receive exactly their map's transform in float32:
mirror `col = float32(n-1)-col` first for indices 4..7, then apply
`(row,col) = (float32(n-1)-col,row)` per quarter turn. Undeclared columns are
unchanged. Padded rows are never transformed; their vectors remain copies of
the original, unaugmented first row, with zero maps/targets/masks. Selection and
alignment are identical with or without prefetch and across exact resume.

The run manifest and exported config/manifest record `vector_points`; changing
the declaration rejects exact resume. It adds no weights or inference transform.
Validation, prediction and exported `infer.py --vectors` still use raw points.

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

### Memory failure recovery

When `-checkpoint-dir` is configured, training also writes a rolling recovery
bundle under `<checkpoint-dir>/recovery`. It saves initialized state before the
first training/preflight update, then completed steps at `-checkpoint-every`
cadence, or validation cadence if that option is omitted. The new complete bundle
is published before removing the previous recovery bundle. Normal periodic and
best checkpoints are retained separately.

On a memory failure, the error gives an explicit `-resume <manifest>` command;
`recovery/failure.json` records that path and completed step. `recovery/latest.json`
identifies the latest bundle directory. The failing step is never retried and its
potentially partial weights/moments are never saved. Resuming replays work after
the last known-good bundle, including pending validation. Recovery checkpoint
writes add CPU readback and disk IO at the configured cadence.

If final validation fails after all training updates have completed, resume the
final-step recovery manifest with the same `training.steps`. A grid bundle at
exactly that step with `validation_pending: true` finishes validation, best-model
selection, the periodic checkpoint (only when its cadence divides the final
step), and the requested `-safetensors` export without another training update.
The completion log explicitly identifies this recovery. Reuse the original
checkpoint directory to preserve any earlier best-model weights. A completed
final bundle without pending validation, or a bundle beyond the configured
training end, is still rejected. Rolling recovery bundles remain pre-validation
snapshots, not completion markers.

Without `-checkpoint-dir` no recovery bundle is created. Local ephemeral disk
does not survive pod deletion; use persistent storage or externally upload the
whole bundle. Recovery is not a guarantee against process kill or machine loss.

### Periodic resume bundles

Periodic checkpoints use the common `mixlab_resume_v1` bundle: a
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
CPU input/target/mask/vector batch buffers are capped at 1 GiB. The usual MLX memory
and cache limits also apply to training and prediction.

Vector-enabled datasets use `mixlab_grid_shard_v2`. The header version is `2`,
word 9 (zero-based) holds positive S, and words 10..255 remain zero. Each record
appends S little-endian float32 values after its map/target/mask payload. Vectors
remain float32 even when the header map dtype is float16. Target-free records
append vectors directly after input floats. Exact length, finite vectors and
manifest/header width agreement are checked. Map-only prepare still writes v1;
both formats remain readable.

## PyTorch State Export

`export-torch-state` is a CPU-only, grid-only weight/layout exporter. It does not
export Python model code or require MLX. Existing grid HF rejection is unchanged.
For rectangular/dilated convolutions the OHWI-to-OIHW weight mapping is unchanged;
the consumer's `torch.nn.Conv2d` must use the same per-axis kernel, dilation, stride
and padding. Map spatial `pad` to `torch.nn.functional.pad` using `(left,right,top,bottom)`
and modes `constant` (value 0), `replicate`, or `circular`, respectively. These settings
are retained in the exported `config.json`, not encoded in the weight tensor.

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

For a vector-enabled package, also pass `--vectors record_vectors.npy`. The
package declares `vector_dim`, `vector_dtype:"float32"`, and `vector_layout:"NS"`.
The helper validates finite float32 `[N,S]` vectors aligned with the input maps,
then calls `model(x, v)` in each batch, including the final partial batch. It
does not normalize vectors. Map-only packages still call `model(x)` and reject
an unexpected `--vectors` argument.

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
pool ties and gradients. Dilated convolution fixtures are regenerated separately:

```bash
python scripts/generate_grid_reference.py --dilation-only --out gpu/testdata/grid_dilation_reference.json
go test -tags mlx ./gpu -run 'TestGridSpatialPyTorchForwardBackward|TestGridDilation' -count=1 -v
go test -tags mlx ./train -run 'TestGridASPPTraining|TestGridDilationTorchExportParity' -count=1 -v
```

These tests cover dilation 2/4 forward and all gradients, stride/padding, optional
bias, a 100-step ASPP training run and post-training PyTorch export parity.
The training test reports compute-step throughput after five warmup steps and
process-wide MLX peak memory; run it alone for comparable memory measurements.
The export check additionally requires Python with PyTorch and safetensors.

`scripts/generate_grid_model_reference.py` derives
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
from the pinned RadioWNet implementation and retain its MIT license notice.
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

## Loader And Throughput

Grid training defaults to two prefetched batches and up to
`min(batch_size, GOMAXPROCS)` parallel read/decode workers. Configure runtime
loading independently of the model:

```json
"grid_loader": {"prefetch_batches": 2, "read_workers": 0}
```

Place this object under `training`. Depth 0 with workers 1 is the serial
baseline. Depth must be 0..64; workers 0 is automatic, otherwise 1..256 (capped
by batch size). Validation and prediction use parallel reads but no prefetch.
Changing these settings on resume preserves sampled order, D4 transforms and
padding. Checkpoints save the consumed cursor, never the read-ahead cursor.

Read, decode and augmentation run on the CPU while the GPU computes the prior
batch. The pool contains at most `prefetch_batches + 1` decoded batches,
including the consumer and any batch under construction. Each read worker
holds at most one open shard and one encoded-record scratch buffer. The banner
reports the decoded pool and scratch bounds separately. Decoded bytes are
`(depth+1) * batch_size * (H * W * (C+2*Ct) + S) * 4`, where S is vector width
(0 when disabled); targets, broadcast masks and float32 vectors count
even when shards are float16. Metadata, OS page cache, GPU input copies, and the
separate validation batch are not included. This is not a whole-dataset cache.
Early stop/error cleanup joins workers; an in-progress filesystem read must
return before shutdown can finish.

Logs and telemetry distinguish:

- `records/s` / `records_per_sec`: compute-only rate, unchanged.
- `train_records/s` / `train_records_per_sec`: consumer step rate including data
  waits and step bookkeeping, excluding validation/checkpoints and log writes.
- `data_wait` / `data_wait_percent`: percentage of measured consumer step time
  blocked obtaining a batch, including synchronous preparation when depth is 0.
- `wall_records/s` / `wall_records_per_sec`: all real records divided by elapsed
  run-loop time, including previous validation/checkpoints and startup loading.

Compute and consumer-step rates exclude the first update after start/resume to
avoid compilation skew. Wall rate includes it, but excludes preflight and
initial validation before the training loop. Padding never counts as records.
The completion line includes final validation/checkpoint/export time.

Compare depth 0/workers 1 against depth 2/automatic workers on the same dataset,
seed, step count and validation/checkpoint cadence. Record all three rates and
data-wait percentage. Test local disk and network storage separately; warm
page-cache measurements are not cold-network measurements. More workers can
overload shared storage, so tune down for concurrent jobs. A wall rate within
10% of compute is a workload target, not a guarantee.

A repeatable host-only overlap check is
`go test ./data -run '^$' -bench BenchmarkGridPrefetchOverlap -benchtime=40x`.
It uses local synthetic 7-channel 256x256 float16 shards and a sleeping consumer
to simulate compute. Its `sim-compute-records/s` is not GPU throughput.
