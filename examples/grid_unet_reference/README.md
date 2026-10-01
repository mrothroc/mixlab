# Two-Stage Grid U-Net Reference

These full-resolution graphs reproduce the pinned RadioWNet architecture, not a
published dataset score. Two-channel total: 13,274,031 parameters; three-channel
total: 13,275,315. Both use 256x256 grids and retain the complete U1/refinement
weight inventory in each stage. `*_shapes.json` records every intermediate NHWC
shape from the actual reference. `*_export_map.json` maps all tensors, including
frozen ones, to strict PyTorch state_dict names and layouts.

The source URL, revision and SHA256 are in the provenance and map files. Both
upstream MIT notices are included. No upstream Python module, trained weights,
dataset, or benchmark-specific preparation logic is bundled.

```bash
mixlab -mode arch -config examples/grid_unet_reference/two_channel_stage1.json \
  -train data/grids/mixlab.dataset.json -safetensors stage1.safetensors
mixlab -mode arch -config examples/grid_unet_reference/two_channel_stage2.json \
  -train data/grids/mixlab.dataset.json -safetensors-load stage1.safetensors \
  -safetensors stage2.safetensors
mixlab -mode export-torch-state \
  -config examples/grid_unet_reference/two_channel_stage2.json \
  -safetensors-load stage2.safetensors \
  -export-map examples/grid_unet_reference/two_channel_export_map.json \
  -export-dir exported-grid
```

Replace `two_channel` with `three_channel` for the three-input-channel variant.
Supply explicitly separated train/validation grids with matching channel counts.
Normalize outside prepare using training-only statistics; track both offset and
scale. [Dense grid guide](../../docs/dense-grid.md) documents the full workflow.

## Recipe Boundaries

- These examples choose 1,000 and 500 **steps** as illustrative run durations,
  not the paper's training budget. Set durations for your dataset and resources.
- AdamW with beta1=0.9, beta2=0.999, epsilon=1e-8 and zero global/per-group decay
  gives Adam-equivalent updates. Step-wise cosine has a 0.01 LR floor, no warmup,
  and a fresh schedule/optimizer in stage two; it is not epoch-wise reference
  scheduling. Batch size is deliberately one to limit memory.
- Compute is FP32, not reference AMP. BF16 custom graphs are currently rejected.
  PyTorch and Mixlab RNG streams differ even with the same seed. Initialization
  uses reference convolution fan-in distributions, not identical random draws.
- Joint D4 augmentation is deterministic within Mixlab; it is not a promise of
  matching reference augmentation draws. No dropout or normalization is added.
- Prepare defaults to FP32 storage. Opting into FP16 shards introduces storage
  quantization; it is distinct from compute precision and must be evaluated.
- Stage one selects U1 and prunes refinement execution; stage two freezes U1
  and follows the reference detach boundaries. Both outputs end in configured
  ReLU. Neither prediction nor export silently clamps outputs.

Native-to-PyTorch acceptance uses four deterministic synthetic maps per variant,
strict state loading and an absolute FP32 output tolerance of 1e-4, before and
after native updates. This does not establish a dataset validation/test score.

Reproduce the small generated files by running
`scripts/generate_grid_model_reference.py` with both channel counts, both phases,
and `--samples 4` into `<fixtures>/<channels>/<phase>`, then
`scripts/generate_grid_reference_recipes.py --fixtures <fixtures> --output <dir>`.
Never add the generated large tensors to the repository.
