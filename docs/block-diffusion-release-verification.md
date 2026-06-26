# Block Diffusion Release Verification

Release: Block Diffusion Walking Skeleton (`1948521454036569230`)

Design source: `docs/block-diffusion-design.md`

## Component Tasks

- `6249957411792026387` - Add `block_diffusion` config schema and validation.
- `6584716334765802512` - Prepare block-diffusion objective batches.
- `4488140685783850519` - Implement block-diffusion attention mask IR and MLX backend.
- `5074017036900732476` - Wire `block_diffusion` training graph and smoke coverage.
- `3556629738940623545` - Refactor evaluation through objective batches.
- `1049652243447570529` - Implement block-diffusion sampler and generation CLI path.
- `553235821174218525` - Document block diffusion and add tiny example config.

## Supported Policy

This release implements the single-tower walking skeleton from
`docs/block-diffusion-design.md`: `training.objective="block_diffusion"`,
`training.diffusion` schema, deterministic block corruption, prefix-plus-block
attention masking, masked CE loss, forward-only objective evaluation, pure
commit-selection sampler helpers, `generate-diffusion` CLI wiring, and a tiny
example config.

The supported v1 block surface is sequential `plain` self-attention plus
position-wise `swiglu`, `geglu`, `mlp`, and `moe` blocks.

Unsupported surfaces intentionally fail or remain deferred: two-tower Nemotron
style context/denoiser towers, bidirectional Mamba or SSM denoisers, hybrid
AR/diffusion schedules, segment attention composition, windowed attention,
distillation, data2vec, top-level MTP, first-byte masking, and HF diffusion
export.

## Verification Commands

The following commands were run from `/Users/mrothroc/IdeaProjects/mixlab`.

```sh
go test ./arch -run 'Test.*BlockDiffusion.*Config|Test.*BlockDiffusion.*Objective|Test.*DiffusionSpec|TestTrainingObjective|TestTrainingAttentionSegmentMask|TestMLMHeadValidation' -count=1
```

Result: passed.

```sh
go test ./train -run 'TestPrepareBlockDiffusionBatch|TestBlockDiffusionObjectiveForStep|TestBlockDiffusionBatchDeterministic|Test.*DiffusionSampler|Test.*Commit.*Diffusion' -count=1
```

Result: passed.

```sh
go test ./arch ./gpu -run 'Test.*BlockDiffusionMask|Test.*BlockDiffusion.*MaskKind' -count=1
```

Result: passed.

```sh
go test ./gpu -run TestIRToGPUOpCodeAlignment -count=1
```

Result: passed.

```sh
go test -tags mlx ./gpu -run TestBlockDiffusionMaskMatchesOracle -count=1
```

Result: passed.

```sh
go test ./train ./gpu -run 'Test.*Evaluate.*Objective|Test.*EvaluateGPU|Test.*ObjectiveInputs' -count=1
```

Result: passed.

```sh
go test -tags mlx ./train -run 'TestMLXGPUTrainerMakeObjectiveInputs' -count=1
```

Result: passed.

```sh
go test ./arch ./train -run 'Test.*BlockDiffusion.*IR|Test.*BlockDiffusion.*Inputs|Test.*BlockDiffusion.*WeightCount|Test.*BlockDiffusion.*Smoke' -count=1
```

Result: passed.

```sh
go test -tags mlx ./train -run 'Test.*BlockDiffusion.*(Inputs|Smoke)' -count=1
```

Result: passed.

```sh
go test ./train -run 'Test.*DiffusionSampler|Test.*Generate.*Diffusion|Test.*Commit.*Diffusion' -count=1
```

Result: passed.

```sh
go test ./train -run 'TestGeneration|TestGenerate' -count=1
```

Result: passed.

```sh
go test -tags mlx ./train -run 'Test.*Generate.*Diffusion.*Smoke' -count=1
```

Result: passed.

```sh
go test ./cmd/mixlab -count=1
```

Result: passed.

```sh
go test ./arch ./train ./gpu ./cmd/mixlab -run 'Test.*BlockDiffusion|Test.*DiffusionSampler|Test.*Generate.*Diffusion|Test.*Evaluate' -count=1
```

Result: passed.

```sh
go test -tags mlx ./gpu ./train -run 'Test.*BlockDiffusion|Test.*Generate.*Diffusion' -count=1
```

Result: passed.

```sh
rg -n '<diffusion no-stub sentinel pattern>' arch train gpu cmd docs examples
```

Result: passed with no matches.

```sh
git diff --check
```

Result: passed.

```sh
weft releases validate --release-id 1948521454036569230
```

Result: passed with no dependency errors or warnings.

```sh
weft releases review --release-id 1948521454036569230 --format summary
```

Result: approved. Review session `4794794a-4f4d-4972-b077-10f648326dc0`
reported no Critical or High findings. The two Low findings were documentation
and operations follow-ups: keep the future `mask_token_id` migration path
visible, and record an MLX-backend-unavailable reason if a future runner lacks
MLX support. This verification run was performed on an MLX-capable machine, and
the MLX backend tests listed above passed.
