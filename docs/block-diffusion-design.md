# Block Diffusion Technical Design

## Problem Analysis

mixlab currently supports architecture experiments where a JSON config builds a
single forward graph, trains it with a small set of objectives, and evaluates it
with a causal next-token graph. Diffusion language model experiments need a new
objective and generation procedure rather than a new normal block type. The
initial goal is a walking skeleton for block-wise masked diffusion that can train
and generate with existing Transformer-style blocks while preserving the current
weight-layout and evaluation invariants.

The target release is not a literal Nemotron-style two-tower implementation.
Nemotron's released model uses a frozen autoregressive context tower and a
trainable denoiser tower with layer-aligned attention KV and Mamba state
transfer. That is useful as a later release, but it is too large for the first
reviewable slice. The first slice should answer whether mixlab can run controlled
block-diffusion experiments end to end.

Reference constraints from the existing code:

- `arch/objective.go` owns objective names, masked-objective detection, and
  attention-mask resolution. The new objective must fit this path instead of
  bypassing it.
- `train/objective.go` prepares per-step objective batches. Block diffusion
  should extend this mechanism with per-example block boundaries and loss masks.
- `arch/builder.go` declares objective-specific graph inputs, builds the shared
  block stack, and emits either dense or masked losses. The new graph must keep
  the same trainable weight count as causal/eval graphs.
- `arch/blocks.go` routes `plain` attention masks through `emitPlainAttentionMaskIR`.
  This is the narrowest place to add block-aware attention behavior.
- `train/gpu_trainer_inputs.go` already passes optional inputs such as
  `loss_mask`, `attention_causal_mask`, and `segment_ids`; block diffusion
  should add inputs through the same objective-batch plumbing.
- `train/train.go` caches and switches training programs by objective and
  sequence length. The new objective must participate in that key without
  changing optimizer state layout.
- `train/generate.go` is strictly one-token autoregressive today. Block-diffusion
  generation needs a separate path that can evaluate objective-specific inputs
  without taking optimizer steps.
- `gpu_trainer_mlx.go` already has `EvaluateGPU` and `EvaluatePerTokenGPU`, but
  those currently accept only token and target arrays. Block diffusion should
  refactor evaluation around an objective-batch entry point instead of adding a
  second, drifting evaluation path.

Important behavioral constraints:

- Causal evaluation remains the common comparison metric. `BuildEvalIRProgramFromConfig`
  already forces `ObjectiveCausal`; block-diffusion training must not change that.
- The training loss should be masked cross-entropy over selected noisy positions,
  not dense next-token loss.
- Attention for the active denoising block must be prefix-plus-block: tokens in
  the noisy block can attend to earlier committed context and bidirectionally
  inside the active block, but not to future blocks.
- Existing `SegmentAttentionMask` is not sufficient for that rule. It makes
  segments independent, while block diffusion requires the active block to see
  prior committed context. Global bidirectional MLM is also insufficient because
  it leaks future context and does not test block-wise AR diffusion.
- Causal Mamba blocks do not become bidirectional denoisers just because the loss
  is masked. Mamba/diffusion support must be staged carefully.

Assumptions:

- The first release targets fixed-length training batches and block sizes that
  divide `seq_len`. Relaxing that can come later.
- A configured `mlm_mask_token_id` remains the mask token source for v1, avoiding
  tokenizer/schema churn. This is acknowledged API debt; a later cleanup should
  rename or alias it to an objective-neutral `mask_token_id`.
- The first release supports `plain` attention blocks plus pointwise FFN/MoE
  blocks. Recurrent, SSM, and custom blocks are either rejected or allowed only
  under explicit documented caveats until their denoising semantics are tested.

Current implementation boundary:

- The checked-in schema validates `training.objective="block_diffusion"` and
  `training.diffusion`, and the pure sampler helper code covers commit
  selection policy. This is enough for config parsing, counting, and IR smoke
  coverage.
- End-to-end block-diffusion training batch preparation, the backend
  prefix-plus-block attention mask, and a public diffusion generation CLI remain
  separate implementation work. Documentation should describe those paths as
  planned behavior until they land.
- The existing `generate` mode remains causal next-token generation. Future CLI
  wording may refer to a `generate-diffusion` path, but should not present it as
  an implemented mode in this release.

## Proposed Solution

Add `training.objective="block_diffusion"` as a new masked objective. It uses the
existing token embedding, block registry, final norm, output head, and
`MaskedCrossEntropy` loss path, but supplies block-specific corruption and
attention-mask inputs.

Add a nested diffusion config under `training`:

```json
{
  "training": {
    "objective": "block_diffusion",
    "mlm_mask_token_id": 3,
    "diffusion": {
      "block_size": 16,
      "steps_per_block": 16,
      "min_mask_fraction": 0.05,
      "max_mask_fraction": 1.0,
      "confidence_threshold": 0.8,
      "commit_floor": 1
    }
  }
}
```

These field names match the v1 schema. `mlm_mask_token_id` identifies the mask
token, while `training.diffusion` controls block scheduling and sampling
behavior.

`training.diffusion` intentionally contains both training-time and generation-time
fields in v1. `block_size`, `min_mask_fraction`, and `max_mask_fraction` affect
training corruption. `steps_per_block`, `confidence_threshold`, and `commit_floor`
affect generation. Keeping them together is a pragmatic first API because the
sampler must match the trained block size; if this grows, a later release can
split sampler-only settings into an eval/generation namespace.

Training batch preparation:

1. For each sequence in the batch, select one active diffusion block
   `[block_start, block_end)`.
2. Sample a mask fraction `t` in `[min_mask_fraction, max_mask_fraction]`.
3. Mask selected positions inside the active block in `x`.
4. Set `y` to the original clean tokens.
5. Set `loss_mask=1` only for selected masked positions.
6. Pass `diffusion_block_start` and `diffusion_block_end` as per-example int32
   vectors of length `B`.

IR and attention:

- Add `ObjectiveBlockDiffusion`.
- Treat it as a masked objective for loss and `loss_mask` input declaration.
- Add a new attention-mask mode, tentatively `AttentionMaskBlockDiffusion`.
- Add an IR op such as `OpBlockDiffusionMask` with inputs:
  `scores`, `diffusion_block_start`, `diffusion_block_end`.
- The backend op masks `[B,H,T,T]` attention scores with this rule:
  - query position `< block_start`: causal prefix behavior, `key <= query`;
  - query position in `[block_start, block_end)`: allow keys `< block_end`;
  - query position `>= block_end`: causal behavior, `key <= query`.
- Reject `window_size` with block diffusion in v1. The sliding-window semantics
  across prefix-plus-block attention are not obvious and should not be guessed.
- Reject segment attention masking with block diffusion in v1 unless a later
  design extends the mask op to compose both constraints.
- Implement the mask with the same additive masking form as `OpCausalMask`:
  `mx::where(mask, mx::full_like(scores, -1e9f), scores)`. The op has no trainable
  state and relies on normal MLX autograd through `where`; it does not need a
  custom backward or VJP.

Generation:

- Keep the existing `generate` behavior as the default AR path.
- Add a block-diffusion generation path, either as a new CLI mode
  (`-mode generate-diffusion`) or as `-generate-mode block_diffusion`.
- Build a block-diffusion eval graph with objective inputs and no optimizer
  update.
- Add a trainer method for forward-only objective evaluation, for example
  `EvaluateObjectiveGPU(batch objectiveBatch, batchSize, seqLen int)`. This should
  become the general evaluation implementation: existing `EvaluateGPU` delegates
  by wrapping `{x, y}` into an `objectiveBatch`, and per-token evaluation follows
  the same input-construction path where applicable.
- For each output block:
  1. Append `block_size` mask tokens to the committed context.
  2. For up to `steps_per_block` iterations, run the denoiser over the sequence.
  3. Convert logits for unresolved positions into probabilities.
  4. Commit tokens above `confidence_threshold`.
  5. If nothing meets the threshold, commit at least `commit_floor` highest
     confidence positions so the block completes.
  6. Keep committed positions fixed and leave unresolved positions masked.
  7. Once the block is complete, move to the next block.

Alternatives considered:

- Reuse global MLM only. This is easy but not a block-diffusion experiment; it
  leaks future context and cannot evaluate block-wise decoding.
- Reuse `SegmentAttentionMask`. It prevents cross-segment leakage but also hides
  prior context from the active block, which is the wrong conditioning pattern.
- Add a `diffusion` block type. This puts the behavior at the wrong abstraction
  level; diffusion is an objective and sampler over an existing backbone.
- Start with two towers. This matches Nemotron more closely, but it multiplies
  weight layout, checkpoint, generation, and state-transfer complexity before
  proving the basic objective.

## Implementation Plan

Phase 1: schema and validation

- Add `ObjectiveBlockDiffusion` and normalize/validate it in `arch/objective.go`.
- Add `DiffusionSpec` to `TrainingSpec` with defaults and validation.
- Require `mlm_mask_token_id` for `block_diffusion` in v1, and document the
  planned objective-neutral `mask_token_id` rename/alias.
- Require `diffusion.block_size > 0`, `diffusion.block_size <= seq_len`, and
  initially `seq_len % diffusion.block_size == 0`.
- Reject incompatible v1 features: `mtp`, `first_byte_mask`, distillation, segment
  attention masking, and `plain.window_size`.
- Document block support. Initial implementation should allow sequential
  `plain`/`swiglu`/`geglu`/`mlp`/`moe` stacks. Recurrent/Mamba/custom support
  should be rejected unless a follow-up explicitly tests their semantics.

Phase 2: objective batch preparation

- Extend `objectiveBatch` with:
  - `diffusionBlockStart []int32`
  - `diffusionBlockEnd []int32`
- Add `prepareBlockDiffusionBatch`.
- Add deterministic RNG seeding derived from training seed and step, following
  the existing MLM/MNTP pattern.
- Ensure each example masks at least one token when the mask range allows it.
- Preserve `unmaskedX` so data2vec/segment helpers remain well-defined if later
  support is added.

Phase 3: IR and backend mask

- Add `OpBlockDiffusionMask` in `arch/ir.go`, `gpu/mlx_types.go`, `gpu/ir.h`,
  `gpu/lower.go`, and `gpu/ir.cpp`.
- Add `Program.BlockDiffusionMask(...)` wrapper.
- Add a small mask-kind resolver in `arch/objective.go` so objective resolution
  produces a normalized attention mask kind. `blocks.go` should dispatch on that
  kind instead of growing direct objective-name knowledge inside block emission.
- Extend `emitPlainAttentionMaskIR` to emit the new op for the resolved
  block-diffusion mask kind.
- Declare `diffusion_block_start` and `diffusion_block_end` inputs for the
  objective.
- Extend `mlxGPUTrainer.makeObjectiveInputs` to pass the new vectors.
- Add pure Go and MLX oracle tests for the mask pattern.

Phase 4: training integration

- Extend `objectiveForStep` and `canonicalObjective`.
- Ensure program-cache keys already include the objective string and continue to
  switch correctly.
- Confirm `BuildEvalIRProgramFromConfig` remains causal for validation and
  standard generation.
- Add smoke tests proving loss decreases on a tiny block-diffusion config.

Phase 5: generation integration

- Add forward-only objective evaluation to the trainer interface and MLX trainer.
  Refactor existing `EvaluateGPU` to delegate to `EvaluateObjectiveGPU` with a
  plain `{x,y}` objective batch so causal and diffusion evaluation cannot drift.
- Implement a block-diffusion sampler with deterministic RNG, top-k/temperature
  reuse where sensible, confidence-based commit, and guaranteed completion.
- Put sampler logic in `train/diffusion_sampler.go`. Keep pure commit-selection
  and unresolved-position bookkeeping separate from the GPU forward call so the
  highest-risk loop is unit-testable without MLX.
- Add CLI flags or a new mode without changing existing `generate` defaults.
- Keep output format compatible with current `generated token_ids:...` output.

Phase 6: examples and documentation

- Keep a tiny example config at `examples/block_diffusion_tiny.json`.
- Update `docs/config-reference.md`, `docs/architecture.md`, and `docs/cli.md`.
- Add a short experiment note that explains v1 limitations and recommended
  comparisons against causal and MLM baselines.

Deferred releases:

- Hybrid AR/block-diffusion schedule inside one run.
- Two-tower context/denoiser support with checkpoint layout and frozen tower
  semantics.
- Bidirectional Mamba or context-seeded Mamba state support.
- HF export support for diffusion generation.

Recommended experiment comparisons:

- Compare block diffusion against a causal baseline with the same `blocks`,
  `model_dim`, `seq_len`, `vocab_size`, optimizer settings, data, and training
  token budget. Keep causal validation BPB/per-token NLL as the common metric.
- Compare against an MLM baseline with the same supported Transformer backbone
  to separate "masked objective" effects from block-wise denoising effects.
- Track diffusion masked training loss separately from causal validation loss;
  they answer different questions and should not be treated as interchangeable.
- Once diffusion generation is implemented, compare sampler quality and latency
  using the same prompts, seeds, `block_size`, and decoding budget across runs.

## Technical Decisions

- Implement diffusion as a training objective plus sampler, not as a registered
  block. This preserves mixlab's block registry model and keeps existing
  architectures reusable.
- Add a new block-aware attention mask op rather than approximate the behavior
  with global bidirectional or segment masks. The mask is the key semantic
  boundary for avoiding future-token leakage.
- Resolve objective-specific attention needs into a mask-kind value before block
  emission. This keeps `blocks.go` focused on emitting the selected mask and
  avoids repeatedly teaching block code about objective names.
- Keep trainable weight layout unchanged across causal eval and block-diffusion
  training graphs. New tensors are runtime inputs only.
- Implement `OpBlockDiffusionMask` as a pure forward `where` mask over attention
  scores, matching the existing causal mask pattern. Backend autograd should
  handle gradients through the unmasked score entries with no custom VJP.
- Use the existing `MaskedCrossEntropy` loss path. This keeps objective metrics
  aligned with MLM/MNTP and avoids introducing a new loss primitive before there
  is evidence it is needed.
- Keep causal eval as the comparable validation metric. `eval_loss` during
  block-diffusion training remains dense CE for telemetry, but final validation
  should use the existing eval graph.
- Start with `plain` attention because it can express bidirectional in-block
  denoising. Causal Mamba is not equivalent to DiffuMamba. Proper Mamba/diffusion
  needs either a bidirectional scan variant or a paired forward/backward block
  design, which should be reviewed separately.
- Prefer deterministic per-step corruption for tests and reproducibility. The
  objective RNG should follow the existing seed/step derivation style rather than
  use global random state.
- Keep `training.diffusion` as the v1 home for both corruption and sampler knobs,
  despite the mixed training/generation concerns, because the sampler must share
  the trained block size. Revisit this if generation policy grows independently.
- Treat reuse of `mlm_mask_token_id` as temporary compatibility debt. The release
  should document it and leave a path to `mask_token_id` without changing behavior
  mid-implementation.

## Testing Strategy

Unit and config tests:

- Parse succeeds for a minimal `training.objective="block_diffusion"` config with
  valid `training.diffusion`.
- Parse fails when `mlm_mask_token_id` is missing, block size is invalid,
  `seq_len % block_size != 0`, or incompatible features are configured.
- `objectiveForStep` and `canonicalObjective` return `block_diffusion`.
- `NeedsMaskedLoss` treats block diffusion as masked.

Objective-prep tests:

- `prepareBlockDiffusionBatch` masks only positions inside the selected block.
- `loss_mask` is active only where `x` was replaced with the mask token.
- `y` preserves original clean tokens.
- `diffusionBlockStart` and `diffusionBlockEnd` match configured block size.
- Selection is deterministic for a fixed seed and step.

IR tests:

- Block-diffusion training graph declares `loss_mask`,
  `diffusion_block_start`, and `diffusion_block_end`.
- `plain` blocks emit `OpBlockDiffusionMask` instead of `OpCausalMask` or
  `OpSelectiveCausalMask`.
- Causal eval graph for the same config emits normal causal masks and does not
  declare diffusion inputs.
- Weight counts match between causal eval and block-diffusion training graphs.

Backend tests:

- Add an oracle test for `OpBlockDiffusionMask` over small `[B,H,T,T]` scores and
  varied block boundaries.
- Include boundary cases: block at start, block at end, block size 1, and whole
  sequence block.
- Run MLX smoke tests for tiny Transformer block-diffusion training.

Generation tests:

- Unit-test commit selection: threshold commit, fallback commit floor, and fixed
  committed-token behavior.
- Unit-test the pure sampler helpers in `train/diffusion_sampler.go` without MLX,
  using fake logits/probabilities and explicit unresolved-position sets.
- Add a fake-logit sampler test that proves all positions in a block resolve
  within `steps_per_block`.
- Add a small end-to-end generation smoke behind the existing MLX test pattern.

Release acceptance criteria:

- Structural: new objective/config types, attention mask op, objective batch
  inputs, generation path, docs, and example config exist.
- No-stub: no incomplete implementation markers, panic-based implementation
  markers, or unimplemented-path markers remain in touched diffusion code.
- Behavioral: relevant Go unit tests pass; MLX mask and smoke tests pass where
  the backend is available; existing causal/MLM objective tests continue to pass.

## Risk Assessment

- Risk: future-token leakage through attention masks.
  Mitigation: implement a dedicated block-diffusion mask op and test it with
  small explicit oracle matrices.

- Risk: the first release overpromises Mamba/diffusion.
  Mitigation: document v1 as Transformer block diffusion. Reject or clearly gate
  recurrent/Mamba blocks until bidirectional scan semantics are designed and
  tested.

- Risk: weight layout changes break checkpoint load, SWA, recurrence scheduling,
  or program switching.
  Mitigation: add explicit weight-count parity tests between training and eval
  graphs and keep all new diffusion state as runtime inputs.

- Risk: generation requires objective-specific inputs but the current trainer
  evaluate path only accepts token and target arrays.
  Mitigation: add `EvaluateObjectiveGPU` and route it through existing
  `makeObjectiveInputs` so training and generation share input construction.

- Risk: validation metrics become hard to compare with causal baselines.
  Mitigation: keep causal eval as the primary validation metric and report
  diffusion training loss separately.

- Risk: block-size and schedule fields become premature API baggage.
  Mitigation: use a small nested `training.diffusion` namespace and document v1
  constraints. Keep defaults conservative and fail fast on unsupported
  combinations.

- Risk: backend op growth increases MLX compile pressure.
  Mitigation: the new mask op is shape-stable and contains no trainable weights.
  Initial smoke tests should use small configs; scale testing can be a separate
  release after the semantics are verified.
