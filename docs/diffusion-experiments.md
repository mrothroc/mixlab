# Diffusion Experiment Workflow

This workflow uses existing mixlab modes only. It is meant for comparing pure
causal, MLM, block diffusion, and hybrid causal/block-diffusion runs on the same
backbone, data, tokenizer, context length, optimizer, and training-token budget.

## Matrix

Start with a small matrix before scaling:

| Run | Config starting point | Purpose |
|-----|-----------------------|---------|
| Causal baseline | `examples/plain_3L.json` or a matched local config | Common validation-loss reference |
| MLM baseline | `examples/mlm_tiny.json` or matched local config | Separates masked-objective effects from block denoising |
| Block diffusion | `examples/block_diffusion_tiny.json` | Pure block-wise masked diffusion |
| Hybrid diffusion | `examples/hybrid_block_diffusion_tiny.json` | Scheduled causal plus block-diffusion training |
| Multihead scorer + denoiser | `examples/multihead_mntp_diffusion_tiny.json` | Shared trunk with an MNTP scorer head and native block-diffusion denoiser head |
| Sampler variant | `examples/block_diffusion_sampler_sweep.json` | Different generation-time denoising settings |

Keep model shape and training budget fixed across rows. Change one objective or
sampler setting at a time.

## Train

```bash
./mixlab -mode arch -config examples/hybrid_block_diffusion_tiny.json \
  -train 'data/train_*.bin' \
  -safetensors runs/hybrid-block-diffusion.safetensors \
  -log-every 50 -val-every 200
```

For a config directory, use the existing race mode:

```bash
./mixlab -mode arch_race -configs experiments/diffusion_matrix \
  -train 'data/train_*.bin' -log-every 50 -val-every 200
```

## Evaluate

Use causal validation loss/BPB for apples-to-apples comparison:

```bash
./mixlab -mode eval -config examples/hybrid_block_diffusion_tiny.json \
  -train 'data/val_*.bin' \
  -safetensors-load runs/hybrid-block-diffusion.safetensors
```

Optional per-token exports make ranking and uncertainty analysis reusable:

```bash
./mixlab -mode eval -config examples/hybrid_block_diffusion_tiny.json \
  -train 'data/val_*.bin' \
  -safetensors-load runs/hybrid-block-diffusion.safetensors \
  -logprobs-out runs/hybrid.nll.bin \
  -uncertainty-out runs/hybrid.uncertainty.bin
```

Use native block-diffusion PLL scoring when forced-choice ranking should use the
diffusion forward instead of causal next-token likelihood:

```bash
./mixlab -mode score-diffusion \
  -config examples/hybrid_block_diffusion_tiny.json \
  -safetensors-load runs/hybrid-block-diffusion.safetensors \
  -score-in runs/candidates.tokens.jsonl \
  -score-out runs/hybrid.diffusion_scores.jsonl \
  -score-skip-first 1
```

The input is JSONL with caller-tokenized token IDs, for example
`{"id":"case_0","tokens":[1,815,22,4],"score_from":1}`. The score is a
deterministic block-causal pseudo-log-likelihood under the prefix-plus-block
attention graph. It is useful for comparing candidate sentences, but it is not a
true normalized sequence likelihood.

## Sample And Trace

`generate-diffusion` works with pure `block_diffusion` checkpoints, hybrid
checkpoints whose secondary objective is `block_diffusion`, and multihead
checkpoints with a block-diffusion `diffusion_head`.

```bash
./mixlab -mode generate-diffusion \
  -config examples/hybrid_block_diffusion_tiny.json \
  -safetensors-load runs/hybrid-block-diffusion.safetensors \
  -prompt token_ids:0,1,2 -max-tokens 32 \
  -diffusion-trace-out runs/hybrid-sampler.jsonl
```

Sampler overrides are useful for sweeps without editing the trained config:

```bash
./mixlab -mode generate-diffusion \
  -config examples/hybrid_block_diffusion_tiny.json \
  -safetensors-load runs/hybrid-block-diffusion.safetensors \
  -prompt token_ids:0,1,2 -max-tokens 32 \
  -diffusion-steps-per-block 4 \
  -diffusion-confidence-threshold 0.65 \
  -diffusion-commit-floor 2 \
  -diffusion-temperature 0.8 \
  -diffusion-top-k 32 \
  -diffusion-trace-out runs/hybrid-sampler-temp08.jsonl
```

The trace is JSONL with one record per denoising pass. Track at least
`block`, `step`, `unresolved_before`, `committed`, `forced`,
`unresolved_after`, `complete`, and the confidence summary fields.

## Record

For each run, record:

- Config path and checkpoint path.
- Training loss trend and causal validation loss/BPB.
- Generation prompt, max tokens, sampler overrides, and trace path.
- Tokens committed per step, forced-commit rate, unresolved positions at final
  step, total denoising steps, and wall-clock sampling latency.
- A small fixed prompt set so sampler changes can be compared across checkpoints.

Do not compare diffusion masked training loss directly against causal validation
loss. Use the masked loss to debug the objective and causal eval to compare
models.
