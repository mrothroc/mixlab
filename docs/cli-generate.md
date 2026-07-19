# CLI: Generation

Generation modes load a checkpoint and produce token IDs. They do not update
weights.

## `generate`

Generate causal next-token samples:

```bash
./mixlab -mode generate \
  -config examples/plain_3L.json \
  -safetensors-load weights.safetensors \
  -prompt token_ids:0,1,2
```

| Flag | Description |
|------|-------------|
| `-config` | Required. JSON architecture config. |
| `-safetensors-load` | Required. Checkpoint to load. |
| `-prompt` | Prompt token IDs in `token_ids:0,1,2` form. |
| `-max-tokens` | Maximum generated tokens. Default: `256`. |
| `-temperature` | Sampling temperature. Default: `0.8`. |
| `-top-k` | Top-k sampling cutoff. `0` disables the cutoff. |
| `-num-samples` | Number of independent sequences generated while reusing one initialized trainer. Default: `1`. |
| `-gen-seed` | Base sampling seed. `0` uses `training.seed`. Sample `0` retains the base RNG stream; later samples use deterministically mixed seeds. |
| `-eos-token-id` | Stop after emitting this token. Default: `-1` (disabled unless `-sequence-vocab` supplies EOS). |
| `-generate-out` | Optional line-oriented output file. Without a sequence vocabulary each line is token-ID CSV; with one, each line is decoded sequence text. |
| `-sequence-vocab` | Optional `nucleotide_vocab.json`. Enables `sequence:ACGT...` prompts and decoded nucleotide output. |

Bulk generation initializes the model once and streams one completed sample per
line:

```bash
./mixlab -mode generate \
  -config examples/plain_3L.json \
  -safetensors-load weights.safetensors \
  -prompt token_ids:1 \
  -num-samples 1000 \
  -gen-seed 7 \
  -eos-token-id 2 \
  -max-tokens 100 \
  -generate-out samples.txt
```

Token-ID records contain the prompt and generated continuation, including an
emitted EOS token. Decoded sequence records omit framing tokens according to
the supplied vocabulary. Output is deterministic: the first `K` records from a
larger run equal a `K`-sample run using the same seed and sampling settings.

When `-num-samples=1`, `-gen-seed=0`, and `-generate-out` is omitted, Mixlab
retains the existing human-readable `generated token_ids:` output. Bulk or
file-output runs suppress status messages so their records are safe for scripts.
If both `-sequence-vocab` and `-eos-token-id` are supplied, their EOS IDs must
match.

For a nucleotide checkpoint, pass the preparation artifact and a raw sequence
prompt. Mixlab still prints token IDs and additionally prints the decoded
sequence:

```bash
./mixlab -mode generate \
  -config examples/nucleotide_dna_causal_tiny.json \
  -safetensors-load weights.safetensors \
  -sequence-vocab data/reference-dna/nucleotide_vocab.json \
  -prompt sequence:ACGTN
```

`generate` is causal next-token generation. It does not consume
`training.diffusion.steps_per_block`, `confidence_threshold`, or `commit_floor`;
those drive `generate-diffusion` instead.

Eligible TTT-MLP stacks automatically use persistent inner-model and Q/K
convolution state instead of replaying the full prefix. The cached path supports
`ttt_mlp` mixed with pointwise `swiglu`, `geglu`, or `mlp` blocks and can stream
beyond configured `seq_len`. See [TTT-MLP stateful inference](ttt-mlp-stateful-inference.md).
Bulk generation reuses one TTT-MLP session and creates a fresh, explicitly
closed adaptation state for every sample.

## `generate-diffusion`

Generate from a `training.objective: "block_diffusion"` checkpoint, a hybrid
checkpoint with `training.hybrid_secondary_objective: "block_diffusion"`, or a
multihead checkpoint with a configured `training.diffusion_head`.

```bash
./mixlab -mode generate-diffusion \
  -config examples/block_diffusion_tiny.json \
  -safetensors-load weights.safetensors \
  -prompt token_ids:0,1,2 \
  -max-tokens 16
```

Starting from the prompt, mixlab appends a block of mask tokens, then runs up to
`steps_per_block` denoising passes per block, committing positions whose
predicted probability clears `confidence_threshold` and at least `commit_floor`
positions per pass so every block completes.

| Flag | Description |
|------|-------------|
| `-config` | Required. Must set `training.objective: "block_diffusion"`, hybrid with `hybrid_secondary_objective: "block_diffusion"`, or multihead with a block-diffusion `diffusion_head`. |
| `-safetensors-load` | Required. Trained block-diffusion weights. |
| `-prompt` | Prompt token IDs in `token_ids:0,1,2` form. |
| `-max-tokens` | Maximum generated tokens, capped at `seq_len - prompt`. Default: `256`. |
| `-diffusion-steps-per-block` | Override `training.diffusion.steps_per_block`. `0` uses the config. |
| `-diffusion-confidence-threshold` | Override `training.diffusion.confidence_threshold` when explicitly set. |
| `-diffusion-commit-floor` | Override `training.diffusion.commit_floor`. `0` uses the config. |
| `-diffusion-temperature` | Diffusion sampling temperature. `0` keeps deterministic argmax. |
| `-diffusion-top-k` | Diffusion top-k cutoff when `-diffusion-temperature > 0`. `0` disables the cutoff. |
| `-diffusion-trace-out` | Write sampler telemetry JSONL, one denoising pass per line. |

By default, diffusion sampling is deterministic argmax over unresolved
positions. `-temperature` and `-top-k` still apply only to causal `generate`;
use the diffusion-specific temperature/top-k flags for stochastic diffusion
sampling.

For `training.objective: "multihead"`, `generate-diffusion` reads
`head_<diffusion_head>_logits` and ignores scorer-only export heads.
