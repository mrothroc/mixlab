# Hugging Face Export

`export-hf` writes a Hugging Face custom-code directory from a Mixlab JSON config and a Mixlab safetensors checkpoint. Token-model checkpoints export `AutoModel`, `AutoModelForCausalLM`, and `AutoModelForSequenceClassification`; masked and hybrid checkpoints also export `AutoModelForMaskedLM`. Native continuous S4D classifiers export only `AutoModel` and `AutoModelForSequenceClassification`. For masked-capable token exports, `AutoModel` and sequence classification use the bidirectional encoder backbone while `AutoModelForCausalLM` stays causal.

```bash
mixlab -mode export-hf \
  -config examples/plain_3L.json \
  -safetensors-load runs/plain_3L/weights.safetensors \
  -export-dir runs/plain_3L/hf \
  -tokenizer-path data/example/tokenizer.json
```

For training runs with SWA/EMA averaging enabled, pass the averaged checkpoint when you want the exported Hugging Face model to use averaged weights:

```bash
mixlab -mode export-hf \
  -config examples/swa_ema_tiny.json \
  -safetensors-load runs/swa_ema_tiny/model.swa.safetensors \
  -export-dir runs/swa_ema_tiny/hf \
  -tokenizer-path data/example/tokenizer.json
```

The exported directory contains:

- `config.json` with `auto_map` entries for `AutoConfig`, `AutoModel`, `AutoModelForCausalLM`, and `AutoModelForSequenceClassification`; masked-capable exports also include `AutoModelForMaskedLM`
- `configuration_mixlab.py`, `modeling_mixlab.py`, `pooling_mixlab.py`, `ttt_mlp_mixlab.py`, `mamba3_mixlab.py`, and `s4d_mixlab.py` static maintained templates
- `model.safetensors` with Hugging Face state-dict keys
- `weight_map.json` mapping Mixlab `w{index}_{name}` tensors to Hugging Face tensor names
- `tokenizer.json`, plus `tokenizer_config.json` and `special_tokens_map.json`
- `char_features.bin` when token-level character feature embeddings are enabled

Continuous `linear_frames` exports do not contain tokenizer or special-token
files because their public input is a float tensor rather than token IDs.

## Special Token IDs

Export resolves BOS, EOS, and PAD IDs in this order:

1. `tokenizer_config.json`, `special_tokens_map.json`, and special added tokens in `tokenizer.json`.
2. Explicit `-bos-token-id`, `-eos-token-id`, and `-pad-token-id` flags.
3. `special_token_ids` in a `mixlab.dataset.json` beside the tokenizer.
4. Legacy `training.example_framing.bos_id` and `eos_id`.

Every resolved ID must be inside the model vocabulary. Export fails instead of
writing invalid metadata. When an ID remains unknown, export prints a warning
and leaves it unset. Native GPT-2 exports write unresolved BOS, EOS, and PAD
fields as JSON `null`; this prevents Transformers from restoring GPT-2's legacy
`50256` defaults for small custom vocabularies.

The IDs are stored in `config.json`, which Transformers uses to initialize the
model's generation configuration. A separate `generation_config.json` is not
written because it would duplicate the same metadata.

Load it with:

```python
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

model = AutoModelForCausalLM.from_pretrained("runs/plain_3L/hf", trust_remote_code=True)
backbone = AutoModel.from_pretrained("runs/plain_3L/hf", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("runs/plain_3L/hf", trust_remote_code=True)

# Present for MLM/MNTP/hybrid exports.
masked = AutoModelForMaskedLM.from_pretrained("runs/masked_model/hf", trust_remote_code=True)

# For an LM checkpoint, classification weights are freshly initialized for
# downstream fine-tuning.
classifier = AutoModelForSequenceClassification.from_pretrained(
    "runs/plain_3L/hf",
    trust_remote_code=True,
    num_labels=3,
)

# For a native training.objective="classification" checkpoint, num_labels,
# pooling, dropout, and the trained classifier weights come from the export.
classifier = AutoModelForSequenceClassification.from_pretrained(
    "runs/dna_mamba3_cls/hf",
    trust_remote_code=True,
)
```

`trust_remote_code=True` executes the Python modeling files from the export directory. Only use it for directories you created or reviewed.

## Sequence Classification

`MixlabForSequenceClassification` owns padding-aware pooling and a linear
classifier. For LM/pretraining checkpoints the classifier is freshly initialized;
Hugging Face reports its weights as new, and downstream fine-tuning trains and
saves them normally. For a native `training.objective: "classification"`
checkpoint, the exporter writes the trained `head_classifier_proj` and
`head_classifier_bias` as `classifier.weight` and `classifier.bias`, and carries
the checkpoint's `num_labels`, pooling mode, and classifier dropout into
`config.json`. Loading that directory therefore reproduces the native classifier
instead of replacing its task head.

Standard Hugging Face single-label, multi-label, and regression losses are
selected from `num_labels`, label dtype, and `problem_type`.
`classifier_dropout` defaults to the exported `hidden_dropout` for a fresh
downstream head; native classifier exports preserve their configured effective
dropout.

The exporter derives `sequence_classification_pooling` from the concrete `AutoModel`
backbone:

- `last` for causal, TTT-MLP, and canonical Mamba-3 backbones, gathering the final non-padding token in each row;
- `mean` for masked/bidirectional backbones, averaging only non-padding tokens.

Native classification exports use the explicit/defaulted
`training.classification.pooling` from the training config.

### Continuous S4D classifiers

Native `linear_frames` classifiers whose sequential backbone contains only
`s4d` blocks export as fixed-shape custom-code models. They preserve the input
projection and optional bias/LayerNorm, all S4D state parameters, optional
trainable B, grouped `n_ssm`, ZOH or bilinear discretization, bidirectionality,
the optional GLU output transform, post-residual normalization, optional final
norm, mean/last pooling, and the trained classifier.

```bash
mixlab -mode export-hf \
  -config examples/continuous_s4d_lra_image_reference.json \
  -safetensors-load runs/s4d/model.safetensors \
  -export-dir runs/s4d/hf
```

```python
import torch
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "runs/s4d/hf", trust_remote_code=True
).eval()
frames = torch.from_numpy(features).float()  # [B, seq_len, feature_dim]
with torch.no_grad():
    logits = model(input_values=frames).logits
```

This first export contract is deliberately fixed-shape. `input_values` must be
`[B, config.seq_len, input_adapter.feature_dim]`; padding or a mask containing
zero is rejected because it would change the S4D convolution contract.
BatchNorm checkpoints remain native-only because their running buffers do not
yet have export parity coverage. Continuous exports do not accept
`-tokenizer-path` or special-token flags. Stateful S4D generation is not added.

Mixed causal/bidirectional graphs are ambiguous. They continue to export normally,
but loading the classification class requires an explicit policy:

```python
classifier = AutoModelForSequenceClassification.from_pretrained(
    "runs/mixed/hf",
    trust_remote_code=True,
    num_labels=2,
    sequence_classification_pooling="mean",
)
```

For a batch with more than one row, pass a two-dimensional `attention_mask`; the head
raises instead of guessing sequence lengths. Rows containing no real tokens also
raise. `last` and `mean` select the correct positions for left- or right-padded masks,
but this does not make every backbone invariant to changing padding side. TTT-MLP
still requires right padding because padding affects its recurrent state, and learned
absolute positions or token-neighbor feature channels retain their documented input
semantics.

## Native GPT-2 Export

Set top-level `hf_export_format: "gpt2"` when a config is intentionally strict GPT-2-compatible and should export as a native Hugging Face `GPT2LMHeadModel` directory instead of a custom Mixlab directory. Native GPT-2 export writes `model_type: "gpt2"`, `architectures: ["GPT2LMHeadModel"]`, packed `attn.c_attn` QKV tensors, GPT-2 `transformer.wte/wpe/h.*` names, and tied `lm_head.weight`.

The exporter rejects configs that are not exactly representable as GPT-2. The accepted v1 shape is a causal, sequential `plain` stack with `positional_embedding: "learned_absolute"`, affine `norm_type: "layernorm"`, `tie_embeddings: true`, `attn_bias: true`, `ffn_pre_norm: true`, `ffn_bias: true`, `ffn_activation: "gelu_new"` or `"gelu"`, no RoPE/relative attention fields, and no Mixlab training-only or architecture extras.

```json
{
  "hf_export_format": "gpt2",
  "positional_embedding": "learned_absolute",
  "max_positions": 1024,
  "norm_type": "layernorm",
  "tie_embeddings": true,
  "blocks": [
    {"type": "plain", "heads": 12, "attention_mask": "causal", "attn_bias": true, "ffn_activation": "gelu_new", "ffn_pre_norm": true, "ffn_bias": true}
  ]
}
```

## Native-vs-HF Parity Mode

After exporting, use `parity` mode to compare the native MLX forward against the exported Hugging Face directory on real eval shards:

```bash
mixlab -mode parity \
  -config examples/plain_3L.json \
  -safetensors-load runs/plain_3L/weights.safetensors \
  -hf runs/plain_3L/hf \
  -train 'data/example/val_*.bin'
```

`parity` mode loads the native checkpoint, scores the shard glob passed to `-train` directly, and runs the exported directory through Hugging Face on the same token stream. It reports native loss, Hugging Face loss, the loss delta, and max absolute logit difference for a bounded sample. The logit sample defaults to one native eval batch; increase it with `-parity-logit-tokens`, set the loss gate with `-parity-loss-threshold` (legacy alias: `-threshold`), set the logit gate with `-max-logit-diff`, or disable the logit gate with `-max-logit-diff 0`.

The Python checker requires the HF parity dependencies from `requirements-hf.txt` (`torch`, `transformers`, and `safetensors`). Use `-parity-python` or `HF_PARITY_PYTHON` when those packages live in a non-default Python environment.

## Supported Coverage

HF export supports next-token and masked-LM checkpoints using sequential blocks:

- `plain` attention with Mixlab's default adjacent-pair RoPE convention or explicit `rope_convention: "half_rotation"`, including partial `rope_dims`
- `positional_embedding: "learned_absolute"` and `"none"` in the custom Mixlab template; learned absolute exports a `position_embeddings.weight` table
- configurable core norms through `norm_type`, `norm_eps`, `norm_affine`, `norm_placement`, and `ffn_internal_norm` for sequential `plain`, `swiglu`, `geglu`, and `mlp` blocks
- `plain` FFN tails with the default `silu` activation, non-gated `ffn_activation: "gelu"` / `"gelu_new"`, optional `ffn_pre_norm` / `ffn_bias`, or gated `ffn_activation: "geglu"` / `"swiglu"`
- grouped-query attention through `kv_heads`
- `plain` attention projection biases through `attn_bias`
- `plain` attention value gates through `attn_value_gate`
- `plain` attention post-norm placement through `attn_post_norm`, including explicit `before_outproj`
- learned per-head-dimension `qk_norm`
- learned per-head `qk_gain`
- `xsa` attention-output projection
- `sparse_attn_gate` per-head attention output gates
- `attention_mask` values `causal`, `bidirectional`, and `none`
- causal `window_size` sliding attention
- DeBERTa/GPT-BERT-style `relative_attention: "deberta_p2c_c2p"` on `plain` blocks, including log-bucketed `q-k` relative positions, optional `relative_attention_parameterization: "shared_qk_reuse"`, and optional shared-table `relative_attention_embedding_norm: "layernorm"`
- GPT-BERT-style `layer_aggregation: "dwa"` on supported sequential `plain`/`swiglu`/`geglu`/`mlp`/`moe` stacks
- `mlm_head: "bert"` for `AutoModelForMaskedLM`; the masked head exports the BERT transform stack with tied embedding output weight and separate output bias while the causal class still loads the materialized `lm_head_weight`
- `swiglu`, `geglu`, and `mlp` FFN blocks, including MLP activation variants `silu`, `gelu`, `relu`, and `leaky_relu_sq`
- sequential `moe` blocks with a linear router, top-k token routing, and `swiglu`, `geglu`, or `mlp` experts
- causal `ttt_mlp` stacks composed only with pointwise `swiglu`, `geglu`, or `mlp`; the exported model carries request-owned recurrent state through `past_key_values`
- causal or native-classification `mamba3-canonical` stacks composed only with pointwise `swiglu`, `geglu`, or `mlp`; the maintained PyTorch path mirrors the complete canonical scan for non-cached forwards
- fixed-shape native continuous classifiers composed only from `s4d` blocks; the maintained PyTorch path mirrors grouped/trainable-B ZOH or bilinear kernels, bidirectionality, GLU output transforms, and post-residual norms
- embedding-time `char`, `bigram`, and `trigram` feature channels
- tied embeddings; the exporter materializes `lm_head_weight = embed_tokens.weight.T` for Hugging Face consumers
- data2vec-trained checkpoints; the training-only predictor weights and `training.data2vec` spec are stripped, exporting the student/base inference model
- distillation-trained checkpoints; fixed-teacher configs are training-only and are stripped, exporting the student/base inference model
- `training.objective: "mlm"` and `"mntp"` configs; these export both the standard causal head and a masked-LM head whose plain attention blocks are bidirectional
- `training.objective: "hybrid"` configs with MLM/MNTP secondary objectives for causal and masked evaluation; the causal head uses causal plain attention while `AutoModel` and the masked-LM head use bidirectional plain attention
- `training.objective: "classification"` configs on supported backbones; the trained native classifier head and its pooling/dropout metadata are exported
- `hf_export_format: "gpt2"` for strict GPT-2-compatible configs, exported without custom Mixlab Python code

The generated `modeling_mixlab.py` consumes Hugging Face `attention_mask` in `AutoModel`, `AutoModelForCausalLM`, and `AutoModelForMaskedLM`, so padded batches mask pad-token keys instead of letting padding leak into hidden states.

### TTT-MLP execution paths

Exported TTT-MLP blocks use a vectorized chunk dual form for ordinary
stateless `AutoModel` and `AutoModelForCausalLM` calls, including downstream
fine-tuning. Calls that request or provide `past_key_values` retain the exact
online recurrence because the cache includes partial-chunk gradients and
convolution history.

The stateless fine-tuning path is compatible with a full-graph PyTorch compile.
Compile the complete downstream model, not each TTT block separately:

```python
model = AutoModelForSequenceClassification.from_pretrained(
    export_dir,
    trust_remote_code=True,
    num_labels=num_labels,
)
model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
```

The first forward/backward compiles the graph and must be excluded from
throughput measurements. Keep batch shapes stable after warm-up; bucketing and
padding sequence lengths to a `chunk_size` multiple reduce recompilation and
avoid a separate partial-chunk graph shape. Padding must still be on the right.
The exported runtime keeps both the TTT recurrence check and sequence-pooling
empty-row check inside compiled graphs, so compilation does not weaken those
correctness guards.

Variable-length TTT batches must be right padded. Real-token prefix logits are
then identical to scoring each row separately; left padding would advance the
recurrent state before the first real token. This is stricter than attention-only
exports, where `attention_mask` removes padding keys directly.

A left- or interior-padded `attention_mask` raises `ValueError` rather than
returning plausible but wrong logits — note that Hugging Face's batched
`generate()` convention for decoder-only models is `padding_side="left"`, so an
eval harness carrying that default will fail loudly here instead of silently
scoring against a corrupted recurrent state. Set `padding_side="right"` or bucket
sequences by length.

**Final-token pooling needs a second change.** Downstream heads (GLUE-style
sequence classification, `take_final` pooling) commonly left-pad precisely so that
`hidden[:, -1]` is the last real token. Switching such a pipeline to right padding
is necessary but *not sufficient*: `hidden[:, -1]` then lands on a pad for every
row shorter than the longest, which degrades accuracy silently. Gather the last
real position per row instead:

```python
lengths = attention_mask.sum(-1) - 1                     # index of last real token
pooled = hidden[torch.arange(hidden.size(0)), lengths]
```

Mean pooling has the same requirement — weight by `attention_mask` so pads do not
enter the average.

Short TTT sequences contain many small grouped matrix operations. On CPU,
large default PyTorch thread pools can cost more than the operations themselves.
For zero-shot scoring of short examples, start the evaluator with one intra-op
thread and tune upward only for longer sequences or larger batches:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python evaluate.py ...
```

The exported model does not change PyTorch's process-global thread settings.
MPS uses standard supported PyTorch operations in the stateless path, but CPU
is often faster for short batch-one scoring because MPS launch overhead is not
amortized.

TTT blocks activation-checkpoint the entire stateless dual scan by default in
training mode when gradients are enabled. This preserves the exact full
meta-gradient while recomputing one block's scan during backward instead of
retaining every inner-update activation across the trunk. Evaluation,
`torch.no_grad()`, and cached generation do not enter the checkpoint path.

The standard Hugging Face controls are supported. The memory-bounded default is
recommended for downstream fine-tuning; callers with sufficient device memory
can trade memory for backward speed explicitly:

```python
model.gradient_checkpointing_disable()  # faster backward, higher peak memory
model.gradient_checkpointing_enable()   # restore memory-bounded backward
```

The opt-in CUDA release gate measures complete classifier optimizer steps,
including forward, checkpointed backward, and AdamW update. It reports eager
and compiled tokens/second, compilation time, peak allocated memory, and
best-effort GPU utilization:

```bash
HF_TTT_MLP_TRAIN_PERF=1 \
HF_PARITY_PYTHON=/path/to/hf-venv/bin/python \
go test ./train -run TestExportHFTTTMLPTrainingPerformance -count=1 -v -timeout=30m
```

The default gate requires compiled steady-state throughput to be no slower
than eager. Set `HF_TTT_MLP_MIN_COMPILE_SPEEDUP` to a hardware-specific ratio
for release infrastructure. GPU utilization is diagnostic rather than a hard
cross-hardware threshold.

### Canonical Mamba-3 execution path

`mamba3-canonical` export uses a maintained PyTorch reference implementation of
Mixlab's fused block: internal RMSNorm, optional short causal depthwise
convolution, low-rank delta/lambda/theta projections, normalized grouped B/C,
cumulative complex-pair rotation, exponential-trapezoidal recurrence, SiLU
gate, output projection, and residual. It accepts stacks composed from
`mamba3-canonical` plus pointwise `swiglu`, `geglu`, or `mlp` blocks.

The exported scan is sequential over sequence positions and vectorized over
batch, channels, and state. It is intended for portable inference and
classification parity, not as a replacement training kernel for Mixlab's
native Metal/CUDA paths. Cached/streaming continuation is not implemented:
ordinary full-sequence `AutoModel`, `AutoModelForCausalLM`, and
`AutoModelForSequenceClassification` forwards work, while a request for
`past_key_values`/`use_cache` fails explicitly. `attention_mask` treats masked
positions as recurrent no-ops, so padding-aware classification works without
letting pad tokens advance state.

Tokenizer artifacts must come from an explicit `-tokenizer-path`, or from `tokenizer.json` next to the config/checkpoint. If the tokenizer source is missing or unreachable, export fails before writing an incomplete Hugging Face directory. Mixlab writes `tokenizer_config.json` and `special_tokens_map.json` by merging any source sidecars with special-token metadata derived from `tokenizer.json`, and writes matching `pad/eos/bos/unk_token_id` fields to `config.json` when the tokens are present. For masked-capable checkpoints (`mlm`/`mntp`, or `hybrid` with `hybrid_clm_fraction < 1`), it also sets the tokenizer `mask_token`/`mask_token_id` — resolved from `training.mlm_mask_token_id` against the tokenizer vocab — so masked/MNTP eval works without manually patching the tokenizer.

When `char_vocab_size > 0`, `export-hf` also requires `char_features.bin` next to the config, checkpoint, or tokenizer source. The file is copied into the HF directory and loaded by the exported Python model so generated token IDs use the same token-id lookup as Mixlab inference.

## Support Matrix

The detailed support matrix is maintained in [hf-export-support-matrix.md](hf-export-support-matrix.md). It distinguishes supported, gated, unsupported, and training-only features.

Unsupported features fail fast with an error naming the field or block type. The current advanced export path intentionally gates HGRN2, mLSTM, legacy/simplified Mamba-family blocks, RetNet/RWKV, `gated_deltanet`, `custom` blocks, `kv_source`, recurrence, U-Net, parallel residual, backout, MTP, block diffusion, and first-byte masked loss. Canonical Mamba-3 and TTT-MLP each reject mixed attention/SSM trunks outside their parity-covered pointwise compositions.

These guards are part of the export contract: a missing feature should be visible as an actionable error, not as a Hugging Face model that loads but computes different logits.

## Parity Tests

Two layers of parity coverage exist:

1. **Go oracle and static-reference parity** (default Go suite; PyTorch reference fixture gated with `HF_PARITY=1`). Verifies metadata, tokenizer handling, weight mapping, unsupported-feature errors, and deterministic native-vs-HF fixtures. Coverage includes GEGLU/MLP, `plain` gated FFN tails, configurable LayerNorm/no-affine export metadata, BERT-style masked-LM heads, GQA, attention post-norm placement, `qk_norm`, `qk_gain`, XSA, sparse attention gates, masks, causal windowing, DeBERTa relative attention, shared relative embedding LayerNorm, MoE routing and expert variants, feature channels, hybrid causal export semantics, gated recurrent policies, trained native-classifier mapping, and the canonical Mamba-3 PyTorch block against the independent scalar full-block fixture (tolerance `2e-5`).

2. **Native-vs-Python parity** (`TestExportHFNativePythonParity`, gated on `HF_PARITY=1` + MLX + the Python toolchain). This is the load-bearing FR-1 check: it exports deterministic trained-magnitude fixtures, loads each through the real Hugging Face auto classes, runs the *actual* embedded Python forward, checks padded-tokenizer batching where supported, and asserts the CausalLM logits agree with the *actual* native MLX forward (max per-logit abs diff < 1e-3, mean next-token loss diff < 1e-4). Every custom export case also loads `AutoModelForSequenceClassification`, compares its result with an identity-head pooling oracle, checks padded-batch versus independent-row behavior where the backbone is row-independent, and exercises missing-mask/empty-row failures. One case covers classification fine-tune loss, save, and reload. Cases include partial/full RoPE, `qk_norm`, sigmoid SwiGLU, tanh-approx GELU, `plain` gated FFN tails, GQA + `qk_gain` + sliding window, DeBERTa relative attention, XSA + sparse attention gates, top-k MoE (geglu/mlp experts), bigram/trigram/char feature channels, TTT-MLP full-forward plus split recurrent-state parity, and canonical Mamba-3 full-forward parity. `TestExportHFNativeMamba3ClassificationPythonParity` separately compares the real trained native classifier head, mask-weighted mean pooling, and exported HF logits.

Python/HF parity dependencies are declared in `requirements-hf.txt` (verified against torch 2.12 / transformers 5.10). The gated `.github/workflows/hf-parity.yml` workflow installs those dependencies and uses `macos-latest` with Homebrew MLX so the native-vs-Python check runs on an MLX-capable runner, keeping the default Linux CI lightweight.

### Template note: dynamic buffers under `from_pretrained`

`modeling_mixlab.py` computes its causal mask and RoPE tables in `forward()` and lazy-loads the char lookup, rather than caching them as `__init__` buffers. `from_pretrained` initializes custom models on the meta device, where value-dependent non-persistent buffers built from `torch.ones`/`torch.arange` materialize as zeros — which would silently disable masking/rotation or drop the char channel. The parity test above is what makes that class of regression visible.
