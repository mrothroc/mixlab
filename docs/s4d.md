# S4D diagonal state-space block

`s4d` is Mixlab's linear time-invariant diagonal state-space mixer. It is
intended for long, uniformly sampled sequences where input-dependent selective
state updates are not necessarily the right inductive bias.

```json
{
  "type": "s4d",
  "state_size": 64,
  "init": "s4d-lin",
  "n_ssm": 2,
  "bidirectional": true,
  "discretization": "bilinear",
  "trainable_b": true,
  "state_lr": 0.001,
  "dt_min": 0.001,
  "dt_max": 0.1,
  "output_transform": "glu"
}
```

The block is a token mixer. Pair it with a channel mixer such as `swiglu` or
`geglu`:

```json
{
  "blocks": [
    {"type":"s4d","state_size":64},
    {"type":"swiglu"}
  ]
}
```

## Reference contract

The default implementation follows the official minimal S4D-Lin
parameterization:

- diagonal continuous poles `A_n = -1/2 + i*pi*n`
- one learned log time step per model channel, initialized log-uniformly
- learned complex `C`; fixed `B=1` absorbed into `C`
- zero-order-hold discretization
- learned direct term `D`
- real output reconstructed from conjugate pole pairs

Training materializes the closed-form causal convolution kernel and evaluates
it with an FFT. The recurrent form is retained internally for forward and
gradient parity tests; it is not a public execution-mode switch or an
incremental inference cache in v1.

Only `init: "s4d-lin"` is accepted. `output_transform: "glu"` adds the
reference-style GELU, dropout, `D -> 2D` projection, and GLU output path.
Omitting the field preserves the earlier compact GELU-only block exactly.
The additive reference path supports shared A/B groups through `n_ssm`,
trainable complex B, bilinear discretization, and exact bidirectional
length-`2T` FFT convolution. Bidirectionality shares A/B/dt and learns only
the backward C independently. S4D-LegS and incremental recurrent-state
inference remain separate work rather than silent approximations.

## Frequency tuning

S4D-Lin can opt into the two frequency-bias controls from
[Yu et al., ICLR 2025](https://arxiv.org/abs/2410.02035):

```json
{
  "type": "s4d",
  "freq_scale": 3.0,
  "sobolev_filter": {
    "beta_init": 0.0,
    "learning_rate": 0.01
  }
}
```

`freq_scale` changes only the imaginary-pole initialization:
`A_n = -1/2 + i * freq_scale * pi * n`. It defaults to `1.0`; finite values
greater than zero are accepted. Existing checkpoints store the initialized and
subsequently trained poles, so the scale is not applied again when loading.

`sobolev_filter` adds one learned exponent per model feature. The object form
defaults to `beta_init: 0.0` and `learning_rate: 0.01`; `true` is shorthand for
those defaults, while omission or `false` disables it. The exponent has no
weight decay. For FFT bin `k` and transform length `L_fft`, the convolution
spectrum is multiplied by
`(1 + k/L_fft)^beta`. The direct `D*x` contribution is intentionally not
filtered, matching the paper's supplemental implementation. Zero-initialized
beta makes the initial forward exactly the ordinary S4D forward while allowing
training to change the frequency sensitivity.

The two controls are independent. Use baseline, `freq_scale` only, Sobolev
only, and both as a matched-budget four-arm ablation. The paper's LRA Image
result used `freq_scale: 3` and learned beta, but reported results should not be
attributed to these controls unless the baseline and tuned runs use identical
training budgets.

For sequence classification parity with post-norm references, use
`norm_placement: "post_residual"` to compute `Norm(x + Dropout(F(x)))`, and
set `final_norm: false` when the reference has no model-level final norm.
Top-level `tie_dropout: true` samples `[B,1,D]` masks for S4D internal and
residual dropout.

Global `norm_type: "batchnorm"` is supported for fixed-shape native
classification. It computes channel statistics over batch and time, stores
running mean/variance in native checkpoints, and uses those buffers for
validation/evaluation. Padded records, recurrence, SWA, and HF export are
rejected in this first release.

## Optimization

With `state_lr` omitted, S4D keeps the legacy scalar/vector optimizer grouping.
With `state_lr` set, A/B receive the specified LR with no decay, dt uses the
global LR with no decay, and C/D use the global LR. Set
`weight_decay_policy: "all"` when matching an optimizer that decays ordinary
vectors, biases, and norm parameters:

```json
{
  "training": {
    "optimizer": "adamw",
    "lr": 0.01,
    "weight_decay": 0.05,
    "weight_decay_policy": "all"
  }
}
```

The pinned LRA reference uses both `state_lr: 0.001` and
`weight_decay_policy: "all"`. The default `"matrix_only"` policy remains valid
for other S4D recipes, but it does not reproduce that reference optimizer:
ordinary biases, norms, and scalar/vector C/D parameters are not decayed.
An enabled Sobolev filter always uses its own `learning_rate` and no decay,
independently of `state_lr`.

For low-dimensional continuous signals, also keep `input_adapter.norm` at
`"none"` unless projection-scale invariance is intentional. Post-projection
LayerNorm is exactly sign-only at initialization for `feature_dim: 1` while
the projection bias is zero.

## Runtime boundary

- Native token and `linear_frames` classification are supported.
- The FFT path is differentiable through MLX.
- The Sobolev filter is a full-sequence FFT operator. The internal recurrent
  parity path remains available for ordinary S4D but is rejected when the
  filter is enabled; it is not approximated as a streaming filter.
- Bidirectional FFT output remains `D` channels; directional contributions
  are summed before GELU/dropout/GLU.
- Normal recurrence/weight sharing can reuse S4D block weights when using
  stateless RMSNorm/LayerNorm. BatchNorm rejects recurrence in v1.
- Hugging Face export supports fixed-shape native `linear_frames`
  classification checkpoints whose sequential stack contains only S4D blocks.
  The exported custom model accepts float `input_values`, preserves the trained
  classifier, and rejects padding. Token-model S4D export, BatchNorm buffers,
  and stateful generation remain gated. Frequency-scaled poles and learned
  Sobolev exponents are preserved by the fixed-shape export.
- `mode count` includes kernel materialization and FFT work in the estimate.

For continuous input, see [Continuous sequence input](continuous-input.md) and
the [`continuous_s4d_classification_tiny.json`](../examples/continuous_s4d_classification_tiny.json)
example. The pinned LRA Image recipe is
[`continuous_s4d_lra_image_reference.json`](../examples/continuous_s4d_lra_image_reference.json);
it follows the reference's 200 epochs as 180,000 optimizer updates while retaining
the separately configured 200,000-step cosine horizon and 1,000-step warmup. Its
ordinary input, S4D output, and classifier affine layers use
`weight_init: "pytorch_linear"`; S4D state parameters retain their dedicated
`s4d-lin` initializers. The full run remains a hardware-intensive acceptance run.
For a small frequency-tuning surface example, see
[`continuous_s4d_frequency_tuned_tiny.json`](../examples/continuous_s4d_frequency_tuned_tiny.json).
