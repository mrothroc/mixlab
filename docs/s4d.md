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

## Runtime boundary

- Native token and `linear_frames` classification are supported.
- The FFT path is differentiable through MLX.
- Bidirectional FFT output remains `D` channels; directional contributions
  are summed before GELU/dropout/GLU.
- Normal recurrence/weight sharing can reuse S4D block weights when using
  stateless RMSNorm/LayerNorm. BatchNorm rejects recurrence in v1.
- Hugging Face export and stateful generation are not supported in v1 and
  fail explicitly.
- `mode count` includes kernel materialization and FFT work in the estimate.

For continuous input, see [Continuous sequence input](continuous-input.md) and
the [`continuous_s4d_classification_tiny.json`](../examples/continuous_s4d_classification_tiny.json)
example. The pinned LRA Image recipe is
[`continuous_s4d_lra_image_reference.json`](../examples/continuous_s4d_lra_image_reference.json);
the full 200k-step result remains a hardware-intensive acceptance run.
