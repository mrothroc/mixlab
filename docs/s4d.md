# S4D diagonal state-space block

`s4d` is Mixlab's linear time-invariant diagonal state-space mixer. It is
intended for long, uniformly sampled sequences where input-dependent selective
state updates are not necessarily the right inductive bias.

```json
{
  "type": "s4d",
  "state_size": 64,
  "init": "s4d-lin",
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

The v1 implementation follows the official minimal S4D-Lin
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
S4D-LegS, bidirectional S4D, and incremental recurrent-state inference remain
separate follow-up work rather than silent approximations.

Global `norm_type: "batchnorm"` is supported for fixed-shape native
classification. It computes channel statistics over batch and time, stores
running mean/variance in native checkpoints, and uses those buffers for
validation/evaluation. Padded records, recurrence, SWA, and HF export are
rejected in this first release.

## Optimization

S4D's continuous-time parameters and complex kernel coefficients use the
scalar/vector Adam optimizer group even when the rest of a model uses Muon.
This also disables weight decay for those parameters. Use `training.scalar_lr`
to set their learning rate independently:

```json
{
  "training": {
    "optimizer": "muon",
    "lr": 0.001,
    "scalar_lr": 0.0001
  }
}
```

## Runtime boundary

- Native token and `linear_frames` classification are supported.
- The FFT path is differentiable through MLX.
- Normal recurrence/weight sharing can reuse S4D block weights when using
  stateless RMSNorm/LayerNorm. BatchNorm rejects recurrence in v1.
- Hugging Face export and stateful generation are not supported in v1 and
  fail explicitly.
- `mode count` includes kernel materialization and FFT work in the estimate.

For continuous input, see [Continuous sequence input](continuous-input.md) and
the [`continuous_s4d_classification_tiny.json`](../examples/continuous_s4d_classification_tiny.json)
example.
