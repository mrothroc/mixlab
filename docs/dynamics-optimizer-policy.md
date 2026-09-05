# Recurrent dynamics optimizer policy

Structured parameters carry decay policy in their block's weight metadata.
`ForceNoDecay` takes precedence over both `weight_decay_policy: "matrix_only"`
and `"all"`, including explicit nonzero per-group decay settings. It disables
the decay term for that tensor; it does not freeze the tensor or change its
optimizer, learning rate, initialization, checkpoint name, or shape.

## Which runs were affected

Rank decides what the default `matrix_only` policy does, so the exemptions
differ in reach:

| Parameter | Rank | Decayed before this fix |
|---|---|---|
| `mamba3-canonical` `A_log` | 2 (`[inner, state_size]`) | **Yes, under the default policy** and under `all` |
| `mamba3-canonical` `dt_bias` | 1 | Only under `all` |
| `gated_deltanet` `A_log`, `dt_bias` | 1 (`[heads]`) | Only under `all` |

Mamba3's `A_log` is the consequential one: being rank two, it fell on the decay
side of `matrix_only` and so decayed in ordinary runs that never set
`weight_decay_policy`. Runs of the other three tensors were only affected if
they opted into `all`.

## Block audit

| Block | Explicit decay exemptions | Other dynamics-related parameters |
|---|---|---|
| `mamba3-canonical` | `A_log`, `dt_bias` | `w_dt_low/high`, `w_lambda_low/high`, `w_theta_low/high`, `w_B`, and `w_C` remain ordinary learned projections. This Mixlab block has no separate `D` skip tensor. |
| `gated_deltanet` | `A_log`, `dt_bias` | Input-dependent gate projections retain ordinary matrix decay. |
| `s4d` | `s4d_log_dt`, `s4d_log_A_real`, `s4d_A_imag`, and trainable `s4d_B_real/imag` | C/D follow the configured policy; the Sobolev beta has its own explicit decay control. See [S4D](s4d.md). |
| `hgrn2` | No additional hard exemptions | Forget factors are computed from an ordinary `w_f` projection; there is no standalone learned log-state or timestep tensor. |
| `mlstm` | No additional hard exemptions | Input/forget projections are ordinary matrices; `b_i`/`b_f` are vectors, excluded by `matrix_only` but included by `all`. |
| `retnet` | No additional hard exemptions | Mixlab's learned `decay` vector follows the configured policy. The reference uses a fixed decay buffer, so its exclusion is not a directly transferable optimizer rule. |
| `ttt_mlp` | No additional hard exemptions | Inner MLP initial weights and learned learning-rate projections remain trainable outer-model parameters. Runtime adapted state is not an outer optimizer parameter. Inner bias tensors with rank two follow the current matrix policy; no blanket reference-parity claim is made for their grouping. |
| `rwkv`, `legacy_mamba`, `gated_linear_ssm` | No additional hard exemptions | Learned scalar/vector decay controls follow the configured policy; these Mixlab variants are not treated as reference Mamba parameterizations. |

The default `matrix_only` policy already excludes scalar-class and vector
parameters, including normalization scales. Absence of `ForceNoDecay` therefore
does not imply a parameter decays by default. `all` intentionally overrides
those ordinary exclusions, but cannot override the explicit exemptions above.
Bidirectional Mamba3 and Gated DeltaNet reuse the same protected weight metadata.

## Reference basis

The upstream [Mamba implementation](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py)
marks its log-parameterized A and D skip parameter as no-decay. Mixlab retains
that A parameterization in its canonical Mamba3 block. Current upstream
[Mamba3](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba3.py)
instead computes data-dependent A and marks `dt_bias` and D as no-decay.
These are different parameter layouts: upstream's fused input projection does
not establish an exemption for Mixlab's learned dt/lambda/theta projections.

[Gated DeltaNet](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/gated_deltanet.py)
explicitly marks both `A_log` and `dt_bias` as no-decay. The sibling audit also
compared [HGRN2](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/hgrn2.py),
[mLSTM gates](https://github.com/NX-AI/xlstm/blob/main/xlstm/blocks/mlstm/cell.py),
[RetNet's fixed decay buffer](https://github.com/microsoft/torchscale/blob/main/torchscale/architecture/retnet.py),
and [TTT-MLP](https://github.com/test-time-training/ttt-lm-pytorch/blob/main/ttt.py).
This audit covers explicit structured-dynamics exemptions, not replication of
every upstream training recipe's bias/norm optimizer grouping.

## Learning rates and failure diagnosis

Mamba3 and Gated DeltaNet keep existing class-based LR settings. In particular,
Mamba3's rank-two `A_log` uses the matrix group and `dt_bias` uses the scalar
group. A separate Mamba3 `state_lr` is not added by this fix: the checked upstream
parameter declarations do not prescribe a universal reduced LR. Any new
state-only LR group needs an explicit tensor selection and scheduler/resume
contract, rather than an inferred default copied from S4D.

The optimizer circuit breaker's `state_nonfinite` count includes candidate
weights **and optimizer moments**. Finite incoming gradients can still overflow
when squared for second moments. Consequently, `gradient_nonfinite=0` with
`state_nonfinite>0` is not a unique signature of incorrect decay.

For `A = -exp(A_log)` and a positive timestep, the discrete transition magnitude
is `exp(-dt * exp(A_log))`. Decaying `A_log` changes timescales but does not by
itself move this magnitude outside the unit circle. The decay-policy defect and
the cause of a reported non-finite run must be verified separately.

Regression coverage includes metadata checks, actual trainer group construction
under both policies and all supported training optimizers, and native AdamW/LAMB
steps with zero gradients proving protected tensors do not decay while ordinary
weights do. A small-model smoke does not establish long-sequence Speech Commands
accuracy or CUDA stability: acceptance of that report still requires the exact
dataset/configuration rerun past warmup and through held-out evaluation.
