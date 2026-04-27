# GatedDeltaNet Parity Validation

mixlab's `gated_deltanet` block type is a faithful port of the GatedDeltaNet
architecture from `flash-linear-attention` (FLA). This directory contains the
test infrastructure used to validate that faithfulness numerically.

## Faithfulness gates (all must pass)

mixlab's CI exercises three parity tests against an FLA reference implementation:

| Test | Asserts | Tolerance | Measured |
|------|---------|-----------|----------|
| `TestGatedDeltaNetFLAParity` | Single-block forward output matches FLA | L∞ ≤ 1e-3 | 7.4e-05 |
| `TestGatedDeltaNetFLAParityMultiLayer` | 4-layer stacked forward matches FLA | L∞ ≤ 1e-3 | 4.9e-05 |
| `TestGatedDeltaNetFLAGradParity` | Per-weight gradients match FLA's autograd | L∞ ≤ 1e-3 | 5.0e-07 |

The reference is `fla.ops.gated_delta_rule.naive.naive_recurrent_gated_delta_rule`
(triton-free, sequential), itself cross-validated against a hand-written
recurrence to L∞ = 6e-08.

## Recurrence

mixlab implements the gated delta-rule recurrence exactly as FLA does:

```
S_t   = decay_t · S_{t-1}            # apply gate to prior state
pred  = k_t^T · S_t                  # delta-rule prediction from prior state
err   = v_t − pred                   # error-correcting term (the *delta* rule)
S_t   = S_t + outer(k_t, β_t · err)  # update with scaled error
o_t   = q_t · S_t                    # readout
```

This is implemented as a fused IR op (`OP_GATED_DELTA_SCAN`) rather than
composed from `Outer + MatrixScan`, because the `pred` term requires per-step
access to `S_{t-1}` which a vanilla scan cannot express.

The gate `decay_t` is computed FLA-style:
```
dt        = softplus(dt_bias + a_proj(x))   # per-token, per-head positive scalar
decay_t   = exp(−exp(A_log) · dt)           # in (0, 1]
```
where `A_log` is per-head learned (initialized as `log(U(0, 16))`) and
`dt_bias` is per-head learned (inverse-softplus of log-uniform `dt ∈ [1e-3, 1e-1]`).

## Initialization

mixlab matches FLA's default init for every gated_deltanet weight:
- Linear projections (`wq`, `w_kv` or `wk`/`wv`, `w_a`, `w_beta`, `w_out_gate`,
  `wo`): PyTorch `Linear` default = `kaiming_uniform_(a=sqrt(5))` ≡
  `uniform(±1/sqrt(fan_in))`.
- Depthwise convs (`q_conv`, `k_conv`, `v_conv`): PyTorch `Conv1d` default =
  `uniform(±1/sqrt(K)) = uniform(±0.5)` for `kernel_size=4`.
- `A_log`: `log(U(0, 16))`.
- `dt_bias`: inverse-softplus of `log-uniform(1e-3, 1e-1)`.
- Norm scales: `1.0`.

## Usage

JSON config:
```json
{
  "type": "gated_deltanet",
  "heads": 4,
  "d_k": 64,
  "d_v": 128,
  "kv_share": true,
  "weight_group": "gdn_a"
}
```

`d_v` defaults to `2 * d_k` for the "Wider" variant. `kv_share=true` ties K
and V into a single wide projection where K is sliced from the first `d_k`
channels of each V head — matching resouer's `K_KVShare_Wider` from PR #1791.

## Reproducing the parity tests

Requires Python 3.13 with `torch` and `numpy` (no triton needed). FLA's naive
reference is imported directly from the installed package, bypassing the
triton-dependent `fla.ops.__init__`.

```bash
# Activate env with torch
source path/to/venv/bin/activate

# Generate fixtures
python experiments/gated_deltanet_test/dump_fla_reference_pure.py \
    --output /tmp/fla_ref.npz
python experiments/gated_deltanet_test/dump_fla_reference_pure.py \
    --output /tmp/fla_ref_4l.npz --layers 4
python experiments/gated_deltanet_test/dump_fla_grad_reference.py \
    --output /tmp/fla_grad_ref.npz

# Run parity tests
FLA_REFERENCE=/tmp/fla_ref.npz \
    CGO_ENABLED=1 go test -tags mlx ./train -run TestGatedDeltaNetFLAParity -v
FLA_REFERENCE=/tmp/fla_ref_4l.npz \
    CGO_ENABLED=1 go test -tags mlx ./train -run TestGatedDeltaNetFLAParityMultiLayer -v
FLA_GRAD_REFERENCE=/tmp/fla_grad_ref.npz \
    CGO_ENABLED=1 go test -tags mlx ./train -run TestGatedDeltaNetFLAGradParity -v
```

Without the env vars, the tests are skipped (so CI doesn't require torch).

## Training-recipe notes

GatedDeltaNet trains differently from mamba3 / standard attention. The 200-step
control on `data/example/train_*.bin` lands at:

| Architecture | Optimizer recipe | Final train | Final val |
|---|---|---|---|
| GatedDeltaNet (mixlab default) | Muon for matrices, default schedule | 5.55 | 5.82 |
| GatedDeltaNet (matched recipe) | All-AdamW, constant lr=1e-3, wd=0.04 | 4.81 | 4.99 |
| PyTorch reference (faithful) | Same matched recipe | 4.94 | 4.87 |
| mamba3 (matched recipe) | Same matched recipe | 4.56 | 4.63 |

Two takeaways:
1. The default mixlab recipe (Muon-on-everything + 100-step warmup in a 200-step
   run) is hostile to GatedDeltaNet at small scales. For fair comparisons,
   match the recipe explicitly.
2. Under the matched recipe, mixlab's GatedDeltaNet trains within 0.13 nat of
   the verified PyTorch reference — confirming the implementation is faithful
   end-to-end, not just at single-step parity.

mamba3 still trains marginally better on this small example (0.25 nat under
matched recipe), within seed-noise band. Don't read architectural conclusions
from a single 200-step matched-param run on `data/example`; for that, use
`arch_race` with multi-seed verification on real shards.
