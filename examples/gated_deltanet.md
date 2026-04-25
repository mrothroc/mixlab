# gated_deltanet -- Simplified GatedDeltaNet Demo

GatedDeltaNet is a linear-time recurrent architecture that mixes tokens through gated state updates rather than quadratic attention; this example follows the general gated-state idea from Yang et al. (2025) but is intentionally simplified for `mixlab`'s current custom-op surface.

This demo uses a compact residual block built around `scan_tv`: normalize `x`, project a value stream and a time-varying gate, run the gated recurrence `h[t] = gate[t] * h[t-1] + (1 - gate[t]) * value[t]`, project back to `D`, and add the residual. A standard `swiglu` block follows each recurrent block for channel mixing.

The custom ops map to the architecture like this:
- `rmsnorm`: pre-norm on the block input.
- `matmul` + `sigmoid`: build the recurrence gate and beta-style auxiliary gates.
- `scan_tv`: main vector-state recurrence used for the trainable residual path.
- `outer`: forms per-token key/value interaction tensors for an auxiliary KV-style state path.
- `matrix_scan`: rolls those outer-product updates through a matrix-valued gated recurrence.

`outer` and `matrix_scan` are included to demonstrate the new opcodes in a working model, but the block is still a demo, not a faithful paper reproduction: there is no per-timestep matrix-vector readout, multi-head parameterization, or exact DeltaNet update rule.

Train it with:

```bash
./mixlab -mode arch -config examples/gated_deltanet.json -train 'data/example/train_*.bin'
```
