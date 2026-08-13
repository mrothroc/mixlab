# S4D Sobolev beta control experiment

These four configs change only the Sobolev beta control under test:

| Config | Control |
|---|---|
| `unbounded.json` | Reference defaults: trainable channel beta, LR 0.01, no decay |
| `bounded.json` | Trainable channel beta with `bounds: [-2,2]` |
| `fixed.json` | Channel beta pinned at zero |
| `lower_lr.json` | Trainable channel beta with LR 0.001 |

They are small mechanics/smoke experiments, not accuracy claims. For a causal
comparison, copy the same one-field changes into the full target recipe and
hold seed, data split, initialization, optimizer, schedule, and budget fixed.
Compare validation accuracy together with the emitted `s4d_sobolev`
telemetry. Accuracy improvement is not required; the purpose is to determine
whether controlling beta changes training or evaluation behavior.
