# Factorial Race v5

Maverick-driven critique of v4 race: config 04's win cannot be attributed
to "n-gram offload" because it changed model_dim, depth, n-gram vocab,
n-gram dim, and weight-sharing pattern simultaneously. Race v5 isolates
which axis is responsible.

## Two-by-two factorial

|                       | Old n-grams (3072, 80) | Full n-grams (4096, 384) |
|-----------------------|------------------------|--------------------------|
| **320d × 10 blocks**  | v4 config 01 (4.4391)  | A (this race)            |
| **384d × 8 blocks**   | B (this race)          | v4 config 04 (4.3577)    |

If A wins ≈ config 04: full n-grams are doing the work.
If B wins ≈ config 04: width/depth swap is doing the work.
If both fall in between: interaction effect; need orthogonal
implementations of NgramRes (head-level) to disentangle.

## Multi-seed verification

To distinguish single-seed noise from real effect, repeat config 01 and
config 04 at seeds 1337 and 2025 (seed 42 already from v4):

- S01_*_seed1337, S01_*_seed2025: re-runs of v4 config 01
- S04_*_seed1337, S04_*_seed2025: re-runs of v4 config 04

The 04-vs-01 gap from v4 was -0.0814 nats. If the multi-seed mean delta
stays comfortably negative with low std, the effect is real (not seed
noise). If the per-seed deltas span zero, we don't have a real win.

## Total runs

6 new training runs in this race (A, B, plus 2 seeds × 2 configs).
v4 already provided 2 seeds=42 data points (config 01 and config 04).

## Run

All configs share vocab_size=8192, seq_len=1024, batch_tokens=4096, so
arch_race accepts them as one race:

```bash
./mixlab -mode arch_race \
    -configs experiments/factorial_v5/ \
    -train 'data/fineweb_sp8192/train_*.bin'
```

Estimated wall on M4 Max: ~22min × 6 = ~2h15m.

## Outcome interpretation

Three possible verdicts:

1. **Full n-grams matter** (A ≈ config 04 ≫ config 01 ≈ B): n-gram
   embedding capacity is the load-bearing axis. Next: implement proper
   NgramRes (head-level mixture) to test whether the embedding form is
   optimal or just a weak approximation.

2. **Width/depth/sharing matter** (B ≈ config 04 ≫ config 01 ≈ A):
   the win was about parameter allocation, not n-grams. Next: scan
   model_dim and depth at fixed n-grams.

3. **Interaction effect** (A and B both fall between 01 and 04):
   neither axis alone explains the gain. Need to test the cross.

Three multi-seed outcomes:

1. **Effect holds** (mean S04 - S01 stays around -0.08, std ≪ |effect|):
   real win, ready for publication-grade study.

2. **Effect shrinks but stays negative** (mean delta -0.02 to -0.08):
   real but smaller than v4 single-seed suggested.

3. **Effect collapses** (delta crosses zero or std > |delta|):
   v4 was seed noise. Stop and rethink.
