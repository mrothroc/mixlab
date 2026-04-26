# Hybrid Race v1

Six configs testing architectural-axis hybrids. All share the same training
budget (2000 steps, 4096 batch tokens, model_dim=256, seq_len=1024, seed=42)
so val NLL is directly comparable.

## Hypotheses

| Config | Tests |
|--------|-------|
| `01_baseline_dense` | Control: standard attention stack with QK-Gain + parallel residual + Muon |
| `02_depth_recurrent` | Does weight sharing improve val NLL at fixed FLOPs? |
| `03_mamba_only` | Can pure linear attention match quadratic at this scale? |
| `04_sandwich_mamba` | Capability injection: attention boundaries + linear middle |
| `05_depth_rec_mamba` | **Novel:** depth-recurrent linear attention (no public combo) |
| `06_baseline_swa` | What does SWA alone buy on top of the standard stack? |

## Run

```bash
# Prepare data first if not already done
python3 scripts/download_fineweb.py --output data/fineweb_sp1024 \
    --vocab-size 1024 --max-docs 50000

# Race
./mixlab -mode arch_race \
    -configs experiments/hybrid_race_v1/ \
    -train 'data/fineweb_sp1024/train_*.bin'
```

## Reading results

Results print sorted by val loss (lower is better). Look for:
- The control (`01`) is your baseline — everything is measured against it
- Compare `02` vs `01` for the depth-recurrence effect
- Compare `04` vs `01` and `04` vs `03` for the sandwich pattern's value
- Compare `05` vs `03` for depth-recurrence applied to linear attention
- `06` vs `01` isolates the SWA contribution

## Caveat

2000 steps on Shakespeare-scale or small FineWeb subsets is enough to rank
configs but not enough to claim absolute BPB numbers. For headline results,
re-run the top 2-3 configs at 30K steps with multiple seeds.
