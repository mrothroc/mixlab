# Optimized Race v2

Tests whether mamba's win in race v1 holds at scale (10K steps, SP-8192) and
under the full competition-style optimization stack.

## Configs

| Config | Architecture | Stack |
|--------|--------------|-------|
| `01_attn_leader_clone` | Depth-recurrent attention (3 unique pairs, 10 blocks) | QK-Gain 5.25, parallel_residual, Muon, late QAT (int6), SWA, softcap, bigrams |
| `02_mamba_optimized` | Mamba3 stack (5 SSM + 5 FFN) | Muon, late QAT (int6), SWA, softcap, bigrams |
| `03_mamba_depth_rec_optimized` | Depth-recurrent mamba (3 unique SSM + 3 unique FFN, 10 blocks) | Muon, late QAT (int6), SWA, softcap, bigrams |
| `04_mamba_minimal` | Mamba3 stack | AdamW only — control measuring stack value |

All run 10000 steps at model_dim=320, vocab=8192, seq_len=1024,
batch_tokens=4096, seed=42.

## Prerequisites

```bash
python3 scripts/download_fineweb.py --output data/fineweb_sp8192 \
    --vocab-size 8192 --max-docs 50000
```

## Run

```bash
./mixlab -mode arch_race \
    -configs experiments/optimized_race_v2/ \
    -train 'data/fineweb_sp8192/train_*.bin'
```

Estimated wall-clock: 25-50 minutes per config on M4 Max, 2-3.5 hours total.

## What this tests

1. **Does the v1 mamba win persist?** Race v1 showed mamba beat attention at
   2K steps / SP-1024 / minimal stack. Race v2 asks: does the architectural
   advantage hold at 10K steps and SP-8192 with the full leaderboard stack?
2. **Does depth recurrence still pay on mamba at scale?** Config 03 vs 02.
3. **How much of the leader stack is architecture vs training tricks?**
   Config 02 vs 04.

## Caveats

- Single seed. For headline claims, repeat with seeds 1337 and 2025.
- `val` numbers reported during training use live weights, not SWA-averaged
  weights. To see the SWA effect, run `-mode eval` after training with the
  exported safetensors.
- `qat_start: 7000` activates int6 QAT in the last 30% of training. The
  reported live val NLL will tick up briefly when QAT engages — this is
  expected and recoverable.
- 50K-doc FineWeb subset (~95M tokens) is enough for ranking but is a
  fraction of the 10B used by leaderboard records.
