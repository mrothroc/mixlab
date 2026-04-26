# Native Race v3

V2 measured "is mamba better than attention with attention's tokenizer, attention's
optimizer, attention's seq_len, attention's regularization?" — answer: no. V3
measures a different question: each architecture in its **native regime**.

## What's different per config

| Config | Vocab | seq_len | Optimizer | LR | SWA | QAT | Goal |
|--------|-------|---------|-----------|-----|-----|-----|------|
| `01_attn_native` | SP-8192 | 1024 | Muon | 1e-3 | yes | int6@7K | Leader stack (race v2 winner) |
| `02_mamba_handicapped` | SP-8192 | 1024 | AdamW | 3e-4 | no | no | Mamba in attn regime, mamba-appropriate optim |
| `03_mamba_native_T4096` | SP-1024 | 4096 | AdamW | 3e-4 | no | no | Mamba's natural regime |
| `04_mamba_native_T2048` | SP-1024 | 2048 | AdamW | 3e-4 | no | no | Intermediate seq_len |

All run 10K steps, batch_tokens=4096, seed=42. Mamba configs use d=384 to
match parameter budget with attention SP-8192 d=320 (which spends ~5M params
on embed/head alone).

## Why the differences are appropriate, not concessions

- **Optimizer**: Muon's Newton-Schulz orthogonalization was tuned for
  transformer matrix updates. SSM weights (decay vector, conv weight) have
  different gradient geometry. AdamW is the right baseline for SSMs in
  every published mamba result.
- **No SWA**: SWA averages weight values. For attention weights, that's
  approximately quadratic-loss-surface averaging. For SSM decay vectors,
  the dynamics are nonlinear in W (decay enters as `sigmoid(W)` or
  `exp(-exp(W))`). Averaging two SSMs with different decay rates gives a
  third system, not the average.
- **No QAT**: QAT timing is tuned around attention's late-training dynamics.
  Mamba may need different schedule; null hypothesis is no QAT.
- **seq_len differences**: Attention is O(T²·D), mamba is O(T·D²). Letting
  attention run long-context would handicap it for the same reason
  attention's quadratic cost forces SP-8192. Each architecture goes to its
  natural seq_len.
- **Vocab differences**: Attention's quadratic compute makes it pay for long
  T, so the leaders trade vocab parameters for shorter T (SP-8192 saves T).
  Mamba's linear compute removes that pressure — it can run small vocab
  (SP-1024, more tokens per byte) and convert the saved embed/head
  parameters into more SSM body capacity. This is the ByT5 / MambaByte
  insight.

## Run

```bash
# Both data shards already prepared from earlier races
./mixlab -mode arch_race \
    -configs experiments/native_race_v3/ \
    -train 'data/fineweb_sp8192/train_*.bin'
```

NOTE: arch_race uses one train pattern. Configs 01/02 expect SP-8192 shards;
configs 03/04 expect SP-1024. mixlab matches `vocab_size` against shard
contents, so we'll need two race invocations or a different orchestration.

### Two-pass approach

```bash
# Pass 1: attention + mamba_handicapped (both SP-8192)
mkdir -p /tmp/race_v3_sp8192
cp experiments/native_race_v3/01_attn_native.json /tmp/race_v3_sp8192/
cp experiments/native_race_v3/02_mamba_handicapped.json /tmp/race_v3_sp8192/
./mixlab -mode arch_race -configs /tmp/race_v3_sp8192/ \
    -train 'data/fineweb_sp8192/train_*.bin' 2>&1 | tee /tmp/race_v3_pass1.log

# Pass 2: mamba native configs (both SP-1024)
mkdir -p /tmp/race_v3_sp1024
cp experiments/native_race_v3/03_mamba_native_T4096.json /tmp/race_v3_sp1024/
cp experiments/native_race_v3/04_mamba_native_T2048.json /tmp/race_v3_sp1024/
./mixlab -mode arch_race -configs /tmp/race_v3_sp1024/ \
    -train 'data/fineweb_sp1024/train_*.bin' 2>&1 | tee /tmp/race_v3_pass2.log
```

## How to read results

Compare across all four BPB-equivalent. Important: token NLL is **not**
comparable across vocabularies — fewer tokens per byte means each token
carries more information, so val NLL/token won't be apples-to-apples.

To get a comparable BPB, run `-mode eval -logprobs-out` on the held-out
val set for each model and compute byte-level BPB using the per-config
tokenizer. That eval pass uses the prepared LUTs that the prepare script
already wrote.

Loose intuition: at SP-1024 each token covers ~3-4 bytes; at SP-8192,
~5-7 bytes. So an apparent NLL gap of 0.5 nats/token between SP-1024 and
SP-8192 is roughly cancelled if the lower-vocab model has 1.5x more
tokens per byte.

## Hypothesis going in

If mamba's v2 loss was driven by handicap rather than architecture:
- 02_mamba_handicapped should beat v2's 02_mamba_optimized (mamba-correct optim)
- 03 / 04 should beat 02 (long context + small vocab pays off)
- Best mamba config approaches or beats 01 in BPB

If mamba's v2 loss was inherent:
- 02 ≈ 03 ≈ 04, all behind 01
