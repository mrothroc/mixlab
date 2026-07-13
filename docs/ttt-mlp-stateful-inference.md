# TTT-MLP Stateful Inference

Mixlab's native TTT-MLP runtime can retain online inner-model state across
prefill and decode calls. The cache contains the per-head MLP, partial-chunk
gradient accumulators, Q/K convolution history, and each block's chunk offset.
It is an inference cache only: checkpoints and training weight layouts are
unchanged.

## Generation

`-mode generate` selects the cached path automatically when every sequence
mixer in the model has a complete cache contract. Today that means one or more
`ttt_mlp` blocks composed with pointwise `swiglu`, `geglu`, or `mlp` blocks.
Architectures containing attention or another recurrent mixer continue using
full-prefix replay.

```bash
mixlab -mode generate \
  -config examples/ttt_mlp_tiny.json \
  -safetensors-load checkpoint.safetensors \
  -prompt token_ids:1,7,9 \
  -max-tokens 128
```

The cached TTT path is not capped by `seq_len`: it uses chunk-relative RoPE and
bounded recurrent state. The normal replay generator retains its existing
`seq_len` limit.

## Go API

```go
session, err := train.NewTTTMLPInferenceSession(configPath, checkpointPath)
if err != nil { return err }
defer session.Close()

state, err := session.NewState()
if err != nil { return err }
defer state.Close()

logits, err := session.PrefillLast(state, promptTokens)
logits, err = session.Decode(state, nextToken)
stats := session.Stats()
```

`Prefill` returns logits for every input token. `PrefillLast` processes the
same stream but retains only the final row on the host, which keeps host memory
bounded for long streams. Each call may stop in the middle of a TTT chunk; the
next call continues from the pending gradient state. `Reset` restores the
checkpoint's initial inner MLP and clears convolution history.

Every request must own a distinct `TTTMLPInferenceState`. Passing a state to a
different session, using it after close, or continuing after its session is
closed returns an error. State replacement is transactional: failed forwards
leave the previous handles active.

## Apple Runtime Verification

The gated benchmark is reproducible with:

```bash
MIXLAB_TTT_MLP_LONG_BENCH=1 \
go test -tags mlx ./train \
  -run TestTTTMLPInferenceLongContextBenchmark -count=1 -v
```

Results recorded on 2026-07-13 using an Apple M1 Max, float32, and the shipped
64-dim/4-head/1024-vocab example shape:

| Context | Steady prefill tok/s | Active MLX memory |
|---:|---:|---:|
| 512 | 2,852 | 0.62 MiB |
| 2,048 | 2,703 | 0.76 MiB |
| 8,192 | 2,948 | 0.82 MiB |
| 32,768 | 2,920 | 0.57 MiB |

Peak MLX memory was 1.87 MiB and did not grow with context. Steady cached
token decode measured 1.58 ms/token (634 tok/s). The first uncached compile was
1.72 seconds on a cold run; a subsequent run with the platform shader cache
warm measured 27.9 ms. These figures characterize runtime mechanics, not model
quality or a production-size architecture.

The integration fixture also checks a streaming adaptation trace: a prefix
changes the same probe token's logits relative to a fresh state, reset restores
the fresh result, and two request states remain isolated. `Stats` reports token
count, native evaluations, cached program variants, and live request states.
