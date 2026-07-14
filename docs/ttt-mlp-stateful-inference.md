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

## Hugging Face Cache

`export-hf` supports the same cache-safe causal composition: one or more
`ttt_mlp` blocks plus pointwise `swiglu`, `geglu`, or `mlp` blocks. The
generated `MixlabForCausalLM` returns the request state in `past_key_values`:

```python
prefix = model(input_ids=prompt_ids, use_cache=True)
continued = model(
    input_ids=next_ids,
    past_key_values=prefix.past_key_values,
    use_cache=True,
)
```

Each cache entry contains the packed inner MLP, partial chunk gradient,
three-token Q/K convolution history, and chunk offset. Do not share one cache
between independent requests. Mixed attention or SSM trunks fail export until
their own cache states can be composed without replay.

When no cache is requested, the exported module uses the same chunk-level dual
form as native full-sequence execution instead of replaying the online update
one token at a time. This is the normal path for zero-shot scoring, `AutoModel`
feature extraction, and Hugging Face fine-tuning. Cached prefill and decode keep
the online path above so split continuation remains exactly state-compatible.

Right-padded stateless batches are supported and parity-tested. Left-padded
batches are not: padding before a sequence would otherwise update the TTT state
and shift chunk-relative positions. Bucket by length or configure the tokenizer
for right padding.

For short CPU evaluation workloads, use a small PyTorch intra-op thread count.
On the M1 Max release fixture, `OMP_NUM_THREADS=1` reduced a D384, 14-mixer,
43-token stateless forward from 412 ms with eight threads to 63.9 ms. The module
does not set this globally because doing so from library code would affect the
host application's unrelated PyTorch work.

During downstream fine-tuning, exported TTT blocks checkpoint the complete
stateless dual scan by default. Backward recomputes that scan one block at a
time, preserving full meta-gradients while avoiding retention of every inner
update across all layers. `gradient_checkpointing_disable()` opts out when
backward speed matters more than memory; `gradient_checkpointing_enable()`
restores the default. Cache-bearing inference never uses activation
checkpointing.

## CUDA Runtime

Linux CUDA builds use a precompiled causal depthwise-convolution primitive for
stateful Q/K history and retain the differentiable MLX recurrence for the
nonlinear update. Set `MIXLAB_TTT_MLP_DISABLE_CUDA_PRIMITIVE=1` only to debug
the portable fallback. The Docker build requires the TTT kernel in the
embedded CUDA registry.

Run CUDA correctness and long-context performance coverage on an NVIDIA host:

```bash
go test -tags mlx ./gpu ./train -run 'TTTMLP|CUDAGraph' -count=1 -v
MIXLAB_TTT_MLP_LONG_BENCH=1 \
go test -tags mlx ./train \
  -run TestTTTMLPInferenceLongContextBenchmark -count=1 -v
```

CUDA performance numbers are intentionally not inferred from the Apple run;
record them from the target GPU before making backend-specific throughput
claims.

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
