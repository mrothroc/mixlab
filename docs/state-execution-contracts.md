# State And Execution Contracts

Execution contracts describe existing native behavior. They do not allocate
state, introduce a new executor, enable new model combinations, or change
weights, checkpoints, training-resume files, or numerical equations.

## Inspect Without A GPU

```bash
mixlab -mode inspect-contract \
  -config arch/testdata/execution_contracts/ttt.json \
  -contract-profile native-ttt-stateful -contract-require streaming

mixlab -mode inspect-contract -config arch/testdata/execution_contracts/recurrent.json
mixlab -mode inspect-contract -config examples/grid_two_stage_2.json
```

The command works in a build without MLX and needs no dataset or checkpoint.
Stdout contains one JSON document; warnings and structured errors use stderr.
Valid partial reports exit zero. Invalid configs, invalid descriptors,
unsupported profiles, and unsatisfied requirements exit nonzero.

| Flag | Default | Meaning |
|---|---|---|
| `-contract-profile` | `native-full` | `native-full` describes the native full training graph (including declared targets); `native-ttt-stateful` binds the existing inference graph at batch one, one token, offset zero. |
| `-contract-require` | empty | Optionally require `complete`, `streaming`, `refinement`, `save-restore`, or `clone`. Checks support; never enables it. |

## Coverage And Support

| Model/path | Coverage | Cross-call streaming |
|---|---|---|
| Causal TTT with permitted pointwise blocks, stateful profile | Complete | Existing TTT session only |
| TTT full training/classification | Complete | Unsupported in the full profile |
| `gated_linear_ssm` full evaluation | Complete | Unsupported; every scan starts at zero |
| `swiglu`, `geglu`, `mlp` | Explicitly stateless block providers | No standalone session added |
| Dense grid prediction/supervision | Complete separate grid contract | No cross-record state |
| Other blocks or execution paths without providers | Partial | Unknown, never assumed supported |

Existing restrictions still apply. TTT stateful inference rejects classification,
recurrence, weight sharing, grouped execution and uncached token mixers using its
existing validator. Registry coverage does not make those combinations runnable.
Parallel paths not integrated with a provider remain partial. Refinement loops,
inference-state save/restore, and cloning are not implemented.
Configs with mutable model buffers (for example BatchNorm running statistics)
also remain partial until their buffer effects have a provider.

## Report Format

`mixlab.execution_contract.v1` uses explicit typed fields:

- `blocks`: configured `block_id`, execution-site `owner_id`, and
  `parameter_group_id` are separate identities. Reused weights do not alias state.
  `described:false` means unknown, not stateless.
- `parameters`: checkpoint indices, shapes, group ownership, frozen/buffer flags
  and graph reachability. Frozen and unreachable are different properties.
- `program.representations`: dtype, physical shape, named logical axes and packing.
  Sequence logits describe `[batch,time]` packed into their first physical axis.
  Grid boundaries retain NHWC; equal element counts do not establish compatibility.
- `program.states`: tensor or host-control slots, owners, initialization, bindings,
  lifetime, reset, gradient policy, growth and compatibility dependencies.
- `boundaries`: ordered execution stages and explicit state read/write sets.
  Observe/update/emit may be fused and are not independently callable.
- `program.selected_output` and `detached_edges`: grid prediction provenance and
  actual stop-gradient boundaries. Target masks select supervised loss positions;
  they are not input-validity masks or evidence for adaptation.
- `capabilities`: supported, unsupported, or unknown, each with a reason.
- `memory`: tensor payload, conceptual carry, host controls and peak device memory
  are separate. Unknown values are `null` with reasons, never guessed as zero.

IR dtype constants are reused: float32 and int32 storage metadata describe
physical boundary tensors, independently of the reported compute precision.
Contract data is host-only and is not lowered to MLX or saved in checkpoints.

### TTT State

The packed inner MLP uses the existing ordered `w1,b1,w2,b2` parameter slices.
For H heads, head width d and hidden width h, `S = H * (2*d*h + h + d)`.
Batch-one inference owns FP32 MLP and gradient tensors `[1,S]`, convolution
history `[1,2,3,D]`, and a host integer offset in `[0,chunk_size)`.

For D=8, H=2, d=4, h=16: S=296. Packed tensors require 2,368 bytes and
convolution history 192 bytes: **2,560 bytes per block**. Host objects, weights,
cached program variants, outputs and allocator overhead are additional.
This is tensor payload, not measured peak GPU memory.

The gradient accumulator belongs to the online inner update, not the outer
optimizer. Full training differentiates through the existing implementation;
inference state is detached between calls. A successful fragment commits only
after evaluation and output readback. Earlier fragments of a multi-fragment
prefill may remain committed if a later fragment fails. Reset frees old handles
before allocating replacements and does not guarantee rollback on allocation
failure. Session close closes its states; state close is idempotent. The existing
same-goroutine/thread-affinity rules remain; this API adds no concurrency promise.
Each token adds its inner gradient before its query is evaluated. The query uses
the coefficient-scaled accumulated update including that token. At the algorithmic
chunk boundary, that query state becomes the inner MLP and the accumulator clears.
Transport fragment boundaries do not redefine those algorithmic chunks.

`Prefill` retains all requested logits; `PrefillLast` retains only the last row.
Neither output retention nor the session's program cache is recurrent payload.

### Recurrent And Spatial Paths

`gated_linear_ssm` is the simplified gated mixer, not canonical Mamba-3. Its
scan is `h[t] = sigmoid(decay)*h[t-1] + (1-sigmoid(decay))*x[t]`, with zero
initial carry on every evaluation. The conceptual `[B,Dscan]` carry is not an
estimate of the implementation's FFT/sequential/autodiff peak allocations.

Grid contracts reuse resolved graph shapes, selected-output reachability and
weight metadata. Intermediate activations remain within one feed-forward call.
Two independently weighted stages, even with a detached edge, are not a generic
shared-weight refinement loop. Stage labels are not inferred from weight names.

## API And Extension Rules

`arch.DescribeModelContract` builds one native graph; `DescribeProgramContract`
uses an already-built graph. `ValidateModelContract` validates schema, identities,
initialization and effects; `ValidateProgramContract` checks actual declarations,
scan bindings and detached edges. `RequireExecution` rejects partial contracts
for managed requests. `TTTMLPInferenceSession.Contract()` returns a deep snapshot,
without device readback or changing offsets, state or statistics.

To add a provider:

1. Register `BlockRegistration.DescribeContract`; never replace absence with an
   empty-state default.
2. Use resolved builder layout information, not a second shape calculation.
3. Bind stateful ops and validate them during graph construction.
4. Declare ownership, initialization, update/emit timing, resets and effects.
5. Test unsupported combinations, mutation rejection, weight sharing, numerical
   parity and resource estimates before claiming a capability.

Identity-keyed collections are canonicalized; initialization slices and execution
stages retain semantic order. `schema_digest` is SHA-256 over semantic fields,
excluding prose, diagnostics, source revision, checkpoint identity and config
digest. `effective_config_digest` separately hashes normalized config values,
excluding load-time source-path bookkeeping. Neither identifies model weights.
Checkpoint identity is `null` because inspection reads no checkpoint.

A matching `schema_digest` alone does not make state restorable: restore would
also have to match model content and runtime representation. Cross-backend state
portability is not promised. Shared state lifecycle, cross-call recurrent carry,
state serialization and copies are not implemented.
