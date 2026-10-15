# Local Learner Engine

`NewLocalLearnerSession` is an internal integration foundation for a fixed-cohort
DiLoCo learner. It is not a launch mode: DiLoCo CLI execution remains disabled
until the coordinator and managed learner application are integrated.

The session takes validated causal/AdamW config, a local immutable dataset,
immutable learner identity, and the complete initial global model. It creates
no MLX collective group and receives no network client, workload credentials,
allocation state, or coordinator object.

## Lifecycle

- Construction freezes config and binds the graph, layout, schedule, resolved
  precision, character lookup, and dataset identity. Unsupported unkeyed
  `random_normal` graphs reject rather than claiming reproducible replay.
- `Train(ctx, attempts)` runs exactly K local microbatches per optimizer attempt.
  Numerator-weighted accumulation, finite checks, clipping, AdamW update, and
  transactional commit reuse the existing staged trainer. A skip or partial
  microstep failure requires explicit restore; extra batches never compensate
  for a skipped update. Cancellation is observed between microsteps, not by
  interrupting an in-flight GPU operation.
- `InstallGlobal` validates every tensor's name, shape, length and finiteness,
  materializes all native arrays, and swaps the complete model once. Private
  optimizer moments and counters remain intact. Installation during incomplete
  accumulation is rejected by the native bridge.
- `Snapshot` returns owned model/optimizer arrays, optimizer counters, schedule
  position, effective counts, sampler cursor and stable RNG identity at an
  optimizer boundary. The next schedule position is `Attempt`; K does not
  change its meaning. Mutable validation-driven schedules are unsupported.
- `Restore` checks identity, config/graph/layout, precision, finite moments and
  counters. It initializes a replacement engine before replacing the live one;
  a malformed tensor, moment or sampler cursor cannot partially overwrite it.
- `EvaluateGlobal` runs fixed, validated token/target batches in a separate
  dropout-inactive engine. Record-backed programs require their binary loss
  masks; stream programs reject record masks. Training state is untouched.
- `Close` destroys the engine on its owning OS thread. All MLX operations run
  on that pinned thread even when serialized callers use different goroutines.

The learner application owns durable files, start/completion markers, round
binding and recovery authorization. This API supplies complete local state;
it does not supply disk publication or a recovery service.

## Replay

`data.LearnerLoader` reuses the counter-based partitioned sampler without
fabricating a DDP rank/group. Workers share the sample permutation seed and
consume distinct strided ordinals. Snapshots bind the immutable cohort index,
partition and worker; cursor validation ties consumed rows to optimizer work.
One learner is supported without changing the DDP loader's two-rank minimum.

Dropout uses the immutable run/worker/partition identity and seed, then the
logical optimizer attempt, microstep and op ordinal. Hosting attempts,
credential generations, reconnections and wall time are absent. Initial model
weights come from the supplied global model, never a worker-specific seed.

## Verification

Portable tests cover partitioning, direct restore, identity/RNG separation,
skip/cancellation handling and transactional rejection. Metal tests cover
dropout replay, record masks, finite model/moment state, whole-model replacement,
isolated evaluation and multi-round accumulation against a full-batch reference.
