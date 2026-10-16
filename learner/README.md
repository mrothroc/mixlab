# Managed Learner Runtime

This package owns the private round lifecycle for a fixed-cohort DiLoCo worker.
It is composed by the training executable, not by the node-management process.
The numerical engine is a port; this package has no MLX, remote TLS, admission,
credential, or coordinator implementation dependency.

## Durable Ordering

1. Verify the immutable assignment against the local launch authority and fetch
   the hash-verified global model.
2. Restore the prior private optimizer/sampler state, then install the complete
   global model. A new run alone may initialize fresh private state.
3. Publish the start snapshot and commit its hash with the coordinator before
   consuming an inner batch. Round-zero replacement requires that commitment.
4. Run exactly H optimizer attempts, each using the engine's configured K
   microsteps. Any skipped or incomplete attempt fails the window.
5. Publish complete private state, delta, exact update, and completion marker
   together before uploading/submitting. Persist the returned exact receipt.

The protected `head.json` CAS is the local commit point. Model, optimizer,
sampler, schedule and metadata parts are content-addressed and verified before
restore. The completion marker binds the update and completed private state.
An interrupted window restores its start, including sampler and keyed RNG
counters. A completed window retries the identical update without training.
An accepted update can recover its receipt even after the assignment expires.

Operational attempt IDs never enter numerical replay identity. Replacements
must preserve plan, cohort, allocation, worker/partition and precision policy.
Missing, corrupt or mismatched history fails closed; weights-only restart is a
different run, not recovery.

## Bounds And Retention

The host supplies a separate worker-owned private directory and an explicit
disk budget (maximum 8 GiB). Individual artifact bounds remain 1 GiB. A process
lock covers journal reads, publication and cleanup. Start and completed state
plus the pending delta remain until a verified successor assignment permits
retirement. Only unreferenced content-addressed files are reclaimed; unknown
files cause rejection. This is correctness-first snapshot serialization and
can materialize multiple model/state copies in host memory.

## Transport And Evaluation

`learneripc` is the worker's local client. The node-side adapter owns remote
workload/node credentials, pinned TLS and artifact grants. Only semantic values
and bounded artifact chunks cross the inherited-capability local connection.
Session correlation IDs are transport identities; update IDs and receipt hashes
remain stable across reconnects. There is no collective group in the child.

`Runtime.Evaluate` accepts a bound evaluation adapter and verified global model,
without exposing a training-engine mutation port. It records the exact result
beside the completed journal before submission. Coordinator decision/receipt
publication is atomic, and evaluation retries return only the identical result.
The current managed CLI intentionally rejects evaluation without a configured
validation dataset. Service-side dataset selection, designated-evaluator wiring,
submission/status UX and end-to-end multi-host operation are separate integration
work; this foundation does not advertise a completed public DiLoCo service.

## Verification

Unit tests inject lost start replies, interrupted inner work, completion/receipt
publication failures and lost submission replies. They check exact replay,
zero extra work after completion, bounded retention and fail-closed recovery.
Local Unix IPC and authenticated TLS integration tests cover multiple learners,
rounds, attempt replacement, current trust, artifact transfer and evaluation.
Metal-gated tests exercise actual AdamW state and keyed-dropout replay.
