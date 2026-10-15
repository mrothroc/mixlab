# Durable Coordinator Adapters

These adapters implement the round model's ports. They do not install a service,
recruit learners, issue credentials, or enable the DiLoCo CLI. Admission and
managed learner integration remain separate application work.

## Publication

An owner-approved `statehome.Path` stores each run's immutable, SHA-256-addressed
transition records and a small mutable head. A record contains the next state
and any receipt, published round, or evaluation/decision. File and directory
sync precede head CAS, so those values become visible together. A failed sync
after head replacement is ambiguous: reload before retrying, never assume the
write rolled back. OS process locks serialize cooperating writers per run.

Records are canonical JSON, capped at 8 MiB, checked on load, and bound to run,
revision and prior head. Plans, evaluation policy and existing commits cannot
change. Registrations are immutable per staged slot and validated against the
staged plan; persisting one does not authorize training or commit a cohort.

`PruneOrphans` removes unpublished transition records under the same writer
lock. It preserves the entire committed chain. It is not global blob GC; blob
retention requires all run/assignment references and belongs to service policy.
Receipt lookup walks committed history (linear in transitions on a cold call).
This correctness-first implementation is intended for the bounded R2 pilot,
not an unbounded high-rate journal deployment.

## Tensor Artifacts

`Artifacts` uses the pure `artifact/tensors` codec for bounded float32
safetensors. Payload offsets determine tensor order, matching native checkpoint
layout rather than lexical `wN` ordering. Duplicate fields, overlapping/gapped
offsets, unsupported dtypes, non-finite values and trailing data reject.

Writes spool to a private, immediately unlinked temporary file to avoid another
model-sized RAM copy, then stream to the verified immutable byte store; even
process termination cannot strand that spool on supported Unix hosts. Opening
the store/repository sweeps abandoned atomic-file staging under the directory
lock, without touching published files or staged directories.
Reads verify the stored digest before decoding. `PutVerified` permits exact
retries only after verifying both incoming and stored bytes. It never repairs
or overwrites corrupt content under an existing digest.

## Verification

Tests terminate a subprocess before and after head publication, reopen the
repository between round transitions, inject a lost publication acknowledgment,
race independent repository instances, check registration/receipt persistence,
and prune only unpublished records. The native checkpoint compatibility test
lives in `train/export_diloco_tensors_test.go`; the adapter itself imports no
model compiler, trainer, or GPU runtime.
