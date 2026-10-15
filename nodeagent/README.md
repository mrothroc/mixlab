# Node-Owned Allocation And Execution State

Node management owns the exclusive accelerator lease and durable admission and
hosting bindings. DDP jobs and independent DiLoCo learner jobs have separate
versioned manifests and records. Learners are not singleton DDP groups.

Learner preparation resolves only the administrator's local dataset catalog,
checks the installed worker/device and dataset again, and binds the exact
staged plan. Workload preparation publishes a principal intent before asking
the credential owner to generate a key. Only the public certificate result is
stored in the job; the local trainer assignment contains no keys or credentials.
Prepared learner leases and controller renewals cannot extend past the staged
plan's admission deadline. Commit handoff enables the separate job-lifetime cap.

After a signed run commit, `FinalizeLearner` writes the commit before changing
lease authority. That first write fences the submitter even if a crash delays
the lease write. Exact retry completes handoff. Only the designated coordinator
can subsequently renew or release. A late handoff adopts cleanup of an expired
or stopped job without reviving it. The node retains the accelerator until
hosting confirms no child and credential cleanup succeeds.

Recovery requires a live original allocation and an exact local stopped-child
binding. `AuthorizeLearnerRetry` records a bounded node authorization; the
coordinator checks its expected private snapshot and precision before persisting
the new attempt. A fresh agent-owned credential and signed coordinator recovery
acknowledgement are required before `StartLearnerRetry` returns a new hosting
assignment. Old credentials are removed first. Old outcomes cannot stop the new
attempt; attempt history is bounded and IDs cannot be reused. Run, worker,
partition and numerical RNG identity remain unchanged.

These are application boundaries, not remotely callable process primitives.
`ConfirmLearnerStopped` is exclusively a local hosting outcome port. HTTP may
not synthesize child-death evidence or select a local executable or dataset path.
The learner runtime/relay dispatcher must be composed separately before agents
advertise and accept the learner manifest over the public job API.
