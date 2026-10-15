# Authenticated Artifact Transfers

This is a coordinator-side HTTP adapter, not a new public launch mode. Compose
it with the existing managed TLS policy, current trust provider, coordinator
repository, protected byte store and a separate protected grant ledger.

The adapter exposes POST `/v1/diloco/artifacts/read` and
`/v1/diloco/artifacts/create`. Each body begins with a four-byte big-endian
canonical-envelope length (at most 1 MiB), then a `Request` JSON envelope.
Create appends exactly the artifact's declared bytes; read appends nothing.
Transfers are limited to 1 GiB and at most one minute, shortened to grant expiry.
Only full-body create/retry is supported; resumable chunk uploads reject.

Every request checks managed TLS identity/current revocation, the coordinator's
signed grant, exact coordinator audience, committed run, participant/worker,
current assignment, operation, artifact identity and expiry. Read grants name
the current assigned global model/outer state or evaluator model. Private state
is never remotely readable. The agent-to-child attempt fence and local request
authorization are later managed-learner integration responsibilities, not a
permission inferred from these bytes or their digest.

The durable ledger records verified proof acceptance and reserves a bounded
logical use before I/O. Different idempotency IDs consume separate uses; exact
retries reuse a reservation, including after interruption or restart. Failed
I/O never refunds it. Independent processes cannot overspend the grant.
Per-use transfer locks are created only after a successful reservation, so
rejected IDs cannot create unbounded lock files. Exact concurrent retries are
serialized and reauthorized after waiting. Reads and writes reverify bytes.

This does not provide per-byte billing: a permitted logical retry may retransmit
the whole artifact. A future service must own issuance/rate limits, retained
grant cleanup, allocation supervision and aggregate disk quotas. No listener
is started implicitly by this package.

The opt-in `TestArtifactLAN` harness in `trust/integration` uses synthetic data,
disposable test trust, and explicit listener/bundle paths. It reports transfer
duration and throughput plus server CPU/RSS. Its private test bundle must be
transferred over an approved secure channel and removed afterward. Ordinary
tests do not start the LAN measurement.
