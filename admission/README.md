# Recoverable Learner Admission

This context owns one durable cross-context saga per fixed learner run. Its
private journal is separate from coordinator state and node lease state.
`Begin` publishes the selected cohort, immutable model/config/outer recipe,
evaluation policy, and deterministic reservation keys before any remote effect.
`Open` never initializes missing history. `Step` performs one bounded effect;
the owning service reopens and resumes it independently of a submitting client.

Order: reserve, publish a fixed plan, stage, prepare credentials and the local
assignment, start, register, persist the commit candidate, commit, finalize all
node lease authorities. All ports must implement exact-request idempotency and
verify their current authenticated peer. The prepare/start/register adapters
must compare the child's actual program, precision and preparation evidence
before registration, not merely echo the proposed configuration.

Cancellation and admission expiry first fence commit in the coordinator. An
ambiguous successful commit wins over cancellation and requires completing
handoff. A noncommitted saga permanently fences each touched reservation,
including requests whose response was lost, and waits for confirmed child and
credential cleanup. Release intent is never considered confirmed cleanup.

The coordinator's `FenceAdmission` must be the abort port: it can publish an
absent-plan tombstone, so a delayed stage request cannot resurrect compensation.
Do not implement this port as a read followed by an unrelated delete.

This package does not issue credentials, mutate leases, supervise children,
select local paths, or make optimizer decisions. Its ports connect those owners.
The managed learner dispatcher, authenticated learner request service and CLI
composition are subsequent work; this package alone does not enable DiLoCo.
