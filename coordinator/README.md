# Fixed-Cohort Round Model

This package is an application model over repository, artifact, allocation,
clock, and evaluator ports. It does not install a service or enable DiLoCo
training. Tests use in-memory adapters; disk crash recovery and authenticated
network handling remain integration work.

```text
PLAN_STAGED -> COHORT_COMMITTED -> ROUND_OPEN -> AGGREGATING
    -> PUBLISHING -> EVALUATING -> COHORT_COMMITTED | COMPLETE
```

Only a matching immutable run commit permits assignments. Allocation readiness
must establish the original allocation and completed authority handoff; this
model never recruits replacement workers. Authentication, current revocation,
and hosting-attempt fencing belong to the API adapters and must precede calls.

Accepted updates and their receipts commit atomically with repository revision
compare-and-swap. Exact retries return the original receipt, including after
advancement; conflicting content or an unseen stale update rejects. Aggregation
uses the committed cohort and weight-layout order. Global parameters and outer
momentum publish together with the round record. Evaluation must preserve the
designated worker's private-state hash; its result and continue/complete decision
commit together before another round can open.

Repository snapshots must be independent, bounded, and structurally validated.
Artifact adapters must verify digest, size, float32 layout, and bounded decoding.
Writes are immutable and content-addressed. A failed candidate write can leave
an unreferenced blob, never a published partial model. Evaluation requests must
be idempotent for their immutable assignment. A CAS conflict means reload and
retry, not repeat training; an ambiguous acknowledgment is resolved the same way.

`Advance` crosses one boundary per call. Errors leave that transition
uncommitted; service integration must classify retryable failures versus terminal
allocation/private-state loss and persist `Fail` when required. Returning an
error here is not itself a production recovery policy.
