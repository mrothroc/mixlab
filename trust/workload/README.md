# Workload Certificate Issuance

This authority-owned service issues a job-duration transport certificate for a
new protected key. It does not admit jobs, reserve nodes, interpret training
manifests, or launch workers.

The node signs a PKCS#10 request whose subject binds the complete opaque scope.
The controller then signs that exact node request. Issuance authenticates the
actual controller TLS peer and checks both approvals against current trust.
The certificate binds the cluster, node, workload identity, run, lease, job,
attempt, manifest digest, DDP membership digest/generation, member, rank,
audience and deadline. The node-job context computes the deadline from the
admitted runtime plus cleanup allowance; trust enforces the seven-day ceiling.

The authority journals accepted proofs before signing and the issued certificate
before returning it. Exact retries return the same published certificate;
different grants cannot reuse a job/attempt, workload identity, or public key.
Historical approvals are usable only with this journal's durable acceptance
evidence. The caller and node must still authenticate under current trust on
every retry, including prospective revocation. Issuance has no worker-key access.

`Initialize` is explicit local setup. `Open` never recreates missing published
history. An interrupted initializer can complete only its exact empty claim.
The journal is bounded to 512 grants and 32 MiB; reaching capacity fails rather
than discarding replay history. Compaction is not yet implemented.

The experimental authority exposes this service on managed mutual TLS at
`POST /v1/trust/workloads/issue`. Anonymous snapshot clients cannot issue
credentials. The operational node preparation and recruitment workflow still
must ensure grants originate from live prepared jobs; this service alone is not
a complete managed-launch implementation.
