# Cluster Rekey And Reenrollment

Rekeying replaces a trust domain; it is not certificate renewal. Use this
procedure after loss or compromise of root, certificate-issuer or snapshot-signer
keys. Ordinary rotation cannot replace those keys. No command silently changes
an enrolled principal's pinned root.

## Contain The Old Cluster

1. Stop submissions and foreground agents on every affected host. Confirm
   worker and guardian cleanup before starting replacement agents. An unreachable
   host is not confirmed stopped; contain it through administrator-controlled
   network or process access before trusting it again.
2. Retain old journals and protected state for investigation and recovery.
   Do not delete state to clear a busy lease, restore an old trust snapshot, or
   reuse possibly compromised principal keys in the replacement cluster.
3. If only a node/controller key is compromised and the CA-purpose keys remain
   trustworthy, use `revoke -revocation-mode compromise` instead. Push delivery
   reports which reachable agents acknowledged the update; it cannot prove that
   an offline host has stopped. Reenroll the affected principal into a new empty
   directory through an explicitly approved enrollment policy.

## Establish New Trust

Initialize a separate state home, using an administrator-selected reachable
address for the new authority:

```bash
mixlab-cluster init -state-home /absolute/replacement-state \
  -trust-listen 192.168.1.10:7443
```

Record the new cluster fingerprint and phrase. Confirm that they differ from
the old cluster. The initializer creates distinct root, issuer, snapshot signer
and principal keys; it does not import old keys or cross-sign the new root.

Distribute the new fingerprint through a channel independent of the compromised
cluster. Use [verified or protected-file enrollment](cluster-enrollment.md) for
each node and secondary controller. Avoid trusted-LAN first-root acceptance
while investigating a compromise. Use new empty principal directories and
retain old directories separately; enrollment must not overwrite them.

Reinitialize each node's local agent installation with its newly enrolled
principal, approved executable hashes, local dataset catalog and relay address.
See [node setup](cluster-agent.md). Dataset files can remain in place, but new
jobs must bind their current content identity. Start only the replacement agent
and use authenticated `nodes` inventory before submitting a new fixed cohort.

## Verify And Retire

Old-cluster certificates, workload streams, signed jobs and provisioning files
must fail against the replacement pin. Do not transfer leases, physical-attempt
journals or launch permissions into the new cluster. A replacement cluster does
not resume a crashed managed attempt; submit a new job after cleanup.

Confirm new enrollment, authenticated inventory, a small training job and
physical cleanup before retiring old state under the administrator's backup and
incident-retention policy. Preserve useful model artifacts separately from trust
state. Rekey does not automatically delete old Keychain entries, backups or
offline host state, and it makes no claim to revoke a compromised root globally.

If the replacement is interrupted, recover its own recorded initialization with
`init -recover -cluster-state-dir PATH`; do not rerun initialization over existing
state or copy CA-purpose handles into principal directories.
