# Persisted Trust Snapshots

This adapter stores one canonical protected `trust-state.json` containing the
pinned root certificate/fingerprint and complete signed snapshot. Snapshot bytes
and generation commit together through `statehome.CompareAndSwap`; concurrent
writers cannot silently replace a newer view. The caller supplies an independent
expected root fingerprint. Initialization never overwrites existing state.

`Load` may restore an expired snapshot for refresh, but all trust operations
continue to reject it as stale. `Advance` validates a fresh successor, preserves
revocation history, then conditionally commits against the exact old bytes.
Conflicts require reload and revalidation. Missing/corrupt state is an error,
never an implicit initialization or root replacement. A post-rename durability
error requires rereading state before retrying.

This is rollback protection against stale cooperating writers and incoming
snapshots, not hardware-backed protection against an administrator restoring an
old disk backup. R1.1 assumes trusted, administrator-controlled hosts. Restoring
backups, changing roots, and key rotation need explicit owning-context policy.

The adapter stores no private keys and owns no job/run/artifact acceptance
journal. Enrollment services, startup composition, revocation distribution, and
renewal scheduling still need to wire it into their lifecycles.

## Bound Identities

Node/workload bindings use the fixed versioned SAN URI profile. `Open` needs
only the trusted root fingerprint; no external OID registration or namespace
configuration is required. Restore and advance retain the exact root pin.
