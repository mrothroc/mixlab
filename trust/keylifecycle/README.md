# Durable Key Lifecycle

This local trust lifecycle owns key creation intent, recovery and retirement.
It neither issues certificates nor approves enrollment, jobs or revocations.
No operational enrollment command is exposed by this checkpoint.

## Key Inventory

| Owning context | Permitted slots | Purpose |
|---|---|---|
| `cluster-authority` | `root`, `issuer`, `snapshot-signer` | Three separate Ed25519 authority keys |
| `authority` | `principal` | Authority TLS identity, not a CA key |
| `controller` | `principal` | Controller identity and allowed manifest signing |
| `coordinator` | `principal` | Coordinator identity and allowed run proofs |
| `node` | `principal`, `node-envelope` | Independent Ed25519 TLS and X25519 HPKE keys |
| `worker` | `workload` | Attempt-bound short-lived Ed25519 workload key |

Every creation uses fresh randomness and a new handle. The store's scope,
backend, owner, slot and generation are recorded with the public handles. No
private seed or application secret is stored in an intent. Existing contexts
must open the recorded backend explicitly; no fallback or migration occurs.

`Initialize` records the immutable owner/backend/scope for an explicitly new
context. `Open` requires that record and rejects changed ownership or missing
state. Initialization also rejects orphaned slot intents rather than adopting
or overwriting them. A node directory cannot be reopened as a CA owner.

## Transitions

1. `Create`: prepare an in-memory candidate, durably record `creating`, publish
   that exact key, then publish `active`. A lost unpublished candidate requires
   explicit abort; recovery never generates replacement material.
2. `Recover`: inspect the recorded handle. A published candidate completes its
   pending state; missing, corrupt or unavailable active keys fail closed.
3. `Rotate`: record `rotating` with the old key still active, publish a separate
   candidate, then reach `rotation-ready`. New certificates are not issued here.
4. `Activate`: after the owner durably installs approved new credentials, match
   the exact candidate ID and retain the old key in `retiring`.
5. `Retire`: after the owner reconciles old credential/revocation obligations,
   record deletion intent, delete exactly the old handle and finalize `active`.
6. `Abort`: match the candidate ID, record cleanup intent before deletion and
   restore the old active key, or leave an `aborted` tombstone for an initial
   creation. A later explicit `Create` uses a new ID and generation.

CA slots reject ordinary rotation with `ErrRekeyRequired`. Root, issuer or
snapshot-signer loss/compromise needs the explicit cluster rekey workflow,
including new pins and reenrollment; that workflow is not yet implemented.

The protected `key-lifecycle.lock` serializes cooperating processes across
filesystem and Keychain side effects. Never unlink/replace it while the context
exists. Waiting is cancellable, and process exit releases the OS lock. Public
intent updates use checked compare-and-swap; uncertain publication/deletion is
reconciled on the next explicit recovery, not interpreted as rollback.

These APIs are for trusted local composition. `Create` is an explicit new-key
transition, not a fallback after failed recovery. Enrollment must preserve its
own durable context-existence record and never reinterpret missing old state as
a new cluster. Certificate publication, rotation authorization and revocation
remain the owning enrollment/admission workflow's responsibilities.

Tests inject failures before and after intent/outcome publication, reopen state,
verify exact-key recovery and cleanup, check concurrent creation, missing keys,
backend/context mismatches and corrupt records. The host model remains trusted
administrator-controlled machines, not protection from an administrator restoring
an old backup or editing the journal.
