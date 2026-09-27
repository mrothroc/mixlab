# Node Credential Replay Journal

`ReplayJournal` is the durable implementation of `trust.EnvelopeReplayGuard`.
It owns replay state, not trust policy or application authorization. It stores
only the exact admitted binding, signed-envelope digest and reservation bit.
It stores no ciphertext, plaintext or private key.

The node's new-job workflow explicitly calls `InitializeReplay` for one envelope
in a dedicated protected directory. Existing/restarted jobs use `OpenReplay`;
missing, corrupt or substituted state is an error, never implicit initialization.
Authentication is still performed by `trust.OpenCredentialEnvelope` before it
calls `ReserveEnvelope`. Atomic compare-and-swap permits one winner and persists
the reservation before decryption. A post-publication error rejects use; a
restarted reserved journal cannot decrypt again. There is no unreserve/reset.

This deliberately favors safety over automatic retry. A crash after reservation
but before protected credential publication requires reconciliation by the job
owner, not another decryption attempt. A future broker must serialize preparation,
use and terminal cleanup with authoritative lease/job state, protect any plaintext
in owner-only job state, and expose only typed authorized operations to workers.
That broker and its application-specific operations are not implemented here.

Retain the reservation with the job journal; do not delete it as a way to retry.
The terminal-cleanup workflow must prevent further use/reinitialization of that
job/attempt before deleting the containing state. As with other local state,
the trusted-administrator host model does not protect against disk rollback.

## Protected Transport Credentials

`Transport` owns a dedicated workload signing key in an agent-owned job directory,
not the trainer's runtime directory. It persists creation intent before creating
the protected key, binds a node-signed request to the admitted workload scope,
and installs only the authority result for that exact request. The relay receives
an opaque signer; neither private bytes nor signer handles are sent to workers.

Reopening never initializes missing state. Exact request retries retain the same
key and proof. Missing key material or lifecycle history fails closed. Terminal
cleanup records intent before deleting the key; retry resumes deletion but cannot
revive the identity. The owning job must stop the relay and confirm child cleanup
before invoking destruction. Stored certificates remain public audit evidence;
every identity use requires current trust, not the original issuance snapshot.

Real-TLS integration covers node request, controller approval, authority issuance,
protected installation/reopen, exact retry, revocation and key deletion. These
are internal building blocks. Local node-job preparation now records an
initialized marker before key creation, and terminal cleanup retains the lease
until credential destruction succeeds. Missing initialized state is an error,
not permission to regenerate a key. Retained signing handles reject use once
destruction begins. The public agent, relay lifecycle and managed-launch workflow
must still compose these ports before managed launch is ready.
