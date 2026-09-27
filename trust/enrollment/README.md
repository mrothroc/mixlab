# Enrollment And Renewal Core

Internal R1.1 implementation, not an exposed enrollment server. The cluster CLI
currently exposes local initialization only. Provisioned, trusted-LAN and
verified approval rules, live TLS channel binding, protected enrollee publication,
and same-key renewal are implemented internally. Operational enrollment commands
and approval presentation are not yet available. Do not expose this package
through an unauthenticated generic RPC dispatcher.

## Ownership

This package owns one-use provisioning, approval, and principal issuance. It
accepts protected signing ports, a pinned root, and current verified trust from
local composition. It does not load CA keys from requests, select endpoints from
discovery, own leases, or launch workers. `trust.AuthenticatePrincipal` checks
current certificate/role/revocation evidence; transports must additionally
verify key possession and application contexts must authorize operations.

Supported purposes are node, controller and coordinator enrollment. Authority
bootstrap belongs to explicit cluster initialization. External-worker bootstrap
and recovery belong to later release workflows and fail here rather than
accidentally issuing unbound worker credentials.

## Provisioning Flow

1. Explicitly initialize the authority-owned journal. `Open` never creates or
   resets missing state. Supply current, locally verified trust on operations.
2. `Invite` checks the root-signed endpoint/audience and role eligibility. It
   generates a fresh 256-bit secret with a one-use limit and bounded TTL (up to
   one hour). The authority stores a verifier bound to the entire invitation,
   not the secret. `EncodeInvitation` bytes must go to protected file storage.
3. The enrollee decodes canonical protected bytes and validates the exact target
   before sending a secret. It must pin the invitation root and authenticate the
   remote authority against the signed endpoint set before transmission.
4. `NewRequest` signs a domain-separated request with the locally generated
   Ed25519 handle. Node enrollment includes an independent X25519 public key.
5. `Consume` verifies the bearer proof, exact purpose/role/root/target, proof of
   possession, expiry and duplicate/reserved-key rules. It durably commits signed
   receipt and approval together before the certificate issuer is invoked.
6. Issuance commits the resulting chain before returning it. `ValidateResult`
   verifies the chain, exact requested keys/identity, signed approval/receipt
   cross-binding and current snapshot under the original pin. The `enrollee`
   package validates the result, installs it beside locally generated keys, and
   promotes that complete context to an absent final directory. Protected-file
   helpers write/read/delete exact provisioning files; CLI wiring remains pending.

## Interactive Policies

`OpenWindow` is administrator-local. Trusted-LAN windows require an exact
interface, bounded private/link-local/loopback CIDRs, a positive approval limit,
and node-only purposes. Verified windows require both client and local operator
confirmation of the exact request digest; optional interface/CIDR restrictions
are additional constraints, not an auto-approval policy.

The default window is ten minutes, with a one-hour maximum, 16 pending requests,
and 16 new requests per minute. Policy, endpoint/audience, purpose, proposed keys,
nonces and request ID are cryptographically bound. Transport supplies observed
addresses and one-use TLS exporter material; the domain never trusts claimed
network metadata. Raw exporters are cleared and never persisted. A fresh
connection cannot poll or confirm an existing request, even with its ID/phrase.
Closing a window or disconnecting expires unfinished requests. Restart drops all
live evidence; pending requests cannot resume on a new connection.

Approval is durable before issuer invocation; certificate publication is durable
before returning credentials. Administrator decisions are in-process ports only,
not HTTP routes. `transport/enrollmenttls` provides the isolated TLS profile and
single-connection HTTP adapter; `internal/clusterapp` composes bounded handlers.

## Renewal

Renewal requires current mutual-TLS identity plus same-key proof of possession.
It preserves role, principal, signing public key and node envelope public key.
Expired/revoked identities cannot renew. An exact request retry returns its
committed successor certificate; another nonce cannot create a second successor
from the same old leaf. Client-side publication checks eligibility again and
atomically preserves the root/key identity while advancing signed trust.

Both receipts and approvals are signed by the authority TLS principal, never
the root, issuer or snapshot signer. The issuer signs certificates only after
the approval journal commit. Neither kind of proof grants node or job access.

## Recovery And Bounds

Concurrent consumption has one winner. Even byte-identical consume replay is
rejected. An authority-local `RecoverApproved` call can reconcile a lost outcome;
it cannot consume an unused invitation. It preserves the approved principal/key
and returns an already committed certificate byte-for-byte. If a certificate
was signed but not durably published, recovery may sign a new serial for the
same approved identity: the uncommitted certificate was never returned.
Expired approvals cannot mint certificates; revoked authorities and identities
fail current trust checks. This is not an anonymous network retry route.

All journal mutations use cross-process locking plus protected-file CAS. The
canonical journal is bounded to 256 records per category and 16 MiB; capacity fails
explicitly, without evicting consumed records or resetting replay history.
There is no automatic compaction yet. The trusted-administrator/disk-rollback
limitation of the rest of R1.1 applies here as well.

Invite, consume and publication errors return no secret or certificate result.
Callers clear secret buffers after protected output/use. Invitation formatting
is redacted, but explicit JSON serialization is secret-bearing by design.

## Verification

Tests use runtime Ed25519/X.509 and file-backed node signing/encryption handles.
They cover all supported roles, target/TTL validation, forgery, result tampering,
reserved and duplicate keys, replay/concurrency, stale trust, revocation, and
failures before/after approval and issuance publication. The concurrency test
runs under the race detector. No live CA or real credential is used.
