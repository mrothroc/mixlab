# Cluster Trust Foundation

Internal development checkpoint, not an enrollment service or a public cluster
API. No remote listener, CLI command, or trainer import is introduced here.
Do not use this checkpoint as a complete TLS authenticator.

## Boundaries

`trust` validates identity and signed evidence. It does not authorize jobs,
leases, membership, artifact transfers, or enrollment requests. Those contexts
own their schemas, expected audience/context, and durable acceptance journals.
Certificate issuance is confined to `trust/internal/certificates`; an approved
enrollment transition must precede issuance when enrollment is implemented.

This checkpoint supports root, certificate issuer, snapshot signer, long-lived
authority/controller/coordinator/node profiles, and short-lived DDP workers.
Nodes require signed X25519 registration; workers require critical exact
workload bindings in a critical Subject Alternative Name (SAN) extension.
Protected signing stores are in [`securekeys`](../securekeys/README.md), and
snapshot persistence is in [`localstate`](localstate/README.md). Fingerprint word
phrases and HPKE encryption are implemented. TLS adapters, enrollment, renewal,
and the node's credential-use broker remain pending checkpoints. The internal
[provisioned enrollment core](enrollment/README.md) now owns one-use
approval/issuance and recovery; no enrollment endpoint is exposed yet.

## Bound Identities And Envelopes

No PEN, private OID registration, or operator namespace configuration is needed.
Bound certificates use the standard critical SAN extension (`2.5.29.17`) with
exactly two URI entries in order: the canonical principal identity, then
`urn:mixlab:binding:v1:<role>:<payload>`. The payload is unpadded base64url of
canonical binding JSON. Other certificate profiles retain exactly one identity
URI. This private protocol encoding does not claim a registered public URN
namespace. It uses the standard SAN URI container in
[RFC 5280 section 4.2.1.6](https://www.rfc-editor.org/rfc/rfc5280#section-4.2.1.6).

Extension values are canonical, bounded JSON with an explicit profile version.
Mixlab validates URI order, role/version, payload, and raw SAN bytes. Missing,
extra, noncritical, or unknown GeneralNames fail. Unknown critical extensions
still fail. Generic X.509 validation alone is insufficient: application entry
points must use Mixlab's identity/workload verification.
A node cannot reuse its signing public
key as its envelope public key. DDP bindings include cluster, role, principal,
participant, run, job, attempt, audience, issue/expiry, group, generation,
membership hash, member and rank. `VerifyWorkload` requires the complete expected
binding from trusted admission state. Certificates last at most one hour and
issuance cannot exceed the owning workflow's job-plus-cleanup deadline.

`SealCredentialEnvelope` requires a currently eligible enrolled node and a
controller signer. It signs the HPKE payload and exact metadata. The NodeJob
references `EnvelopeDigest`, not ciphertext. `OpenCredentialEnvelope` requires
that exact digest, admitted binding and authenticated prepare-controller ID;
it rechecks fresh trust, revocations, expiry, recipient key and signature before
reserving replay state or decrypting. The opaque credential kind is not an
authorization rule and trust never parses the secret.

The node's future credential-use broker must supply a **durable, atomic**
`EnvelopeReplayGuard`. Reservation happens before decryption and failure rejects
use. A crash after reservation requires owning-workflow reconciliation; it does
not permit blind replay. This library does not implement a durable application
journal. The broker must retain plaintext only in protected per-job state,
provide typed authorized operations (never secret bytes) to workers, and delete
the material during cleanup. No broker, application grants or enrollment API
is implicitly created by these cryptographic ports.

## V1 Rules

- Cryptography uses Go's X.509, Ed25519, CSPRNG, and SHA-256 implementations.
  Root pins are SHA-256 of DER SubjectPublicKeyInfo, not certificate or raw-key
  hashes. Verification never consults platform roots or fetches issuer URLs.
- Certificates have one URI identity (plus a binding URI for node/worker), no subject/display identity, explicit
  usages, bounded lifetimes, and a required immediate issuer. Root and issuer
  keys cannot sign application proofs. Snapshot-signing keys cannot serve as
  principal keys. Key separation must also be enforced by future provisioning.
- Policy lifetimes: root 3650 days, issuer/snapshot signer 365 days, principal
  30 days, workload one hour, snapshot/envelope 15 minutes. Long-lived certificates allow five minutes of issue-time
  clock skew; workload bindings use exact issue/expiry times and expiry is
  exclusive. Children cannot outlive parents. All `now`
  arguments come from the composition's trusted clock, never request fields.
- The root signs authority endpoints. A distinct root-certified key signs
  snapshots. Snapshots contain trust policy only, never application grants.
- An accepted snapshot is immutable. `Advance` requires a newer generation,
  retains revocation tombstones, forbids compromise downgrade and backdated
  new tombstones, and permits recovery from an expired view using a fresh signed
  view. The localstate adapter persists generation and signed bytes atomically.
- Principal signing has a closed role/purpose matrix. Callers must supply the
  exact expected digest, purpose, context, and audience when verifying. A valid
  signature is not authorization to execute its application payload.
- `VerifyProof` returns the exact proof digest and current acceptance generation.
  The owning context must atomically commit both with its domain operation.
  `VerifyHistoricalProof` requires a trusted `CommitJournal` for those exact
  bytes. Never implement that port with network-supplied commit claims.
- Prospective revocation preserves only earlier journaled proofs; compromise
  revocation rejects all history. Signer timestamps are diagnostic, never
  acceptance evidence. Ordinary certificate expiry does not erase prior commits.

## Canonical Encoding

These versioned structs use compact Go `encoding/json` field order, mandatory
fields, integer Unix seconds/generations, lowercase hex IDs/digests, and standard
base64 byte strings. No maps, floats, or optional omitted fields occur in signed
payloads. Incoming bytes must exactly equal decode/re-encode output: unknown or
duplicate fields, whitespace, aliases, and noncanonical numbers are rejected.
Limits are 256 KiB per trust object and 16 KiB per certificate.

An Ed25519 signature covers `SHA256(domain || NUL || canonical_payload)`.
Domains are the explicit endpoint/snapshot/principal-signature version strings.
The unhashed payload excludes its signature. Snapshot/proof signatures include
their exact certificate/evidence bytes. A journal's proof digest hashes the
complete signed proof, deliberately distinct from the caller's content digest.

This is a Mixlab wire encoding, not a claim of RFC 8785 compatibility. Any later
schema or encoding change must be versioned. Restoring a snapshot requires
`DecodeSnapshot` and trusted local rollback protection; decoding a proof alone
does not verify it.

## Tests

`go test ./trust/...` covers real signed chains, invalid profiles, key/purpose
confusion, tampering, immutable views, strict decoding, snapshot rollback,
revocation, and historical acceptance. `go test -race ./trust/...` exercises the
same rules with the race detector. The cluster composition boundary tests keep
trust out of the trainer and constrain this checkpoint to runtime cryptography.
