# Experimental Cluster Enrollment

These commands establish managed identities for the experimental `agent`,
`nodes`, and `submit` workflows, including managed checkpoint/resume. Managed clusters
support trusted, administrator-controlled hosts only. See
[cluster initialization](cluster-initialization.md) first, then
[node hosting and submission](cluster-agent.md). The
[existing unmanaged distributed workflow](distributed-training.md) is unchanged.

All hosts must be trusted and administrator-controlled. Network TLS and
protected state do not isolate untrusted users on the same host. No command
disables the firewall or installs a root into the system trust store.

## Provisioned

On the authority host, create an owner-only delivery directory and use the
authority directory reported by `init`:

```bash
mkdir -m 700 "$HOME/mixlab-invitations"
mixlab-cluster invite \
  -cluster-state-dir /absolute/path/to/authority \
  -invite-output "$HOME/mixlab-invitations/node.json"
```

The command remains in the foreground, serving one temporary TLS endpoint.
It uses the sole root-signed authority endpoint by default; a routable address
must have been selected at initialization to enroll another host. Deliver the
file privately to the intended host, preserving owner-only directory/file
permissions. It contains a short-lived, single-use secret. Never put its
contents in logs, command arguments, or environment variables.

On the new host, choose an **absent** identity directory:

```bash
mixlab-cluster enroll \
  -enrollment-policy provisioned \
  -enrollment-provisioning-file /private/path/node.json \
  -principal-state-dir "$HOME/.mixlab/node-identity"
```

The client verifies the file's pinned root and fresh signed trust before sending
the secret. It generates private keys locally, validates the issued identity,
publishes it atomically, and deletes its exact consumed provisioning file.
Remove any separate delivery copy on the source host. One file enrolls one
principal; create a separate file for each host. `-invite-purpose` also accepts
`controller-enrollment` and `coordinator-enrollment`; these are privileged
identities, not worker nodes. Invitation TTL defaults to ten minutes, maximum
one hour. `-bootstrap-endpoint` must match a signed authority endpoint.

An uncertain network/publication result is an error, not permission to retry
with a new identity automatically. Inspect the reported state and authority
journal before issuing another invitation. Never delete journals to reset
consumption or revocation history.

## Verified

On the authority host, use its separate coordinator identity from `init`:

```bash
mixlab-cluster enrollment serve \
  -cluster-state-dir /absolute/path/to/authority \
  -principal-state-dir /absolute/path/to/coordinator \
  -enrollment-listen 192.168.1.10:7444 \
  -enrollment-policy verified
```

On the intended node:

```bash
mixlab-cluster enroll \
  -enrollment-policy verified \
  -enrollment-coordinator https://192.168.1.10:7444 \
  -principal-state-dir "$HOME/.mixlab/node-identity"
```

Compare the eight-word cluster phrase and full fingerprint over an independent
channel. Then compare the five-word request phrase and request ID. The client
confirms the phrase; the source operator confirms that exact request. Both
confirmations are required on the original TLS connection. Phrases are public
comparison data, not passwords. The prompts use the controlling terminal, not
piped input; there is no automatic confirmation flag or remote approval route.
Cancel or reject on any mismatch. Reconnection requires a fresh request and
fresh comparisons, never fallback to trusted-LAN approval.

The source defaults to one approval in ten minutes. Set
`-enrollment-max-nodes` and `-enrollment-ttl` explicitly for a larger window.
`-enrollment-allow-purpose` is repeatable and defaults to `node-enrollment`.
The client selects a non-default principal purpose with `-enrollment-purpose`.

## Trusted LAN

Use only on an isolated network whose hosts you administer. This policy
explicitly accepts the first valid coordinator root and automatically approves
eligible nodes; it does not perform human root comparison.

```bash
mixlab-cluster enrollment serve \
  -cluster-state-dir /absolute/path/to/authority \
  -principal-state-dir /absolute/path/to/coordinator \
  -enrollment-listen 192.168.1.10:7444 \
  -enrollment-policy trusted-lan \
  -enrollment-interface en0 \
  -enrollment-cidr 192.168.1.0/24 \
  -enrollment-max-nodes 2
```

The client uses `enroll -enrollment-policy trusted-lan` with the same explicit
coordinator URL and an absent principal directory. The source checks the actual
connection's local destination address, its owning interface and the peer's
source CIDR, not client-supplied metadata. The listener must use an explicit
private, link-local or loopback IP, not a wildcard or hostname.

This is **not physical-ingress enforcement**. A multihomed Mac may receive
traffic through another interface, and a router, tunnel or forwarder can present
an allowed source address. Administrators must control those paths. Use verified
or provisioning-file enrollment when that trusted-LAN assumption does not hold.
Only node enrollment is permitted. Interface/CIDR restrictions are optional
additional constraints under `verified`. CIDRs must be wholly private,
link-local, or loopback, and may be repeated.

Sources may opt into `-enrollment-advertise mdns`; the default is `off`.
Clients can use `-enrollment-discover mdns` instead of a coordinator URL.
Exactly one unique endpoint must be found; zero or multiple endpoints fail and
require explicit selection with `-enrollment-coordinator` and discovery `off`.
The default remains `off`; provisioning files never use discovery.
Advertisements are untrusted address hints, never
root proof or approval-policy selection. They publish no dataset, device,
filesystem, job, or credential metadata. Source mDNS requires a canonical literal
listener IP and port matching its enrollment window. Multicast is local-link only; use
explicit addresses across routed networks or when multicast is blocked.

## Authority And Revocation

```bash
mixlab-cluster authority serve -cluster-state-dir /absolute/path/to/authority
mixlab-cluster revoke \
  -cluster-state-dir /absolute/path/to/authority \
  -principal-id PRINCIPAL_ID \
  -revocation-reason 'host retired'
```

Authority serving exposes public signed snapshots and mutually authenticated
same-principal renewal, not administrator approval endpoints. TLS and signed
snapshot checks remain mandatory. Renewal retries retain their original signed
request across interruption. The authority schedules its own renewal before
one third of certificate life remains, with deterministic jitter, five minutes
of clock-skew allowance, and bounded retries. Expiry fails closed. Node/controller
background scheduling runs in foreground agents and long-running controller commands.
`-trust-advertise mdns` optionally advertises the concrete listener; clients must
still check the authority address against root-signed endpoints. The default is
`off`, and discovery never adds an address to that signed allowlist.

Revocation defaults to `prospective`; explicit `compromise` also invalidates
historical proofs. Choose exactly one of `-principal-id` and
`-certificate-serial`. Revocation is committed before delivery. `-pool local`
uses mDNS node hints by default (`-discover mdns`); use `-discover off` and
repeat `-daemon-address host:port` when multicast is unavailable. Explicit
addresses are still attempted if discovery fails. Each destination must present
an unexpired node identity under the pinned root. No secret or client credential
is sent, only the public signed snapshot.

The result includes per-endpoint `delivery` and reports `pushed:true` only when
at least one destination acknowledged the exact snapshot and all attempts
succeeded. Each receipt names the actual TLS-authenticated node at that address;
discovery does not establish an expected node identity or prove complete cluster
coverage. Discovery failure remains a nonzero partial result even if explicit
destinations acknowledge; use `-discover off` for explicit-only delivery.
Delivery failure returns nonzero but never rolls back the committed
revocation. Offline nodes must refresh before recruiting; a zero-target result
is not proof that remote nodes received the update. Active execution checks
updated trust and cancels revoked jobs through normal physical cleanup.

The node's bounded `/v1/trust/snapshot` route can receive newer signed trust
when its cached snapshot is stale. It cannot enroll, replace the pinned root,
revive an expired certificate, or authorize a normal node operation. Job,
capability and artifact routes still require current mutual-TLS authorization.

Every command has `-help`. Default macOS key storage is Keychain; explicit
`-key-backend file` is available for disposable tests and other supported
hosts. A Keychain error never silently changes the backend.

Root, issuer or snapshot-signer compromise requires
[cluster rekey and reenrollment](cluster-rekey.md), not ordinary leaf renewal.
