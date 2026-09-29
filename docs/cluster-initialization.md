# Experimental Cluster Trust Setup

`mixlab-cluster init` creates a local cluster identity and its first authority,
controller, and coordinator credentials. This is a prerequisite for managed
clusters. [Enrollment and authority commands](cluster-enrollment.md), discovery,
[node hosting and submission](cluster-agent.md) are available experimentally.
Signed-package M1/M4 acceptance has passed; managed operation remains limited
to trusted, administrator-controlled hosts. The existing [unmanaged distributed workflow](distributed-training.md)
remains available independently.

## Initialize

```bash
mixlab-cluster init -state-home "$HOME/.mixlab" -trust-listen 192.168.1.10:7443
```

Use the authority machine's LAN address. The default, `127.0.0.1:7443`, suits a
single-machine trial only: the address is signed into the cluster identity, and
no other machine can enroll against a loopback authority. State paths must not
pass through a symbolic link, such as `/tmp` or `/var` on macOS. For a complete
walkthrough, see [managed clusters: getting started](cluster-quickstart.md).

The command outputs JSON containing the cluster ID, full root fingerprint,
eight-word verification phrase, and credential directory locations. These
identifiers are public; no private key or enrollment secret is printed.
The phrase identifies a cluster, but does not authorize enrollment.

| Flag | Default | Meaning |
|------|---------|---------|
| `-state-home` | `MIXLAB_STATE_HOME`, then `~/.mixlab` | Root for contexts without an exact-directory override. |
| `-cluster-state-dir` | Under the state root | Exact authority CA directory; required with `-recover`. |
| `-authority-principal-state-dir` | Under the state root | Separate initial authority TLS credentials. |
| `-controller-principal-state-dir` | Under the state root | Separate initial controller credentials. |
| `-coordinator-principal-state-dir` | Under the state root | Separate initial coordinator credentials. |
| `-trust-listen` | `127.0.0.1:7443` | Concrete host:port recorded in the signed future authority endpoint. No socket is opened. Wildcard addresses and port zero are rejected. |
| `-trust-advertise` | `off` | Only `off` is supported at this checkpoint. |
| `-key-backend` | `keychain` on macOS; `file` elsewhere | New key storage. No silent fallback from Keychain to files. |
| `-recover` | `false` | Resume the exact persisted initialization using only `-cluster-state-dir`. |

The authority owns three distinct signing keys: root, issuer, and snapshot
signer. Each of the three initial principals has its own signing key. A
principal directory contains only its own protected key handle and public
certificates/trust, never the CA private-key handles. The controller and
coordinator directories must not be nested in the authority directory.

Default layout:

```text
STATE_HOME/clusters/CLUSTER/authority
STATE_HOME/principals/CLUSTER/authority/PRINCIPAL
STATE_HOME/principals/CLUSTER/controller/PRINCIPAL
STATE_HOME/principals/CLUSTER/coordinator/PRINCIPAL
STATE_HOME/staging/enrollment/PRINCIPAL
```

Staged principal directories are moved atomically to their final destinations
after verification. An exact-directory override must be on the same filesystem
as the staging directory; cross-filesystem copying of keys is not attempted.
Existing destination directories are never replaced.

## Recovery And Trust

```bash
mixlab-cluster init -recover -cluster-state-dir /absolute/path/to/authority
```

Recovery reads the recorded backend, IDs, paths, and key intents. Do not pass
new initialization options. It reconciles completed publication after an
interrupted response and checks already-published credentials without changing
their identities. It never generates a replacement for a missing active key.
An interruption between recording key creation intent and publishing that key
can require administrator recovery from protected state/backups. The command
fails rather than silently choosing a new identity; do not delete journals to
make an error disappear.

Initial trust snapshots authorize operations for only 15 minutes. Initialization
also creates durable authority snapshot and enrollment journals. Explicit init
recovery can refresh signed trust while preserving revocation history; it never
restores an old principal snapshot over a legitimately refreshed one. A missing
journal after successful initialization is an error, not permission to recreate
an empty history. Expired certificates require explicit renewal/reenrollment,
not rerunning initialization. A running authority refreshes snapshots and
renews its own still-valid principal before expiry. Offline, expired identities
require explicit approved reenrollment; restarting a command does not reset trust.

R1.1 assumes trusted, administrator-controlled hosts. Owner-only directories,
link/ACL checks, and protected key storage do not isolate against a malicious
administrator or another process running as the same account. Initialization
does not install system trust, modify the firewall, advertise a service, or
start GPU work. Keep authority state and backups private.

For lost or compromised CA-purpose keys, follow the
[explicit rekey and reenrollment procedure](cluster-rekey.md). It creates a
separate trust domain rather than mutating existing pins or replay history.
