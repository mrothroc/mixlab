# macOS Cluster Network Evidence

Sanitized summary of the October 2026 hosting spikes. These are connectivity
experiments, not model-training benchmarks or a LAN-speedup claim. Private host
addresses, identities, credentials and raw operator logs are intentionally absent.

## Spike D: Ordinary-User System Daemon

On a physical Apple Silicon Mac running macOS 27.0.1, the same signed executable
launched directly by a system LaunchDaemon at UID 501 failed its outgoing LAN
connection with `connect: no route to host`. A detached ordinary-user control
also failed. A root diagnostic using the same binary connected. The direct run
removed a shell-wrapper attribution ambiguity. The Intel host running macOS 26
was not applicable evidence for the macOS 27 problem.

This does not establish that every socket error is privacy denial. It does
establish that installing an ordinary-user daemon alone was not a solution in
that test. Root orchestration/training and a privileged socket broker are not
part of the supported workflow. Temporary helpers were removed.

## Spike E: Administrator Destination Exemption

Used a disposable macOS 27.0.1 / 26A434 VM, Apple Virtualization with bridged
Ethernet (not NAT), two reachable non-gateway LAN peers, one signed executable,
and UID 501. No Local Network prompts were approved. The guest's default
firewall was unchanged. Only the guest received the root defaults exemption
and restarts; no physical Mac's privacy policy was changed.

After desktop login/logout on each boot, with the console logged out:

| Guest policy | Detached process to listed peer | Detached process to other peer | Direct ordinary-user daemon to listed peer |
|---|---|---|---|
| No exemption | Denied | Denied | Denied |
| One exact peer `/32` | Connected | Denied | Connected |
| Exemption removed, restart, login/logout | Denied | Denied | Denied |

Denied connections reported `no route to host`. Independent SSH-session probes
still reached both peers at reversal. This supports a selective, reversible
policy effect, not a peer outage. The boot daemon also connected with the
exemption, without a GUI login.

**Caveat:** before first desktop login on every tested boot, both peers were
reachable even without the exemption. Therefore boot connectivity did not prove
the exemption's effect in that state; the original no-login negative control
was not satisfied. The cause is undiagnosed. Do not rely on this observed
fail-open behavior or advertise a pre-login security/availability guarantee.

The guest, its disk, restore image and temporary credentials/binary were removed.
No production application protocol or model code changed in these spikes.

## Support Boundary

The [administrator-allowlist workflow](cluster-headless.md) combines Apple's
[TN3179 policy](https://developer.apple.com/documentation/technotes/tn3179-understanding-local-network-privacy)
with the qualified VM evidence above. **It has not been verified on a physical
Mac**; that reboot gate was waived because physical restarts were unavailable.
This is not an assertion that authenticated multi-host training, trust renewal,
or detached lifecycle acceptance passed through the exemption.

GUI-approved LaunchAgents passed the separate physical reviewer gate on candidate
`29cb1d2`: non-interactive SSH service management, an eight-round two-Mac Metal
run under ambient host load, trust refresh beyond 15 minutes, status/logs,
service stop/restart and cleanup. The M1 required a fresh console approval;
an identical signing requirement did not eliminate that prompt in this run.
This is path-1 acceptance, not a detached-process or physical allowlist pass.
It cannot establish the headless exemption's behavior on an unapproved host,
and it is not hardware verification of subsequent code changes.
No automatic reboot recovery, sleep/wake handling or FileVault unlock is promised.
See the [support matrix](feature-matrix.md#managed-cluster-hosting).
