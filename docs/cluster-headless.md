# Manual Headless Clusters Over SSH (Experimental)

This is the administrator-allowlist alternative to [GUI-approved LaunchAgents](cluster-services.md#macos-hosting-choices).
It uses existing foreground commands, explicit file-backed identities and
`nohup`; it adds no daemon, process manager, tunnel or root training process.
Use it only on trusted, administrator-controlled hosts.

**Evidence boundary:** the Local Network exemption was selective and reversible
in a bridged macOS 27.0.1 VM after desktop login/logout, with ordinary-user
processes. It has **not been verified on a physical Mac**. Physical reboot
verification was waived for this release, not passed. See the
[sanitized spike evidence](cluster-network-evidence.md). The VM allowed unlisted
peers before the first desktop login on each tested boot; that behavior is
undocumented and must not be relied on. We make no promise about behavior before
first login, restart survival, boot-time daemons, sleep/wake recovery or FileVault
unlock. The administrator's policy persists; the manual processes do not restart
automatically.

## Administrator Network Setup

Apple's [TN3179](https://developer.apple.com/documentation/technotes/tn3179-understanding-local-network-privacy)
documents these preferences on macOS 15.5 and later. Inspect and record existing
values **on each dialing host**, using the same root defaults domain:

```bash
sudo defaults read com.apple.network.local-network AllowedEthernetLocalNetworkAddresses
sudo defaults read com.apple.network.local-network AllowedWiFiLocalNetworkAddresses
```

A missing key is not an error to repair. Preserve any existing administrator
policy. The following examples use placeholder peers: replace them with the
actual addresses of the authority, coordinator and nodes this host dials.
Include controller and local fleet-management traffic when choosing coverage.

```bash
sudo defaults write com.apple.network.local-network AllowedEthernetLocalNetworkAddresses -array "192.168.1.10/32" "192.168.1.20/32"
# Only if Wi-Fi may carry this traffic:
sudo defaults write com.apple.network.local-network AllowedWiFiLocalNetworkAddresses -array "192.168.1.10/32" "192.168.1.20/32"
```

These are Apple's command form with example peer CIDRs substituted. `-array`
**replaces** the existing array: include any entries the administrator intends
to retain. Restart the Mac once, at an administrator-approved time, for changes
to take effect. Mixlab never runs these commands or restarts a host for you.

- Prefer exact IPv4 `/32` peers (IPv6 `/128` for exact IPv6 peers). They minimize
  the exemption but require stable addresses and policy maintenance when nodes
  change. Avoid allowing every address just to accommodate DHCP.
- A controlled lab subnet is easier to administer but exempts every destination
  in that subnet, including unrelated machines. The administrator chooses this
  tradeoff; Mixlab does not expand it automatically.
- This is **system-wide for every program**, not a Mixlab-only grant. Configure
  Ethernet and Wi-Fi separately for the interfaces actually used. It is not an
  application sandbox, a port restriction or an inbound firewall rule.
- Treat the recipe as outgoing connection coverage. Each dialing host needs its
  own destination list; the receiver's list does not grant the sender access.
  Keep Application Firewall, routing and inbound listener policy intact. Use
  explicit endpoints and discovery off; this recipe does not promise multicast
  discovery coverage.
- mTLS still authenticates peers. Enrollment, revocation, signed authority
  endpoints, executable approval and workload authorization remain unchanged.
  Signing/notarization, Gatekeeper and Local Network privacy are separate.
  Use the signed package; do not remove quarantine or disable security controls.
  Initial software installation may still require separate Gatekeeper approval.

To undo **only keys introduced by this setup** when they were previously absent:

```bash
sudo defaults delete com.apple.network.local-network AllowedEthernetLocalNetworkAddresses
# Only if this setup introduced the Wi-Fi key too:
sudo defaults delete com.apple.network.local-network AllowedWiFiLocalNetworkAddresses
```

If a key existed before, restore its recorded array instead of deleting it.
Restart again for the restored policy to take effect. This is separate from
stopping roles or removing cluster state; coordinate policy changes with other
applications that use the exemption.

## Provision Identities Before Detaching

Follow [initialization](cluster-initialization.md), [enrollment](cluster-enrollment.md)
and [node registration](cluster-agent.md#local-setup). Use `-key-backend file`
on **`init` and every `enroll`**, including coordinator, admission-controller,
submitting-controller and node identities. For example, for a new cluster:

```bash
mixlab-cluster init -key-backend file \
  -state-home "$HOME/.mixlab" \
  -cluster-state-dir "$HOME/.mixlab/authority" \
  -authority-principal-state-dir "$HOME/.mixlab/authority-principal" \
  -controller-principal-state-dir "$HOME/.mixlab/controller" \
  -coordinator-principal-state-dir "$HOME/.mixlab/coordinator-principal" \
  -trust-listen 192.168.1.10:7443
```

Retain the initialization report; use its actual paths and enroll nodes with
`enroll -key-backend file` plus your chosen enrollment flags. Never expose
invitation secrets or keys in chat/logs. State must be owner-only, on real paths
without symlink components. File keys are not a silent fallback from Keychain:
existing Keychain identities need an explicit migration/reenrollment decision,
not an edit to backend metadata or a new runtime flag.

Authority state and TLS principals intentionally have independent lifecycles.
`-cluster-state-dir` relocates **only authority state**, not the principals.
Without exact principal flags, all three initial roles (authority, controller,
coordinator) are written under `<state-home>/principals/<cluster>/<role>/<id>`.
The state root is `-state-home`, then `MIXLAB_STATE_HOME`, then `~/.mixlab`.
Temporary initialization staging uses `<state-home>/staging/enrollment/<id>` even when
exact principal directories are supplied. For a disposable deployment choose a
dedicated `-state-home` and keep all exact paths inside it; do not accidentally
leave authority credentials in the default home. Successful initialization
removes its staging directories; interrupted initialization must be recovered
with `init -recover -cluster-state-dir ...` before considering cleanup.

**Serving commands do not accept `-key-backend`.** They reopen the backend
recorded at provisioning. The node foreground command is `agent`, not
`agent serve`. Validate the installed version with `-help` before starting.

## Start Roles From An Agent

The examples below use SSH aliases `controller` and `worker`; substitute your
hosts, paths and addresses. All remote commands run as the ordinary owning
user, without a TTY. The agent's own machine is also a fleet member: use
`ssh -o BatchMode=yes localhost ...` for it after configuring authorized SSH
access. Nothing here launches Terminal.app or uses `osascript`.

Before starting, finish active jobs and remove conflicting installed service
registrations with the existing `ROLE uninstall` command (which retains state),
or use separate disposable identities/state/listeners for testing. Do not run
manual and installed copies on the same state. `ROLE status` reports installed
services only; it does not discover a manual `nohup` process. Avoid duplicate
starts and serialize operator actions on each host. PID files below are operator
notes, not locks or a new Mixlab lifecycle API.

Start authority first (the directory must already be initialized):

```bash
ssh -o BatchMode=yes controller 'bash -s' <<'REMOTE'
set -eu
umask 077
bin=/Applications/Mixlab/mixlab-cluster
logs="$HOME/.mixlab/manual"
mkdir -p "$logs"
test ! -e "$logs/authority.pid"  # inspect an existing record; never overwrite blindly
nohup "$bin" authority serve -cluster-state-dir "$HOME/.mixlab/authority" \
  -trust-advertise off </dev/null >>"$logs/authority.log" 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$logs/authority.pid"
ps -p "$pid" -o pid=,uid=,lstart=,command=
REMOTE
```

On each enrolled/initialized node, including the controller if it trains:

```bash
ssh -o BatchMode=yes worker 'bash -s' <<'REMOTE'
set -eu
umask 077
bin=/Applications/Mixlab/mixlab-cluster
logs="$HOME/.mixlab/manual"
mkdir -p "$logs"
test ! -e "$logs/agent.pid"
nohup "$bin" agent -agent-state-dir "$HOME/.mixlab/node-agent" \
  -agent-listen 192.168.1.20:7445 -agent-advertise off \
  </dev/null >>"$logs/agent.log" 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$logs/agent.pid"
ps -p "$pid" -o pid=,uid=,lstart=,command=
REMOTE
```

For DiLoCo, initialize the six [coordinator contexts](cluster-diloco.md#coordinator-contexts)
once before this start command. These example paths must match that initialization
exactly. Port 7447 here avoids a controller-node relay on 7446:

```bash
ssh -o BatchMode=yes controller 'bash -s' <<'REMOTE'
set -eu
umask 077
bin=/Applications/Mixlab/mixlab-cluster
logs="$HOME/.mixlab/manual"
mkdir -p "$logs"
test ! -e "$logs/coordinator.pid"
nohup "$bin" coordinator serve \
  -principal-state-dir "$HOME/.mixlab/coordinator-principal" \
  -admission-principal-state-dir "$HOME/.mixlab/controller" \
  -coordinator-state-dir "$HOME/.mixlab/diloco/coordinator" \
  -admission-state-dir "$HOME/.mixlab/diloco/admission" \
  -checkpoint-dir "$HOME/.mixlab/diloco/checkpoints" \
  -artifact-dir "$HOME/.mixlab/diloco/artifacts" \
  -coordinator-listen 192.168.1.10:7447 \
  </dev/null >>"$logs/coordinator.log" 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$logs/coordinator.pid"
ps -p "$pid" -o pid=,uid=,lstart=,command=
REMOTE
```

All three streams are detached; SSH returns without waiting for the role to
exit. `$!` or a successful SSH exit is **not readiness**. Reconnect to check:

```bash
ssh -o BatchMode=yes worker '/Applications/Mixlab/mixlab-cluster doctor -agent-state-dir "$HOME/.mixlab/node-agent"'
ssh -o BatchMode=yes controller '/Applications/Mixlab/mixlab-cluster nodes -principal-state-dir "$HOME/.mixlab/controller" -discover off -node 192.168.1.20:7445'
ssh -o BatchMode=yes controller '/Applications/Mixlab/mixlab-cluster doctor -principal-state-dir "$HOME/.mixlab/controller" -coordinator-endpoint https://192.168.1.10:7447 -coordinator-id COORDINATOR_ID'
ssh -o BatchMode=yes worker 'tail -n 80 "$HOME/.mixlab/manual/agent.log"'
ssh -o BatchMode=yes controller 'tail -n 80 "$HOME/.mixlab/manual/authority.log" "$HOME/.mixlab/manual/coordinator.log"'
```

Replace `COORDINATOR_ID` with the enrolled coordinator ID. Require authenticated
node availability and `coordinator_tls_trust: ok`, not only TCP success. Doctor
is read-only; its `*_service` rows describe installed OS registrations and their
state paths, never manual processes. An agent registration for a different
`-agent-state-dir` is marked `mismatch`; registrations for other roles may belong
to another cluster and are not evidence about this deployment. Checks may remain unknown for manual roles,
and `local_network_permission` remains unknown rather than pretending to inspect
the effective OS grant. A terminal probe does not prove the background process's
outbound permission. Check startup/refresh logs and continued authenticated
operation beyond the initial trust snapshot window.

Use [DiLoCo submit/status](cluster-diloco.md#submit-and-observe) from a new
non-interactive SSH command with the matching endpoint and IDs. The coordinator
owns a committed run after the submitting command exits. Retain its `run_id`
and check `submit diloco -status -run-id ID` with the same client identity;
`ACCEPTED` is not completion. Ordinary DDP `submit` instead must stay alive
through the job: do not apply DiLoCo's detach semantics to it.

## Stop, Restart And Logs

Finish jobs and confirm worker cleanup before planned shutdown. Inspect the
recorded process on its owning host, for example:

```bash
ssh -o BatchMode=yes worker 'pid=$(cat "$HOME/.mixlab/manual/agent.pid"); ps -p "$pid" -o pid=,uid=,lstart=,command='
```

Compare UID, start time, executable and state arguments with the launch record.
Do **not** signal a recycled PID or use `pkill mixlab-cluster`. After verifying
the exact live PID, send `kill -TERM VERIFIED_PID` over SSH as its owner. Stop
the coordinator, agents, then authority; wait for each role to exit and inspect
cleanup logs before removing its PID note. Do not automatically escalate to
SIGKILL if draining is slow. A PID note alone never authorizes killing a process.

Restart by repeating the start command with the **same durable state**, after
confirmed exit and removal of that role's old PID note. No rerun of `init`,
reenrollment or journal deletion is needed. Recovery still obeys trust, leases,
allocation deadlines and retained private-state requirements; restarting does
not guarantee a run can resume. No automatic restart or structured manual
start/stop API is provided. Existing `ROLE start|stop|status` controls installed
services only, not these processes.

Manual logs belong to the operator: unlike installed-service logs, redirected
files are **not rotated or size-bounded by Mixlab**. Monitor disk use and rotate
or archive only after stopping the process (an open file descriptor keeps
writing to the old file). Preserve run evidence and required checkpoints;
remove only this deployment's PID notes, logs and disposable state after cleanup.
Do not delete live leases or unrelated service installations. Roll back the
administrator exemption separately if the deployment no longer needs it.

For permanent teardown of a **disposable** deployment, inventory the retained
init report's `authority_dir` and every `principals[].directory` (authority,
controller and coordinator), plus any node/enrolled-principal and coordinator
context paths subsequently created. Include the chosen state root's `staging/enrollment`
staging area if initialization was interrupted; never delete it to work around
a recovery error. Also account for manual PID/log files and any OS service
registrations installed separately. Removing only `-cluster-state-dir` does not
remove principals. Delete only verified deployment-owned paths after all roles
and workers stop; do not recursively delete a shared `~/.mixlab` or shared
enrollment root. File-backed keys live with their owning state; Keychain-backed
identities require their existing key-lifecycle procedure, not just file deletion.
