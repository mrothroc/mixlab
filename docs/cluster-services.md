# Background Cluster Services

Per-user service administration for the authority and node agent, added in
v0.120.0. Services do not enroll nodes, change firewall rules, unlock Keychains,
grant network permissions, or approve new executables automatically.

## macOS

Use the Developer ID-signed, notarized package (macOS 26), not the source-built
Homebrew formula:

```bash
brew unlink mixlab 2>/dev/null   # only if the source-built formula is installed
brew install --cask mrothroc/tap/mixlab-signed
```

The cask installs the package as `Mixlab` under Homebrew's application directory
(normally `/Applications/Mixlab`) and links both commands into Homebrew's `bin`,
without stripping or re-signing them. Alternatively, copy the whole directory
from the release disk image to a stable location. The macOS Application Firewall
allows the signed `mixlab-cluster` to accept connections; an unsigned build can
be silently blocked. [Distribution](macos-distribution.md) describes how the
package is built.

After enrollment and `agent init`, run these on the appropriate machines:

```bash
mixlab-cluster authority install -cluster-state-dir "$AUTHORITY"
mixlab-cluster agent install -agent-state-dir ~/.mixlab/node-agent \
  -agent-listen 192.168.1.20:7445 -agent-advertise mdns
```

Installation writes `~/Library/LaunchAgents/com.mixlab.cluster-authority.plist`
or `com.mixlab.cluster-agent.plist`, then loads it into `gui/<uid>`. This requires
a logged-in graphical session. At the Mac's desktop or through Screen Sharing,
click **Allow** on the Local Network alert for `mixlab-cluster`. A service stays
alive and retries failed startup every 30 seconds so a denied connection has
time to produce the alert. Check its log rather than assuming installation means
the node is already ready.

Closing Terminal or SSH does not stop these services. **Logging out does.** They
restart after the next GUI login, including after a reboot. There is no pre-login
or headless root daemon. An administrator may choose auto-login for a lab
machine, but Mixlab never configures it. Locking the screen is not logging out;
sleep can still make the node unreachable.

Stable Developer ID signing is intended to preserve both Local Network identity
and Keychain access across upgrades. The initial Keychain must be unlocked in the
user session. Existing keys made by an ad-hoc binary may require migration or
interactive approval; switching to the signed package does not silently change
their ACL or storage backend. Select `-key-backend file` explicitly at enrollment
when Keychain storage is unsuitable. A locked or unavailable Keychain fails
closed, never falls back to files.

## Linux

The same commands install user units under `~/.config/systemd/user/`, enable them,
and start them using `systemctl --user`. A functioning user service manager is
required. Units restart after failure and use a private umask. They do not run as
root and do not change firewall, SELinux or AppArmor policy.

For operation while logged out or before login, ask the administrator to enable
lingering for the service account using `loginctl enable-linger USER`. This is an
explicit administrator action, not something Mixlab performs automatically.

Put executables in a permanent installation directory. If using a versioned
Homebrew installation on Linux, provide the stable path at install time:

```bash
mixlab-cluster agent install -agent-state-dir ~/.mixlab/node-agent \
  -agent-listen 192.168.1.20:7445 \
  -cluster-binary "$(brew --prefix)/opt/mixlab/bin/mixlab-cluster"
```

The path must resolve to the executable running the install command. It is used
by the OS job; executable contents still require explicit approval after upgrade.

## Control And Logs

```bash
mixlab-cluster agent status
mixlab-cluster agent stop
mixlab-cluster agent start
mixlab-cluster agent uninstall
```

The same actions are available under `authority`. There is one service per role
per OS account. `install` accepts `-state-home`, the role's state-directory,
listener and advertise flags, and optional `-cluster-binary`. Other actions take
no installation options. Repeating an identical install is safe; changing its
settings requires uninstalling that registration first.
`stop` pauses the current service but preserves automatic startup at macOS GUI
login or Linux user-manager startup. `uninstall` removes that registration.

Mixlab retains at most 256 KiB of recent service output per role at
`~/.mixlab/services/ROLE/service.log`. Settings and logs live there even when the
cluster's operational state uses `-state-home` elsewhere. Uninstall removes the
OS registration and service configuration, not identities, leases, datasets,
checkpoints or logs. Stop services before removing their binaries. Do not use
`brew services` in addition to these registrations.

## Upgrade Without Reenrollment

1. Finish or abort active jobs and wait for cleanup. Check `nodes` and local
   `doctor` output before upgrading every member of the cohort.
2. Stop agents and then the authority using the commands above.
3. Upgrade both executables together, preserving the stable installation path.
4. On each node, run:

   ```bash
   mixlab-cluster agent reapprove -agent-state-dir ~/.mixlab/node-agent \
     -worker-binary "$(command -v mixlab)"
   ```

5. Start the authority, then the agents. Check authenticated `nodes` results.

`agent reapprove` pins the new worker and the currently running cluster binary,
probes the worker, and advances its capability generation. It preserves the
principal, dataset registrations, resource limits and all lease history. It
requires the agent to be stopped and rejects active or uncleared leases. A failed
publication fences recruitment until the same command is retried; never repair
it by deleting lease state or making a new agent directory.

Stop jobs before rolling upgrades: mismatched builds cannot form a training
cohort, and an old checkpoint may not be compatible with a new numerical build.

## Diagnose Before Changing Policy

```bash
mixlab-cluster doctor -agent-state-dir ~/.mixlab/node-agent
mixlab-cluster doctor -principal-state-dir "$CONTROLLER" \
  -node 192.168.1.10:7445 -node 192.168.1.20:7445
```

`doctor` emits JSON checks for local service state, approved binaries, leases,
principal usability, signed trust freshness, authority TCP reachability and
authenticated node observations when a controller identity is supplied. It is
read-only. Failed checks exit nonzero; `unknown` means an observation could not
establish the answer, not that the check passed. With no identity, TCP success
does not prove authentication. System clocks must be synchronized; doctor checks
local credential windows but does not measure absolute peer clock skew.
Busy or pending-cleanup leases are informational, not a doctor failure: a healthy
busy node still exits zero. Read `lease_state` before upgrades; `agent reapprove`
independently rejects every active or uncleared lease.

`nodes` distinguishes connection refusal, timeout, DNS failure, route/permission
failure, TLS rejection, stale capabilities and busy nodes when supported by the
observation. It does not echo remote error bodies or credential paths. Remote TLS
rejection alone cannot distinguish stale peer trust from another authentication
failure; run doctor on that peer and inspect its service log.

On macOS, repeated `EHOSTUNREACH` to a reachable on-link machine **may** indicate
Local Network denial. It can also be a real routing failure. BSD sockets do not
provide a definitive permission query. Check Privacy & Security > Local Network
and allow the signed service at the console. Gatekeeper, Local Network privacy,
Application Firewall and third-party filters are separate controls. Do not
disable them or remove quarantine to make a test pass.

## Release Acceptance

Before releasing a change to service behavior, verify the published quickstart on two fresh user
installations, both services running without SSH sessions, signed rebuild/upgrade
without lost identity or Keychain prompts, busy-node reapproval rejection,
logout/login and reboot/login recovery, authenticated training and checkpoint
resume. Linux requires a real systemd user-manager lifecycle check, including
the documented lingering policy. Portable rendering tests alone are not evidence
that these operating-system and permission gates passed.
