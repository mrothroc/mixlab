# Managed DiLoCo (Experimental)

DiLoCo uses a fixed cohort of single-process Metal learners and a durable
coordinator service. It does not use an MLX collective group or job-time SSH.
This is an experimental workflow, not a claim of two-host service acceptance,
exact recovery on every failure, or LAN speedup. Use a build that exposes
`mixlab-cluster coordinator init -help` and `submit diloco -help`;
availability in source does not mean a signed release has been published.

Start with [cluster setup](cluster-quickstart.md), [enrollment](cluster-enrollment.md),
and [node hosting](cluster-agent.md). Enroll and approve both executables, register
the training datasets on every node and the validation dataset on the designated
learner, and keep the authority and agents running. Use matching builds, dataset
hashes and supported causal AdamW configs. Learner precision capabilities must be
compatible with the immutable plan. The cluster executable does not link MLX.

## Coordinator Contexts

On the authority host, keep the existing enrolled coordinator principal and a
separate enrolled controller principal for admission. Neither is the authority
key store. The submitting controller is a client; admission reserves and starts
learners using the service's controller identity. After commit, the coordinator
owns allocation renewal and release.

All of `coordinator init`, `coordinator install` and `coordinator serve` require
these **six independent paths**, plus the listener:

| Flag | Owner and contents |
|------|--------------------|
| `-principal-state-dir` | Existing enrolled coordinator identity and trust. |
| `-admission-principal-state-dir` | Existing enrolled controller identity for admission. |
| `-coordinator-state-dir` | Durable coordinator plans, rounds, receipts and authorization journals. |
| `-admission-state-dir` | Durable accepted submissions and admission/compensation journal. |
| `-checkpoint-dir` | Published global checkpoint references. |
| `-artifact-dir` | Content-addressed global model and delta artifacts. |
| `-coordinator-listen` | Concrete reachable local `host:port`, also advertised as the HTTPS origin. |

Paths must be canonical absolute directories, with no duplicates or ancestor/
descendant overlap. Sibling directories under one private parent are allowed.
There is no `-state-home` shortcut or implicit authority directory for these
commands. Each context receives only its own path. Use stable directories with
private ownership and permissions; do not copy one context over another.
Wildcards such as `0.0.0.0` or `::` are not advertised addresses.

For example, using paths from your initialization report for the two principals:

```bash
coordinator_args=(
  -principal-state-dir "$COORDINATOR_PRINCIPAL"
  -admission-principal-state-dir "$ADMISSION_CONTROLLER"
  -coordinator-state-dir "$HOME/.mixlab/diloco/coordinator"
  -admission-state-dir "$HOME/.mixlab/diloco/admission"
  -checkpoint-dir "$HOME/.mixlab/diloco/checkpoints"
  -artifact-dir "$HOME/.mixlab/diloco/artifacts"
  -coordinator-listen 192.168.1.10:7446
)
mixlab-cluster coordinator init "${coordinator_args[@]}"
mixlab-cluster coordinator install "${coordinator_args[@]}"
mixlab-cluster coordinator status
```

`init` validates the existing principal roles/cluster and initializes the four
storage roots without opening a listener. Repeating the same initialization is
safe; changing the owning identities or origin is rejected. It is not a reset
or migration command. `install` registers and starts the per-user service;
it does not enroll identities or initialize those storage roots. It optionally
accepts `-cluster-binary`, a stable absolute path resolving to the executable
running the install command. On macOS installation requires the signed package.
Setup is explicit: no default startup installs, stops or replaces an existing
service. Inspect existing registrations before choosing `install` or a control
action; a conflicting registration is rejected, not silently overwritten.

For foreground diagnosis, stop the service first and run
`mixlab-cluster coordinator serve "${coordinator_args[@]}"`. Foreground serving
ends with its process/terminal. Never run foreground and installed copies over
the same state. `coordinator start|stop|status|uninstall` take no configuration
flags. Changing service settings requires uninstalling the registration first;
that does not migrate or erase its operational state.

## Login And Terminal Contract

The coordinator reuses the [existing user-service lifecycle](cluster-services.md).
macOS uses `com.mixlab.cluster-coordinator` in the logged-in GUI session, not a
root daemon. Closing the setup/submission terminal or SSH session does not stop
an installed service. **Logging out does**; reboot recovery requires the next
GUI login and usable Keychain. Locking the screen is not logout, but sleep and
network loss still interrupt availability. Linux uses a systemd user service;
logged-out operation requires administrator-enabled lingering.

Perform initial permission approval in a desktop Terminal or through Screen
Sharing with access to the user's GUI and unlocked login Keychain. Allow the
signed service's Local Network prompt there. A submission terminal must itself
be able to reach and authenticate the coordinator and selected nodes,
and access its enrolled controller backend. A terminal-hosted app or SSH context
that fails these checks is not made usable merely by approving a LaunchAgent.
Use a verified desktop terminal for setup/submission when another context fails.
No command unlocks a Keychain, changes firewall/privacy policy, or falls back
from Keychain storage to file keys automatically.
The client attempts an authority trust refresh; during an authority outage it
can continue with a still-current cached snapshot. Stale or invalid local trust
fails closed when that refresh cannot complete.

## Submit And Observe

The config's `training.distributed.mode` must be `diloco`. Specify positive
`inner_steps`, `outer_optimizer: "nesterov"`, positive `outer_lr`, explicit
`outer_momentum` in `[0,1)`, and `aggregation: "uniform"` or `"effective_tokens"`.
The positive `training.steps` must be divisible by `inner_steps`. See the
[config contract](config-reference.md#diloco-contract-preview) for restrictions.

```bash
service_args=(
  -principal-state-dir "$SUBMITTING_CONTROLLER"
  -coordinator-endpoint https://192.168.1.10:7446
  -coordinator-id "$COORDINATOR_ID"
)
mixlab-cluster submit diloco "${service_args[@]}" \
  -config model.json -worker-binary /Applications/Mixlab/mixlab \
  -train corpus-train -dataset-id "$TRAIN_SHA256" \
  -validation corpus-validation -validation-id "$VALIDATION_SHA256" \
  -validation-batches 4 -workers 2 -discover off \
  -node 192.168.1.20:7445 -node 192.168.1.30:7445
```

`-workers` is 2..64, fixed for the run. `-discover` defaults to `off`, requiring
repeatable `-node` endpoints; `mdns` supplies discovery hints, not trust. Data
selectors are registered node-local logical names, not submitting-host paths.
The first ordered selected learner is the designated evaluator and must advertise
the requested `-validation` selector and `-validation-id` hash. Both are required
for a new submission; selection does not automatically switch evaluators when
the first ordered learner lacks them. It evaluates verified global checkpoints
without changing its private training state. The coordinator owns durable global
evaluation and optional `-target-loss` early stopping; validation never silently
falls back to training data. `-validation-batches` defaults to 1. An explicit
`-target-loss` must be finite and non-negative; omitting it disables target-based
early stopping, whereas explicit zero is a real target.

Resource ceilings are `-max-runtime-seconds` (3600), `-max-cpu-seconds` (7200 per
learner), `-max-memory-bytes` (4294967296), `-max-disk-bytes` (4294967296 private
disk) and `-max-log-bytes` (16777216). Node policy can reject these requests.
`-authority-endpoint` optionally selects a root-signed authority URL;
`-state-home` is available to the submitting client, not coordinator configuration.

Retain the `run_id` printed on stderr before submission and in the JSON response.
Printing a run ID alone does not establish acceptance: it precedes the initial
model upload, which can fail without creating a submission record.
`ACCEPTED` means the service persisted the request, not that training has started
or completed. Admission continues independently of the submitting terminal and
attempts to commit the fixed cohort or compensate its partial effects. It can
remain pending while required services or cleanup operations are unavailable. A committed
run is not canceled when the terminal exits. Query before resubmitting after an
ambiguous response; a fresh submission creates a different run.

```bash
mixlab-cluster submit diloco "${service_args[@]}" -status -run-id "$RUN_ID"
mixlab-cluster submit diloco "${service_args[@]}" -cancel -run-id "$RUN_ID"
```

Status/cancel use the original submitting controller and retained service
identity. They reject training, discovery and resource flags. JSON reports
`admission`, `run_phase`, `round`, `failure`, `service_error` and `observed_at`
when applicable. Failed runs or aborted admissions exit nonzero; a successful
status query is not a promise of progress. `-cancel` requests bounded pre-commit
compensation only; it cannot cancel an already committed run, and cleanup is not
necessarily complete when the request returns. Inspect status and service logs.
The request and individual effects are bounded, not the total time to finish
compensation during an outage. A cancel can also race commit; inspect the durable
status rather than assuming that an acknowledged cancel prevented commitment.

## State Retention And Recovery

Keep coordinator/admission journals, checkpoint references and their artifact
bytes together, plus each node's agent state and private learner attempt storage.
The checkpoint directory alone is not a complete backup. Private learner state
includes AdamW moments/counters, schedule position, sampler/cursor and RNG
identity, precision evidence and completion records. A global model is not a
replacement for this state. Start/completion snapshots are retained until a
verified successor permits cleanup. Do not manually prune live attempt state,
delete journals to clear a busy lease, or infer safe deletion from a lost ACK.
Service uninstall preserves operational state and logs; it is not run cleanup.

Service restart refreshes both service principals' public trust through the
pinned authority before requiring active credentials. A stale public snapshot
can recover when the authority is reachable and the enrolled identity is still
valid; no reenrollment or state deletion is needed for snapshot staleness alone.
If refresh fails and local trust is stale, startup fails closed. Expired/revoked
identities or an inaccessible Keychain are not repaired by public trust refresh.
Doctor is read-only and does not perform this refresh.

Service restart reopens the durable inbox and coordinator history. This is not
unbounded fault tolerance: trust must remain usable, allocations and deadlines
must remain valid, and learner recovery needs the exact expected private-state
hash and a fenced attempt on the same allocation. Missing/corrupt private state,
incompatible config/build/precision, expired leases, or missing global artifacts
cannot be repaired by silently starting fresh. The cohort does not elastically
replace failed members. A long logout/reboot or outage can exhaust deadlines and
fail the run even though the service restarts successfully.

The coordinator observes stopped learner processes and automatically attempts
same-allocation recovery, capped at three replacement attempts per worker per
run. Recovery intent, its fresh attempt identity and its 60-second deadline are
durable, so restarting the coordinator does not reset the allowance. An unknown
private-state hash (including failure before the first recorded start), a
permanent authorization rejection, or exhausted recovery fails the run explicitly.
An unavailable node-status endpoint also has a persisted 60-second budget.
Status includes `recovery_worker` and `recovery_started_at` during recovery,
or `learner_unavailable_since` while node status cannot be read. Exhaustion
changes `run_phase` to `FAILED` with a sanitized `failure` reason.
Fresh credentials fence the old attempt; private optimizer, sampler and RNG
state are restored, never replaced with newly initialized state.

The agent retries only semantic learner HTTP requests after transient connection
refusal/reset, route unavailability, EOF or timeout, or authenticated HTTP 503
while the coordinator is unavailable/draining or retryable state contention
occurs, using the exact encoded request and fresh pinned
trust policy for each attempt. Retries fit within the caller's deadline and a
20-second total cap, with at most five seconds per attempt and 100 ms to one
second backoff. Authentication, certificate, scope, other HTTP-status and malformed
response failures are not retried. Lost acknowledgments use the coordinator's
existing idempotency records, not extra inner training. This is a short-outage
allowance, not indefinite restart tolerance. Raw artifact reads/writes are not
automatically replayed: their caller-owned streams have no implicit rewind or
destination reset, so an interruption can still fail the learner attempt.
That failure uses the same bounded, private-state-preserving recovery path;
it is not a claim that every interrupted transfer preserves the original PID.
The local IPC boundary distinguishes exhausted transient unavailability from
terminal rejection without disclosing remote credential diagnostics.

To start a **new weights-only run**, optionally pass
`-initial-model /absolute/path/model.safetensors` with the new-run flags. It must
be a bounded float32 tensor artifact matching the config/layout. It does not
restore outer momentum, learner AdamW state, schedules, cursors, RNG progress,
cohort identity or prior run history. This is not exact resume, and R1 `-resume`
bundles are not DiLoCo recovery inputs. Without `-initial-model`, the approved
worker produces one deterministic initial model from the config and the service
uses that common artifact for all learners.

## Diagnostics

```bash
mixlab-cluster doctor "${service_args[@]}"
mixlab-cluster coordinator status
```

`doctor` reports local `coordinator_service` independently of remote reachability.
A service-manager success only establishes a registered job observation, not
application readiness. `tcp:HOST:PORT` establishes only a TCP connection from
the doctor process. `coordinator_tls_trust` uses a bounded, read-only authenticated
Info request and checks the exact coordinator identity and installed HTTPS origin.
Without a controller identity it remains `unknown`, even if TCP succeeds.
Local credential/trust failures are distinct from peer TLS rejection. Doctor
does not renew credentials, submit jobs or alter service settings.

Refusal, DNS failure, timeout, TLS rejection and route/permission errors are
observations, not proof of macOS Local Network Privacy denial. BSD sockets expose
no definitive permission query. A successful terminal probe does not establish
the service process's outbound permission, and a failed handshake does not prove
which host's trust or clock is wrong. Inspect both hosts' doctor results and
`~/.mixlab/services/coordinator/service.log`; distinguish routing, Local Network,
Application Firewall and other filters without disabling them.
