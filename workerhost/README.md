# Local Worker Hosting Foundation

This is an internal R1.1 checkpoint, not a managed cluster service. The public
`mixlab-cluster` executable exposes experimental trust setup/enrollment, not
managed worker launch. Node authorization and transport integration are separate
contexts; this package does not expose a remote job endpoint.

The administrator-owned build registry constructs `Supervisor` with an absolute
installed executable and its approved SHA256. Its `LaunchPlan` contains only a
typed `workerjob.Assignment`, private worker directory and bounded hosting
limits. There is no submitted executable path, shell command, argv, environment
map, credential or dynamic-library path. Approval must happen before this API;
matching a file hash does not approve an untrusted executable.

The current local assignment is deliberately separate from a future signed node
manifest: its dataset selector has already been resolved to a local shard glob
by the agent. A remote controller must never populate `local_train_pattern`.
The assignment binds exact config bytes, program/dataset content digests,
executable digest, ordered membership, local member/rank, job and attempt.
Only fresh same-host IPv4-loopback ring attempts are supported. An encrypted,
agent-owned transport plan is required before adding cross-host execution.

## Lifecycle

1. Validate the approved build and private attempt directory. Publish a private
   ring file and create an exclusive, byte-limited log.
2. Start the approved executable in a new process group with a closed allowlist
   of arguments/environment. Pass the capability by inherited anonymous pipe,
   never by argv, environment or a persistent credential file.
3. Authenticate the actual child PID/UID and exact assignment binding using
   `workercontrol/local`. Deliver the immutable assignment once.
4. Worker verifies its own executable, parses the assigned config, checks
   dataset/program hashes and exact loopback launch inputs. Strict MLX bootstrap
   validates actual rank/world against ordered membership before loader use.
5. Worker reports readiness after trainer initialization, periodic heartbeats,
   and log-cadence optimizer progress. Heartbeat means process/control-thread
   liveness, **not** proof that a GPU operation is making progress.
6. Success requires a successful authenticated terminal event and exit zero.
   Failure, disconnection, context cancellation, startup/runtime expiry, log
   exhaustion or malformed/replayed protocol closes admission and terminates
   the process group, with a bounded SIGTERM grace followed by SIGKILL. Native
   calls may ignore cooperative cancellation; hosting owns the hard deadline.
7. Always reap the child and remove the control socket. Retain the bounded log
   and non-secret ring file for the owning attempt journal. Callers decide when
   to delete the attempt directory; do not reuse it.

Exit observation now retains the leader PID until process-group cleanup, using
Darwin process events or Linux `waitid(WNOWAIT)`. Signals are serialized with
reaping, and no numeric process-group signal is sent after reaping. Failure to
confirm group cleanup returns `ErrReconciliationRequired`, which cannot publish
a no-child terminal outcome. Approved workers must not detach descendants into
another process group. This is not hostile-process containment; stronger
descendant containment remains outside the trusted-worker contract.

`Run` owns a single child, not a cohort. Its eventual node/coordinator caller
must cancel the whole fixed world when any participant fails. No auto-restart
is performed. Artifact/checkpoint publication, process-level recovery and an
idle-progress watchdog are not implemented in this foundation. Startup time,
total runtime, shutdown grace, log bytes and IPC frame/message/byte budgets are
enforced separately from the monitored resource budgets below.

## Monitored Resource Budgets

Every launch requires the approved CPU, memory, disk and log budgets; none
silently defaults to unlimited. The owner samples immediately after launch and
then approximately once per second, with a two-second sampling deadline.
Exceeding a budget or failing to obtain a sample cancels supervision and invokes
the same process-group termination/reaping path as other worker failures.

- CPU is cumulative observed user/system CPU time across the owned process
  group, not a CPU-utilization percentage. Observed consumption is retained
  after children exit. Short-lived children that disappear between samples
  cannot be fully accounted for.
- Memory is summed process-group RSS. Shared pages may be counted more than
  once; this is not GPU VRAM accounting or a hard unified-memory allocation cap.
- Disk is logical file size under the private attempt directory, including
  retained logs and scratch/output files. Symlinks are not followed outside
  the directory; sparse files and hard links are conservatively counted by
  logical size. Dataset files elsewhere and unlinked-but-open files are not
  included. Disk is also checked before launch and before successful return.
- The sampler bounds process output to 8 MiB, entries to 100,000 and directory
  depth to 128. Exceeding a sampler bound fails the attempt rather than silently
  disabling monitoring. Process sampling shares the PID ownership/reap fence.

These are monitored termination budgets, **not hard OS quotas**. Brief spikes
can be missed, and sampling plus shutdown grace permits overshoot. Local
filesystem stalls and OS scheduling may delay observation; the sampling deadline
is not a kernel I/O preemption guarantee. This approved R1.1 policy does not
provide containment against workers that escape the process group or write
outside the attempt directory. Such behavior is excluded by the trusted,
administrator-approved-worker contract.

This supports trusted administrator-controlled hosts only. Approved executable,
library and dataset files must remain immutable while in use. It is not a
sandbox against a hostile same-UID/root process, and the raw MLX loopback socket
does not authenticate other local processes.

## Durable Execution Contract

`AttemptStore` now records immutable approved attempts before calling a launcher
port, publishes started/terminal outcomes, and refuses to launch an existing
nonterminal attempt on retry. A permanent claim prevents missing execution
history from being mistaken for a fresh directory. An unstarted cancellation
is a durable hosting fence; later start requests cannot bypass it.

`LocalRunner` now carries the exact approved resource budgets into `Supervisor`,
publishes the started callback before assignment delivery, and persists a
physical exit receipt only after confirmed cleanup. The execution journal then
publishes its terminal outcome. Resource-monitor failures remain explicit failed
outcomes, not successful exits. A permanent physical claim fences direct retries.

Recovery requires independent evidence that no child survives. `LocalRunner`
accepts the exact physical exit receipt, or a changed host boot UUID proving
that processes from the prior boot cannot survive. A same-boot interruption
without that receipt remains fenced; a missing or apparently dead saved PID is
not confirmation.

`GuardianRunner` runs this supervisor in a per-attempt helper of the existing
GPU-free `mixlab-cluster` executable. It has no listener, remote API or workload
credentials. An inherited bounded Unix channel carries the approved assignment
and agent liveness. Agent death closes that channel; the guardian terminates and
reaps its worker even when the worker cannot cooperate inside a native call.
The durable pending/owned/done journal serializes delayed launch against restart
fencing. Recovery consumes exact physical cleanup receipts, including a crash
between receipt publication and the guardian's final journal update. If both
agent and guardian die before cleanup is proved, the node remains fenced. A slow
guardian is not forcibly killed by the parent, which would discard its worker
ownership. Interrupted attempts fail rather than silently rerun, even when a
successful terminal response may have been lost.

`RuntimeStore` allocates per-attempt directories with a separate durable index.
Interrupted empty-directory publication can finish, but a missing published
directory or journal cannot be recreated as a fresh execution. Long persistent
paths use a short owner-only temporary control-socket directory; normal cleanup
removes it without deleting the persistent execution history. An abnormal helper
kill may leave an inert temporary socket directory, not a reusable credential.
The experimental operational agent composes these ports; see
[node hosting](../docs/cluster-agent.md). Final cross-host release acceptance
remains separate from these local hosting tests.

## Verification

`go test -race ./workerhost ./workerjob` exercises real child admission,
readiness/progress/exit semantics, bounded logs, failures, cancellation and
cleanup. Native sockets/processes must be permitted by the test environment.

Build an MLX trainer, then set `MIXLAB_MANAGED_CLI` to its absolute path and run
`go test ./train -run '^TestManagedWorkerCLITraining$' -count=1 -v`. This runs
two real local MLX ranks through the production managed entry point, checks
committed optimizer attempts and rank-agreed loss, and deletes test artifacts.

Set `MIXLAB_CLUSTER_CLI` to the matching control executable and run
`go test ./train -run '^TestManagedGuardianCLITraining$' -count=1 -v` to exercise
real native training through durable runtime allocation, the guardian executable,
and terminal-retry/cleanup checks.
