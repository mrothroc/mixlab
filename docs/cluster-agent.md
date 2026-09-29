# Experimental Node Hosting

R1.1 provides experimental managed training on trusted, administrator-controlled
hosts. Foreground node hosting, authenticated `nodes`, and fixed-world
`submit` are available. Final weights stream to verified private node-local
storage and download to the controller after successful cleanup. Managed exact
checkpoint/resume is available for successful checkpoint stops. Signed-package
M1/M4 acceptance covers encrypted training, exact resume, enrollment, peer
rejection and fault cleanup. This is not hostile-local-user isolation, and it
does not make a LAN acceleration claim. Do not expose an agent to untrusted hosts.

Start with [managed clusters: getting started](cluster-quickstart.md) for a
tested two-machine walkthrough and the current limitations. In particular, the
agent and the authority must stay in the foreground of a Terminal.app window or
an open SSH session: on macOS a detached process loses Local Network access, its
trust goes stale, and it then rejects every controller.

## Local Setup

Enroll a node using the [enrollment workflow](cluster-enrollment.md). Then
explicitly approve the installed worker and register local prepared shards:

```bash
mixlab-cluster agent init \
  -principal-state-dir /absolute/node-identity \
  -agent-state-dir /absolute/node-state \
  -worker-binary /absolute/installed/mixlab \
  -agent-relay-listen 192.168.1.20:7446 \
  -dataset 'train=/absolute/shards/train_*.bin'
```

Setup records the exact SHA256 of both installed executables, a bounded child
device probe, the local dataset content identity, and the explicit relay bind
address. It creates no listener. It refuses an existing installation; startup
never repairs missing journals or silently approves changed executables.
Keep the installation and associated libraries immutable while jobs run.
Initialization uses the same pinned trust refresh as agent startup. If the
enrollment snapshot has expired, the signed authority endpoint must be
reachable; a still-valid enrolled identity does not need to be enrolled again.
An expired or revoked identity still fails closed.

Setup options:

| Flag | Default | Meaning |
| --- | --- | --- |
| `-state-home` | normal state resolver | Root used when an exact agent directory is omitted |
| `-agent-state-dir` | resolved node directory | New, private node state directory |
| `-principal-state-dir` | required | Enrolled node identity, not a controller/authority |
| `-worker-binary` | required | Absolute installed `mixlab` executable |
| `-agent-relay-listen` | required | Canonical local IP and port, not a wildcard |
| `-name` | `mixlab-node` | Display name |
| `-dataset` | none | Repeatable `NAME=absolute-glob`; paths stay node-local |
| `-max-runtime-seconds` | `86400` | Per-job wall-time ceiling |
| `-max-cpu-seconds` | `172800` | Aggregate monitored child CPU budget |
| `-max-memory-bytes` | `4294967296` | Monitored resident-memory budget |
| `-max-disk-bytes` | `4294967296` | Monitored private attempt disk budget |
| `-max-log-bytes` | `16777216` | Bounded retained worker logs |

CPU, memory and disk limits use monitored termination, not hard OS quotas.
Sampling can overshoot; short-lived processes can escape accounting. These
limits are for trusted workloads on administrator-controlled hosts, not hostile
process isolation. Local datasets must remain immutable during a job.

## Foreground Agent

```bash
mixlab-cluster agent \
  -agent-state-dir /absolute/node-state \
  -agent-listen 192.168.1.20:7445 \
  -agent-advertise off
```

`-agent-listen` defaults to `127.0.0.1:7445`. `-agent-advertise` accepts `off`
(default) or explicit `mdns`; advertisements grant no authority. `-state-home`
may be used instead of an exact state directory when lookup is unambiguous.
Both control and ring listeners belong to `mixlab-cluster`, not the trainer.
Nothing changes firewall settings or system trust.

The agent authenticates controller requests with mutual TLS, verifies signed
manifests, and resolves dataset selectors through its local catalog. The
encrypted ring binds only the configured address. Workload keys stay in the
agent's protected state; children receive only local assignments and opaque
loopback collective traffic. Additional application credential envelopes and
credential-use requests are rejected in R1.1.

Trust refresh and principal renewal use only root-signed authority endpoints.
Startup first attempts a bounded public trust refresh using the existing pin,
key and unexpired node certificate. Current cached trust tolerates an authority
outage; stale trust, revoked identities and expired certificates do not permit
work. Startup refresh never renews an expired certificate or reenrolls a node.
Agents also accept bounded public signed-snapshot pushes from `revoke`; that
route advances pinned trust only and grants no job or artifact access.
Unavailable or stale trust fails closed. Idle device probes never compete with
an active lease. One node execution owner supervises one attempt; restart fences
interrupted work instead of silently restarting training.

On shutdown, the server first stops admitting requests and drains pending
mutations, then cancels and joins workers and encrypted streams. The lease is
released only after physical child cleanup and workload-key destruction are
confirmed. Uncertain cleanup keeps the accelerator unavailable and reports an
error. Runtime journals and bounded diagnostic logs are retained for recovery;
do not delete them to make a busy node appear available.

## Inventory And Submission

Use an enrolled controller identity, never node or authority keys, for remote
operations. Explicit endpoints work without multicast:

```bash
mixlab-cluster nodes -principal-state-dir /absolute/controller \
  -cluster-state-dir /absolute/authority -node 192.168.1.20:7445
```

`-cluster-state-dir` starts a temporary authority service at its root-signed
endpoint and joins it before exiting. Omit it when `authority serve` is already
running. `-authority-endpoint` selects among multiple signed URLs; it cannot add
an unsigned endpoint. `-pool` accepts only `local`. `-discover mdns` explicitly
enables multicast hints; the default is `off`. `-node` is repeatable. The older
`-controller-state-dir` spelling is an alias for `-principal-state-dir` in
`nodes`; do not supply both. JSON inventory includes authenticated capabilities
or a stable rejection reason, not trusted advertisement claims.

The config must set `training.optimizer` to `adamw` and `training.distributed`
to `{"mode": "ddp", "backend": "ring"}`. The `-dataset-id` value is the `id` of
the dataset selector in each node's `nodes` inventory; every recruited node must
report the same one.

Experimental fixed-world submission:

```bash
mixlab-cluster submit -principal-state-dir /absolute/controller \
  -cluster-state-dir /absolute/authority \
  -worker-binary /absolute/installed/mixlab -config model.json \
  -workers 2 -train train -dataset-id EXPECTED_DATASET_SHA256 \
  -node 192.168.1.20:7445 -node 192.168.1.21:7445 \
  -attempt-state-dir /absolute/new-attempt
```

The local approved executable inspects the config and probes device support in
bounded children. The controller itself never links MLX. All recruited workers
must match its build, numerical contract, device family, and the requested
logical dataset identity. The dataset digest comes from the registered prepared
shards, not an advertisement. No remote executable, shell command, environment,
or local data path is accepted. The current submit path requires a matching
local accelerator for capability inspection; recovery does not.

Requested resource flags are `-max-runtime-seconds` (3600),
`-max-cpu-seconds` (7200), `-max-memory-bytes` (4294967296),
`-max-disk-bytes` (4294967296), and `-max-log-bytes` (16777216).
They must fit every node's administrator-approved limits. Remote port conflicts
abort the fixed attempt; the controller never silently changes membership.

The submit process remains alive, renews leases and controller trust, and
monitors the whole group. Failure cancels the fixed cohort. Success requires all
workers to exit successfully and all leases to reach confirmed cleanup. Lost
responses retain the journal; a delayed reserve request cannot undo a recorded
abort. Interrupted attempts never automatically restart training.

Retry incomplete cleanup using only the retained journal:

```bash
mixlab-cluster submit -abort -principal-state-dir /absolute/controller \
  -cluster-state-dir /absolute/authority \
  -attempt-state-dir /absolute/new-attempt
```

Do not delete attempt or node journals to retry a job. Recovery needs no model
config, local worker executable, or GPU. An unreachable node keeps cleanup
pending rather than being reported as released.

Successful submission downloads rank zero's final weights to
`model.safetensors` inside the retained attempt directory. Only the job's owning
controller can read the output, after successful exit and confirmed cleanup.
Transfers are bounded to 1 GiB, use small authenticated chunks with current
trust checks, and verify the complete checksum before publishing the private
file. Output capacity also depends on the approved disk/log budgets.

If training succeeds but download is interrupted, fetch again without starting
workers or requiring a local GPU:

```bash
mixlab-cluster submit -fetch -principal-state-dir /absolute/controller \
  -cluster-state-dir /absolute/authority \
  -attempt-state-dir /absolute/new-attempt
```

An exact retry verifies an existing output; it never overwrites a different or
corrupt file. Retain the node's runtime directory until download succeeds.
Ordinary final weights do not include optimizer or loader state.

## Managed Checkpoint And Resume

Add `-checkpoint-at N` to a normal submission to stop successfully after global
optimizer attempt N and download `checkpoint.mixlab` instead of final weights.
N must be after the restored attempt and at or before the configured training
horizon. It does not shorten that horizon or restart warmup. Skipped optimizer
updates still count as attempts; the bundle also records committed updates.

Resume using the same normal submission flags, config and approved executable,
plus the successful source attempt and a **new** attempt directory:

```bash
mixlab-cluster submit -principal-state-dir /absolute/controller \
  -cluster-state-dir /absolute/authority \
  -worker-binary /absolute/installed/mixlab -config model.json \
  -workers 2 -train train -dataset-id EXPECTED_DATASET_SHA256 \
  -node 192.168.1.20:7445 -node 192.168.1.21:7445 \
  -resume-from /absolute/checkpoint-attempt \
  -attempt-state-dir /absolute/resumed-attempt
```

Omit `-checkpoint-at` on resume to finish the original horizon and download
ordinary final weights. Set a larger `-checkpoint-at` to produce another resume
bundle. If downloading the source was interrupted, use `submit -fetch` first.
Keep the source attempt journal, `download.json`, and `checkpoint.mixlab`
together: the receipt binds the artifact checksum to that retained attempt.

Exact resume requires the same controller, build, config, program, weight layout,
optimizer, logical dataset and ordered worker nodes/members. The new submission
gets a fresh attempt, leases and workload credentials, but keeps the original
run/group identity. Trainer-side compatibility checks restore weights, optimizer
state, sampler, counters and schedule; there is no weights-only fallback.

Checkpoint inputs use authenticated, signed-job-bound chunks with exact-retry
checks. The bounded container contains only a manifest, model tensors and state
tensors; it cannot supply archive paths. Both transfer and extraction verify the
complete SHA256. The 1 GiB artifact limit applies to the whole bundle. Input
staging, extracted state and output copies also consume the approved disk budget;
allow more disk than for weights-only output. Node input chunks and verified
artifacts are retained with the attempt for recovery. Administrators may remove
them only after confirmed cleanup and after retaining any required controller
checkpoint; repeated submissions otherwise accumulate retained data.
Transfer admission also remains bounded by lease expiry and the signed job's
15-minute admission window. A very slow transfer or large serial cohort can
exceed that window; the attempt aborts and cleans up rather than silently
extending authorization.

This is explicit successful-stop/resubmit, not automatic recovery from a failed
job, periodic checkpoint retrieval from a failed job, elastic membership, or
dataset transfer. Interrupted attempts must finish cleanup before resubmission.
