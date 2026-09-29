# Managed Clusters: Getting Started (Experimental)

> **Experimental.** `mixlab-cluster` trains one model across several Macs on a
> trusted LAN. It works, but it is not yet convenient: every service runs in the
> foreground of a terminal window or SSH session, and upgrades need manual steps.
> Read [Current limitations](#current-limitations) before you start. For training
> on one machine, keep using `mixlab` as before; nothing here changes it.

## How it works

`mixlab-cluster` never trains. It manages identities and machines, and starts the
ordinary `mixlab` binary on each node to do the training. One executable plays
several roles:

| Role | Command | Runs on | Job |
|------|---------|---------|-----|
| Authority | `mixlab-cluster init`, `authority serve` | one machine, usually the controller | Holds the cluster identity, issues credentials, and serves the signed trust that every node refreshes. |
| Controller | `mixlab-cluster nodes`, `submit` | the machine you work from | Lists nodes, submits a training job, and downloads the result. Does not train. |
| Node agent | `mixlab-cluster agent` | every training machine | Accepts authenticated jobs, carries encrypted traffic between nodes, and starts `mixlab` as a supervised child to train. |

Every machine that trains needs both `mixlab` and `mixlab-cluster`. The
controller can also be a node.

| Port | Listener | Needed by |
|------|----------|-----------|
| 7443 | authority (`authority serve`, and `invite` while enrolling) | nodes refreshing trust; enrolling machines |
| 7445 | node agent | the controller |
| 7446 | node agent relay | other nodes during training |

This is the **managed** path. The [unmanaged distributed workflow](distributed-training.md)
runs `mixlab` directly on each machine with `mlx.launch` and needs no enrollment.

## Current limitations

- **Foreground only.** The authority and every node agent must keep running in an
  open Terminal window or SSH session. On macOS, a process detached from its
  session (`nohup`, `screen`, a closed SSH connection) loses Local Network access:
  a node agent then cannot refresh trust, and after 15 minutes it rejects the
  controller. Background service mode is planned.
- **Run commands from Terminal.app or SSH.** macOS grants Local Network access per
  app. Terminal.app and SSH sessions have it; some third-party terminals do not,
  even with the setting enabled. See [Troubleshooting](#troubleshooting).
- **Upgrades need re-registration.** Each node pins the exact `mixlab` and
  `mixlab-cluster` builds it runs. After an upgrade the agent refuses to start
  until you register it again; see [Upgrading](#upgrading).
- **Trusted machines only.** All hosts must be yours and administered by you. This
  is not isolation from other users of the same machine.
- **Fixed membership.** A job uses exactly the nodes you name; there is no
  automatic recovery or elastic membership.
- **Apple Silicon Macs.** The managed path is tested on macOS with Metal.

## Before you start

- Two or more Apple Silicon Macs on the same network, each with a fixed LAN
  address for the duration.
- `mixlab` and `mixlab-cluster` on every machine, the **same version** everywhere.
  `brew install mrothroc/tap/mixlab` installs both; the signed disk image on each
  [GitHub release](https://github.com/mrothroc/mixlab/releases) contains both as well.
- A prepared dataset. Follow the [README quickstart](../README.md#quickstart) once,
  then copy the **identical** shard files to the same kind of location on every
  node. The examples below use `~/mixlab-data/train_*.bin`.
- `jq`, used below to read JSON output (`brew install jq`).

Run every `mixlab-cluster` command below from **Terminal.app** or an **SSH
session**. Over SSH, the login Keychain is locked, so add `-key-backend file` to
`init` and `enroll` there; locally on macOS the default Keychain storage is used.

The examples use a controller at `192.168.1.10` that is also a node, and a second
node at `192.168.1.20`. Substitute your own addresses.

## 1. Create the cluster (controller)

```bash
mixlab-cluster init -trust-listen 192.168.1.10:7443 > cluster.json
AUTHORITY=$(jq -r .authority_dir cluster.json)
CONTROLLER=$(jq -r '.principals[] | select(.role=="controller") | .directory' cluster.json)
jq -r .identity.phrase cluster.json
```

`-trust-listen` must be the controller's **LAN address**. The default,
`127.0.0.1:7443`, works on one machine but no other machine can ever enroll.
`cluster.json` contains public identifiers and directory paths, no secrets. Keep
it: later steps read the authority and controller directories from it.

## 2. Enroll each training machine

Repeat for every node, including the controller if it trains. On the
controller, create a one-time invitation:

```bash
mkdir -m 700 -p ~/mixlab-invitations
mixlab-cluster invite -cluster-state-dir "$AUTHORITY" \
  -invite-output ~/mixlab-invitations/node.json
```

`invite` stays in the foreground, listening on port 7443, until the invitation is
used or expires after ten minutes; to enroll the controller itself, run `enroll`
in a second window. Copy the file privately to the node, for
example `scp`, into an owner-only directory, and delete the controller's copy
once it is delivered. The file holds a single-use secret: never paste it into
chat, logs or command lines.

On the node:

```bash
mixlab-cluster enroll -enrollment-policy provisioned \
  -enrollment-provisioning-file ~/mixlab-invitations/node.json \
  -principal-state-dir ~/.mixlab/node-identity
```

`enroll` deletes the invitation it consumed. `invite` and `authority serve` share
port 7443, so enroll every node before step 3, or stop the authority while you
enroll another. [Enrollment](cluster-enrollment.md) describes two alternatives:
verified pairing by comparing phrases, and trusted-LAN auto-enrollment.

## 3. Start the authority (controller)

In its own window, and leave it running:

```bash
mixlab-cluster authority serve -cluster-state-dir "$AUTHORITY"
```

Nodes refresh their trust from it. If it stops, nodes keep working for up to 15
minutes, then refuse jobs until it returns.

## 4. Register and start each node

On each node, with that node's own LAN address:

```bash
mixlab-cluster agent init \
  -principal-state-dir ~/.mixlab/node-identity \
  -agent-state-dir ~/.mixlab/node-agent \
  -worker-binary "$(command -v mixlab)" \
  -agent-relay-listen 192.168.1.20:7446 \
  -dataset "train=$HOME/mixlab-data/train_*.bin"

mixlab-cluster agent -agent-state-dir ~/.mixlab/node-agent -agent-listen 192.168.1.20:7445
```

`agent init` records the exact `mixlab` and `mixlab-cluster` builds and the
dataset's content identity; `train` is the name jobs use to select it. The agent
prints `"Status":"ready"` and stays in the foreground. Leave it running.

## 5. Check the nodes (controller)

```bash
mixlab-cluster nodes -principal-state-dir "$CONTROLLER" \
  -node 192.168.1.10:7445 -node 192.168.1.20:7445 > nodes.json
jq -c '.[] | {endpoint, reason}' nodes.json
DATASET=$(jq -r '[.[].capabilities.datasets[]? | select(.selector=="train") | .id] | unique | .[]' nodes.json)
echo "$DATASET"
```

Every node must report `"reason":"available"`, and `DATASET` must be exactly one
ID. Two IDs mean the nodes' shards differ.

## 6. Write a config

A managed run needs `training.distributed` with the `ring` backend and the AdamW
optimizer:

```json
{
  "name": "cluster_quickstart",
  "model_dim": 128,
  "vocab_size": 1024,
  "seq_len": 128,
  "blocks": [
    {"type": "plain", "heads": 4},
    {"type": "swiglu"},
    {"type": "plain", "heads": 4},
    {"type": "swiglu"}
  ],
  "training": {
    "optimizer": "adamw",
    "steps": 50,
    "lr": 0.001,
    "seed": 42,
    "batch_tokens": 1024,
    "distributed": {"mode": "ddp", "backend": "ring"}
  }
}
```

Save it as `cluster_model.json` and check it with
`mixlab -mode validate -config cluster_model.json`. `vocab_size` 1024 matches
the README quickstart data.

## 7. Train (controller)

```bash
mixlab-cluster submit -principal-state-dir "$CONTROLLER" \
  -worker-binary "$(command -v mixlab)" -config cluster_model.json \
  -workers 2 -train train -dataset-id "$DATASET" \
  -node 192.168.1.10:7445 -node 192.168.1.20:7445 \
  -attempt-state-dir ~/mixlab-runs/first
```

The controller needs the same `mixlab` version as the nodes, and an Apple
Silicon GPU to inspect the config. `-attempt-state-dir` must be a new directory
for each job. On success the command prints `"status":"succeeded"`, and the
trained weights are at `~/mixlab-runs/first/model.safetensors`:

```bash
mixlab -mode generate -config cluster_model.json \
  -safetensors-load ~/mixlab-runs/first/model.safetensors \
  -max-tokens 16 -temperature 0 -prompt token_ids:5,17
```

[Node hosting](cluster-agent.md) covers checkpoint and resume, `submit -abort`,
`submit -fetch`, and resource limits.

## Upgrading

Upgrade `mixlab` and `mixlab-cluster` on every machine to the same version. On
each node:

1. Stop the agent.
2. Register again into a **new** agent directory, reusing the node identity: run
   step 4 with `-agent-state-dir ~/.mixlab/node-agent-2`. `agent init` refuses an
   existing directory, and the agent refuses the old one because the approved
   builds changed.
3. Start the agent with the new directory. Keep the old directory until no job
   from before the upgrade needs cleanup.

With Homebrew, each upgrade produces a newly built binary, so macOS asks once, at
the machine, for access to each Keychain-stored key. A node using
`-key-backend file` does not ask.

## Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| `nodes` shows `unavailable_or_unauthenticated` for a node, but `nc -z <node> 7445` connects | Usually one of two things. **(1)** The node agent's trust is stale: its output shows `node trust refresh: ... no route to host`. Restart the agent from Terminal.app or an open SSH session, and keep the authority running. **(2)** The terminal running `nodes` has no Local Network access. Run it from Terminal.app. |
| `connect: no route to host` to another Mac, although `ping` works | macOS Local Network privacy. Programs started from Terminal.app and SSH sessions have access. Check System Settings → Privacy & Security → Local Network; a third-party terminal can be denied even when shown as enabled. |
| `key-store backend unavailable; explicit backend selection required` | The Keychain is not usable in this session, typically over SSH. Use `-key-backend file`. |
| `approved executable changed; explicit local reapproval required` | `mixlab` or `mixlab-cluster` was upgraded. Follow [Upgrading](#upgrading). |
| `unsafe state path: not a real directory` | A state directory path passes through a symbolic link, such as `/tmp` or `/var` on macOS. Use a real path under your home directory. |
| `training.distributed: optimizer must be adamw` | Add `"optimizer": "adamw"` to `training`. |
| `enroll` cannot reach the authority | `init` used `127.0.0.1`, or the controller's `-trust-listen` address is not its LAN address. Create the cluster again with the LAN address. |

## Removing a test cluster

Stop the agents and the authority with Ctrl-C. Then remove `~/.mixlab` on every
machine, along with any `-attempt-state-dir` directories. On a Mac that used the
default storage, also delete the Keychain items whose names start with
`org.mixlab.signing.v1`, using Keychain Access.

## Reference

- [Cluster initialization](cluster-initialization.md): all `init` flags, recovery, and the trust model.
- [Enrollment](cluster-enrollment.md): verified and trusted-LAN enrollment, revocation.
- [Node hosting](cluster-agent.md): agent limits, submission, checkpoint and resume.
- [Rekey and reenrollment](cluster-rekey.md): replacing compromised cluster keys.
