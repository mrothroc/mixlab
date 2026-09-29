# Managed Clusters: Getting Started (Experimental)

> **Is it worth it?** On a LAN, distributed training is usually *slower* per
> token than the fastest Mac alone. Read
> [When distributed training helps](distributed-when.md) before you start.

> **Experimental.** `mixlab-cluster` trains one model across several Macs on a
> trusted LAN. Use the signed package for macOS background services and complete
> the initial Local Network approval at each Mac's desktop.
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

- **Logged-in macOS session required.** Signed per-user LaunchAgents run without
  an open terminal, but stop at logout and return after GUI login. They do not
  run before login. Do not substitute `nohup` or `screen` for service installation.
- **Run commands from Terminal.app or SSH.** macOS grants Local Network access per
  app. Terminal.app and SSH sessions have it; some third-party terminals do not,
  even with the setting enabled. See [Troubleshooting](#troubleshooting).
- **Upgrades need explicit reapproval.** Each node pins the exact `mixlab` and
  `mixlab-cluster` builds it runs. After an upgrade the agent refuses to start
  until you reapprove the new binaries; see [Upgrading](#upgrading).
- **Trusted machines only.** All hosts must be yours and administered by you. This
  is not isolation from other users of the same machine.
- **Fixed membership.** A job uses exactly the nodes you name; there is no
  automatic recovery or elastic membership.
- **Apple Silicon Macs on macOS 26.** The signed package, which background
  services require, is built for macOS 26. On macOS 15, use the Homebrew formula
  and run the authority and agents in the foreground.

## Before you start

- Two or more Apple Silicon Macs on the same network, each with a fixed LAN
  address for the duration.
- `mixlab` and `mixlab-cluster` on every machine, the **same version** everywhere,
  from the signed package:

  ```bash
  brew unlink mixlab 2>/dev/null   # only if the source-built formula is installed
  brew install --cask mrothroc/tap/mixlab-signed
  ```

  Or copy the whole directory from the signed disk image on the
  [release](https://github.com/mrothroc/mixlab/releases) to a stable path. The
  source-built formula (`brew install mrothroc/tap/mixlab`) is for foreground and
  development use: macOS background services require the signed package, and the
  macOS firewall can silently block an unsigned node.
- A prepared dataset. Follow the [README quickstart](../README.md#quickstart) once,
  then copy the **identical** shard files to the same kind of location on every
  node. The examples below use `~/mixlab-data/train_*.bin`.
- `jq`, used below to read JSON output (`brew install jq`).

Run every `mixlab-cluster` command below from **Terminal.app** or an **SSH
session**. The login Keychain may be unavailable or locked over SSH. For a
file-backed setup, explicitly add `-key-backend file` to both `init` and `enroll`;
otherwise the default macOS Keychain backend must be usable in that session.
Services never silently switch key-storage backends.

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

Install the authority service in the logged-in user session:

```bash
mixlab-cluster authority install -cluster-state-dir "$AUTHORITY"
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

mixlab-cluster agent install -agent-state-dir ~/.mixlab/node-agent -agent-listen 192.168.1.20:7445
```

`agent init` records the exact `mixlab` and `mixlab-cluster` builds and the
dataset's content identity; `train` is the name jobs use to select it. Click Allow
for Local Network access at each Mac's desktop. Inspect
`~/.mixlab/services/agent/service.log` for `"Status":"ready"` and run
`mixlab-cluster doctor -agent-state-dir ~/.mixlab/node-agent`. Closing the setup
terminal no longer stops the service.

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

1. Finish jobs and wait for cleanup, then run `agent stop` on each node and
   `authority stop` on the authority host.
2. Upgrade the matched signed package at the same installation path.
3. Run `mixlab-cluster agent reapprove -agent-state-dir ~/.mixlab/node-agent
   -worker-binary "$(command -v mixlab)"` on each node.
4. Run `authority start`, then `agent start` on every node and check `nodes`.

Do not create a new agent directory: reapproval preserves identity and history.
See [upgrade and key-storage details](cluster-services.md#upgrade-without-reenrollment).

## Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| `nodes` reports TLS rejection or an unknown failure although TCP connects | Run `doctor` on both hosts; inspect the agent's service log for stale trust or refresh failure. Keep the authority service running. TCP reachability alone does not establish identity. |
| `connect: no route to host` to another Mac, although `ping` works | Check routing and Local Network privacy. Allow the signed LaunchAgent at the desktop. This socket error alone is not proof of permission denial. |
| `key-store backend unavailable; explicit backend selection required` | The Keychain is not usable in this session, typically over SSH. Use `-key-backend file`. |
| `approved executable changed; explicit local reapproval required` | `mixlab` or `mixlab-cluster` was upgraded. Follow [Upgrading](#upgrading). |
| `unsafe state path: not a real directory` | A state directory path passes through a symbolic link, such as `/tmp` or `/var` on macOS. Use a real path under your home directory. |
| `training.distributed: optimizer must be adamw` | Add `"optimizer": "adamw"` to `training`. |
| A node reports `timeout` although `nc -z <node> 7445` connects, and the node shows connections to port 7445 in `CLOSE_WAIT` (`netstat -an -p tcp`) | The macOS Application Firewall is withholding connections from an unsigned `mixlab-cluster`. Use the signed package on that node; it is allowed with the firewall on. Do not disable the firewall. |
| `enroll` cannot reach the authority | `init` used `127.0.0.1`, or the controller's `-trust-listen` address is not its LAN address. Create the cluster again with the LAN address. |

## Uninstalling

Remove a cluster in this order on **every** machine: services first, then
Keychain items, then state, while the state still records which keys are yours.

1. Finish or abort jobs (`mixlab-cluster submit -abort -principal-state-dir "$CONTROLLER" -attempt-state-dir DIR`) and
   copy any `model.safetensors` or `checkpoint.mixlab` you want to keep.
2. Remove the services:

   ```bash
   mixlab-cluster agent uninstall        # on each node
   mixlab-cluster authority uninstall    # on the authority host
   ```

3. Remove this cluster's Keychain items. Each key set records its Keychain scope
   in a `key-context.json` beside it, so this deletes only keys belonging to the
   state under `~/.mixlab`:

   ```bash
   find ~/.mixlab -name key-context.json \
     -exec jq -r 'select(.backend=="keychain") | .scope' {} \; | sort -u |
   while read -r scope; do
     while security delete-generic-password -s "org.mixlab.signing.v1.$scope" >/dev/null 2>&1; do :; done
   done
   ```

   If other clusters you want to keep also live under `~/.mixlab`, limit `find` to
   this cluster's directories instead.
4. Remove the state: `rm -rf ~/.mixlab`, or only this cluster's directories, plus
   any `-attempt-state-dir` directories and `~/mixlab-invitations`.
5. Remove the software:

   ```bash
   brew uninstall --cask mrothroc/tap/mixlab-signed   # or: brew uninstall mrothroc/tap/mixlab
   brew untrust mrothroc/tap && brew untap mrothroc/tap   # only if nothing else uses the tap
   ```

6. Optionally, switch `mixlab-cluster` off in System Settings → Privacy &
   Security → Local Network. macOS keeps the entry but it no longer grants access.

## Reference

- [Background services](cluster-services.md): install, upgrades, logs and diagnostics.
- [Cluster initialization](cluster-initialization.md): all `init` flags, recovery, and the trust model.
- [Enrollment](cluster-enrollment.md): verified and trusted-LAN enrollment, revocation.
- [Node hosting](cluster-agent.md): agent limits, submission, checkpoint and resume.
- [Rekey and reenrollment](cluster-rekey.md): replacing compromised cluster keys.
