# Matched DiLoCo experiment tools

These bounded tools compare the production local learner on one host with the
managed two-host DiLoCo path. They are experiment tooling, not a new training
CLI. No throughput or LAN-speedup claim follows merely from completing a run.

## Data

`prepare.py` streams the first 20,000 documents from the pinned revision of
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), subset
`sample-10BT` (ODC-BY). It removes exact duplicates, assigns whole documents by
text hash, then fits ByteLevel BPE on training documents only. This is a small
web-language experiment, not the official FineWeb benchmark or a domain-disjoint
evaluation. Near duplicates are not additionally filtered. It uses Mixlab's
existing shard writer and records source IDs, hashes and dependency versions.

```sh
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python experiments/diloco_p8/prepare.py \
  --output /private/tmp/mixlab-p8-data --docs 20000 --vocab-size 8192
```

Use `datasets==4.8.4`, `tokenizers==0.22.2`, and `numpy==2.4.4` to reproduce the
recorded preparation. Existing output directories are rejected. Review source
data licensing before distributing prepared artifacts.

## Measurement

Build `./experiments/diloco_p8` with the same MLX headers/library as the managed
workers, using the repository's MLX build environment. The baseline calls
`LocalLearnerSession` and `EvaluateLocalLearnerModel` directly. It deliberately
does not use outer optimization or collectives. This is a **matched inner-learner
baseline**, not the fastest possible generic trainer: it includes a model/state
snapshot every H steps and does not use the main training loop's lookahead.

Create a private host manifest containing `remote`, `baseline`, `train_pattern`,
`validation_pattern`, `validation_batches`, `cluster`, `worker`, `controller`,
`coordinator_endpoint`, `coordinator_id`, `coordinator_state`, `nodes`,
`train_hash`, and `validation_hash`. All executable/config/data paths must exist
at the same absolute paths on both Macs. Register `p8-train` and `p8-validation`
in the agents' catalogs using the supported CLI. Preserve original service
configurations and restore them after testing. The monitor currently targets
the experiment's wired `en7` interface on both Macs.

```sh
.venv/bin/python experiments/diloco_p8/measure.py \
  --manifest /private/tmp/mixlab-p8-evidence/hosts.json \
  --arm m1 --config /private/tmp/mixlab-p8-evidence/seed17.json \
  --output /private/tmp/mixlab-p8-evidence/seed17-m1 --require-idle
```

Repeat for `m4` and `pair` in the preregistered seed rotation. Never run arms
concurrently. The pair uses the existing managed coordinator and authenticated
agents. A failed observer/submission does not prove the managed job stopped;
inspect its run ID and leases before another arm or cleanup. Outputs are never
overwritten. Three idle samples are required before each final arm. A blocked
preflight submits no training job. Retain preflight output in the evidence log.

## Analysis

```sh
.venv/bin/python experiments/diloco_p8/analyze.py \
  --registration /private/tmp/mixlab-p8-evidence/registration.json \
  /private/tmp/mixlab-p8-evidence/seed{17,29,43}-{m1,m4,pair}
.venv/bin/python -m unittest discover -s experiments/diloco_p8 -p 'test_*.py'
go test ./experiments/diloco_p8
```

Freeze target, tolerance, configs, seeds, placement and exclusion rules before
final runs. All failures and target misses stay in the report. Repeat an entire
contaminated three-arm seed session, retaining its first attempt; never rerun
just the slow arm. Report both attempts and explain exclusion explicitly.

If the user explicitly permits a loaded-host observation before final runs,
retain that amendment and omit `--require-idle`. Keep all load flags and arms;
do not subtract guessed interference or interpret the results as idle-host
scaling. The recorded P8 experiment used this amended policy because the M1's
background load could not be reduced.

Time-to-target is the first observed validation crossing, without interpolation.
Baseline times include process/SSH overhead. Pair times include admission,
training, transfer, synchronization, reduction, recovery and global validation.
Both train to the fixed horizon even after crossing. Pair compute/token counts
are summed across both workers; more tokens are not themselves a speedup.

The history reader verifies the committed hash chain. Artifact bytes are logical
payload estimates, not TLS/wire bytes. Interface counters include all traffic.
Worker timestamps have one-second resolution; compute intervals include private
snapshot/delta persistence, and transfer-inclusive envelopes overlap. Exclusive
transport time and JIT time are not independently instrumented. Do not subtract
those envelopes to manufacture an exclusive breakdown.

Coordinator `COMPLETE` proves numerical run completion, not a clean process
shutdown. Preserve node attempt outcomes and service/worker logs separately;
post-completion poll rejection or cancellation must remain visible in the
review report even when all global models and evaluations committed.
