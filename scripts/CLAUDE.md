# scripts/ — data preparation and the RunPod handler

Two unrelated concerns share this directory. Both ship inside the RunPod image
(`docker/runpod.Dockerfile` copies `scripts/` and runs `handler.py` as the
container `CMD`), so a change here reaches production only through an image
rebuild and a new pinned digest — never through a Go rebuild.

## Data preparation
`prepare.py` and `prepare_records.py` turn text/JSONL/FASTA/npz into shards plus
a manifest. The Python writer and the Go reader are a **byte-exact contract**:
magic, version, header size, field order, and endianness must change together.
See [`../data/CLAUDE.md`](../data/CLAUDE.md) for the formats and the manifest
invariants. `assets.go` embeds these two scripts in the binary, which is why
`mixlab -mode prepare` works with no source checkout.

## RunPod handler
`handler.py` builds the mixlab argv, runs setup/main/post, and returns output.
Job-input fields and their semantics are documented in
[`../docker/README.md`](../docker/README.md#runpod-serverless).

- `handler_process.py` — drains stdout and stderr **concurrently** via
  `selectors`. The original drained stdout to EOF first, so a child writing more
  than a pipe buffer to stderr deadlocked the worker forever; the job `timeout`
  could not interrupt it, because the block happened before `wait()`. It also
  bounds what the old path left unbounded: capture tail, console line length,
  and a wall-clock deadline across the whole drain. Cleanup kills the child's
  **process group**, since killing only the direct child leaves a shell
  wrapper's trainer running with its GPU allocation.
- `handler_watchdog.py` — optional stall detection keyed on *committed optimizer
  steps* read from the trainer's progress file, never on console traffic or GPU
  queries, so a wedged GPU is detectable without calling into it.
- `handler_log.py` — one bounded background queue for dashboard writes. Printing
  inline from the drain loop meant a slow or blocked log sink could stall pipe
  draining, the deadline, and the watchdog.

Validation that can be decided from the job input alone runs **before** setup
commands, so a job that cannot succeed does not first download data and write to
the volume.

## Conventions
- Tests are plain `unittest` and run in CI (`python -m unittest discover -s
  scripts -p '*_test.py'`). They ran only by hand until that step was added —
  worth remembering, because nothing else covers this directory.
- Keep the handler dependency-light: `handler_process.py` deliberately imports
  no RunPod package so it stays testable off-platform.
