# arch/ — IR builder, block emitters, config

This package builds the IR program from a JSON `ArchConfig`. It runs in pure Go (no GPU, no cgo).

## Key files
- `ir.go` — IR types, op codes, `Program` builder methods. Op codes are stable IDs (don't reuse).
- `config.go` — `ArchConfig`, `BlockSpec`, validators
- `registry.go` — block dispatch (`builtinBlockEmitter`) + public delegation API
- `recurrent_blocks.go` — emit functions for mamba/mamba3-canonical/rwkv/retnet/gated_deltanet
- `blocks.go` — emit functions for plain/swiglu/mlp/perceiver/bottleneck/etc.
- `weight_shapes.go` — per-block `WeightMeta` for the optimizer
- `objective.go` — training objectives + validators (causal/mlm/mntp/hybrid/block-diffusion/multihead/classification)
- `ir_bridge.go` — `Build{IR,Training,Eval,Generation}IRProgramFromConfig`; assembles backbone + head per objective/state (`TrainingProgramState`)
- `classification_ir.go` — classification head: truncate backbone at `x_hidden`, pool, linear + cross-entropy

## Objectives & non-block IR heads
Objective/task heads (classification, batched-generation gather) are built by
**composing existing ops** — no new op code, no `gpu/` change. Only a genuinely
new primitive needs the full add-a-block dance below. `Dropout` is **keyed**: it
declares a `dropout_keys` input + per-op ordinal so masks are deterministic from
`(seed, step, ordinal)` — the prerequisite for reproducible resume.

## Public registry-delegation API (v0.19.1+)
For downstream packages (mixlab-jazz) that compose registered blocks inside custom containers:
```go
arch.EmitBlock(prog, spec, stream, wi, D, T, B, V, idx, opts) (int, error)
arch.BlockWeightCount(spec, blockScales, residMix) (int, error)
arch.BlockWeightShapes(spec, D, T, B, V) ([]WeightMeta, error)
```

## Adding a new block
1. Pick a unique op code in `ir.go` (use the next free integer; don't reuse retired codes).
2. Add a `(p *Program) MyBlock(...)` helper in `ir.go` that calls `p.AddOp(OpMyBlock, ...)`.
3. Add `WeightMeta` for the block's weights in `weight_shapes.go::builtinBlockWeightShapes`.
4. Add an emitter in `blocks.go` or `recurrent_blocks.go`.
5. Register the type in `registry.go::builtinBlocks` with both emitter and weight-shape function.
6. Add `case "myblock"` to `validateBlockSpec` in `config.go` if the type has structural constraints.
7. Update `docs/config-reference.md` with the new block's options.
8. Add a runtime case in `gpu/ir.cpp` for the op (and Metal/CUDA primitive if needed — see `gpu/CLAUDE.md`).

## Naming gotcha
`mamba3` is a deprecated alias for `gated_linear_ssm` (the simplified gated-linear-scan, not canonical Mamba-3). The canonical Mamba-3 block is named `mamba3-canonical`. Both `mamba3` and `gated_linear_ssm` resolve to the same emitter; configs using `mamba3` get a one-time deprecation warning. The `mamba3` slot will be reassigned to canonical Mamba-3 in a future release.
