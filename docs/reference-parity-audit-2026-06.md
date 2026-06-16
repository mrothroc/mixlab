# Reference-Parity Audit - 2026-06-16

## Scope

This audit reviews Mixlab areas where a bug can pass ordinary unit tests because
the implementation and the test oracle share the same local interpretation of a
paper, reference model, or export template.

Coverage labels:

- `External parity`: compares Mixlab against an external implementation, fixture,
  or independently shipped runtime.
- `CPU oracle`: compares GPU/IR behavior against an independent local oracle.
- `Self-consistency`: compares two Mixlab paths, such as fused vs expanded IR.
- `Smoke/invariant`: verifies finite training, shapes, validation, or broad
  behavioral invariants.

## Findings

| Risk | Area | Current coverage | Gap | Recommended action |
| --- | --- | --- | --- | --- |
| Low | `hgrn2` and `mlstm` blocks | Weight-shape, IR emission, MLX scan vs local CPU oracle, static scan-reference fixtures, static full-block reference fixtures, finite training smoke | Full block projection layout, normalization, output gating, scan, and residual wiring are now covered by checked-in independent scalar fixtures. The remaining gap is no fixture generated directly from a pinned upstream HGRN2/xLSTM implementation. | Current v1 coverage is adequate; consider pinned upstream fixtures only if importing upstream weights or changing block semantics. |
| Low | `training.data2vec` runtime | Config/IR tests, SmoothL1 GPU CPU oracle, deterministic tests for tau scheduling, EMA mixing, target averaging, target normalization, hybrid/causal skip, hybrid training smoke, and MLX named-output checks for positive masked `data2vec_loss` plus zero hybrid-causal skip loss | Coverage now checks the real concrete training graph outputs. Remaining risk is performance/order integration beyond the tiny fixture. | Current v1 coverage is adequate; add larger integration cases only when changing teacher scheduling or lookahead semantics. |
| Low | `gated_deltanet` FLA parity | Strong optional FLA forward and gradient parity tests exist, plus default-running static full-block scalar reference fixture for projection, short conv, scan, gating, and residual wiring | Default runs no longer rely solely on `FLA_REFERENCE`/`FLA_GRAD_REFERENCE`; upstream FLA fixtures are still optional for cross-library drift detection. | Keep optional FLA parity for upstream comparisons; update the static scalar fixture when public block semantics intentionally change. |
| Low | `mamba3-canonical` | Analytical CPU forward/backward oracle for selective scan, fused-vs-expanded canonical block gradient parity, smoke loss test, and static full-block scalar fixture covering short convolution, dt/lambda/theta projections, B/C normalization and bias, complex-pair scan, gate, output projection, and residual | No fixture is generated directly from a separately maintained upstream implementation, but complete v1 semantics are pinned by independent scalar math and internal gradient parity. | Current v1 coverage is adequate; add upstream fixtures if a canonical external implementation becomes part of import/export compatibility. |
| Low | Distillation teacher ensemble | KL loss has GPU CPU-oracle coverage; teacher runtime verifies normalized probabilities and smoke training; fake-teacher pure-Go oracle pins `mean_logits` and `mean_logprobs` ensemble probabilities | Remaining risk is full teacher model execution, which is already covered by runtime smoke rather than exhaustive fixtures. | Current coverage is adequate. |
| Low | Muon variants (`muon_eq_r`, `normuon`) | Invariant tests verify row normalization and divergence from baseline Muon; one-step GPU-vs-CPU oracle tests pin momentum, Nesterov, Newton-Schulz, aspect scaling, RowL2, and NorMuon normalization | Remaining risk is mostly around future Newton-Schulz variants or weight-decay modifiers. | Current coverage is adequate; add new fixtures when adding optimizer modifiers or alternate variants. |
| Low | DeBERTa/GPT-BERT relative attention | GPU CPU oracle, GPT-BERT bucket-index invariant, P2C sign check, export CPU oracle, native-vs-HF parity case | Coverage now targets the bug class that escaped: bucket sizing, log bucketing, and P2C bucket direction. Remaining risk is mainly around future feature composition. | Keep the new bucket/sign tests load-bearing and include relative attention in any future native-vs-HF parity expansion. |
| Low | Plain attention core (`RoPE`, `qk_norm`, `qk_gain`, masks, windowing, GQA) | IR validation, GPU CPU oracles for RoPE details, export CPU oracle, native-vs-HF parity cases | Export parity proves Mixlab and generated HF agree; it is not always proof of paper/reference semantics. | Accept current coverage for stable features; add focused CPU oracles when adding new position or mask policies. |
| Low | MoE top-k FFN | GPU CPU oracle for top-k routing, weighted expert combination, aux loss, entropy, finite gradients, HF parity case | No external MoE implementation fixture, but v1 semantics are simple and explicitly local. | Current coverage is adequate unless capacity limits, expert parallelism, router noise, or alternate routing policies are added. |
| Low | Masked objectives and hybrid mask switching | Deterministic batch-prep tests, leak-prevention invariants, IR mask switching tests, MLX smoke tests | No external reference is needed for these Mixlab-defined semantics. | Current coverage is adequate; keep adding leakage fixtures when objective semantics change. |
| Low | LAMB | Multi-step GPU CPU oracle covers trust ratio, decoupled decay, gradient clipping, and zero-norm fallback | No known reference gap for v1 formula. | Current coverage is adequate. |

## Immediate Priority

No medium-risk reference-parity gaps remain from this audit. Future work should
be tied to new feature changes or external import/export targets rather than
this audit queue.
