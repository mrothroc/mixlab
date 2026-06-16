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
| Medium | `training.data2vec` runtime | Config/IR tests, SmoothL1 GPU CPU oracle, deterministic tests for tau scheduling, EMA mixing, target averaging, target normalization, and hybrid/causal skip, plus hybrid training smoke | Full GPU teacher integration is still primarily covered by smoke testing, so failures in the bridge from teacher hidden outputs to target buffers may require integration tests to catch. | Add a small MLX integration assertion that reads `data2vec_loss` and verifies masked secondary steps produce non-zero loss while hybrid causal steps produce exactly zero. |
| Medium | `gated_deltanet` FLA parity | Strong FLA forward and gradient parity tests exist, but they are gated by `FLA_REFERENCE` and `FLA_GRAD_REFERENCE` env vars | Default local and CI runs skip these tests unless external `.npz` fixtures are supplied. | Commit small generated reference fixtures or add a documented fixture-generation target and CI job that runs the parity tests on a pinned FLA version. |
| Medium | `mamba3-canonical` | Analytical CPU forward/backward oracle for selective scan, fused-vs-expanded canonical block gradient parity, smoke loss test | Good internal math coverage, but no fixture from a canonical external/reference implementation for the complete block. | Add a tiny reference fixture for the complete canonical block, including short convolution, dt/lambda/theta projections, complex pair rotation, scan, gate, and output projection. |
| Medium | Distillation teacher ensemble | KL loss has GPU CPU-oracle coverage; teacher runtime verifies normalized probabilities and smoke training | `mean_logits` and `mean_logprobs` ensemble semantics are not tested against deterministic multi-teacher logits without running real teacher models. | Add a small pure-Go ensemble oracle test with two fake teacher logits and expected probabilities for both strategies. Keep existing runtime smoke as integration coverage. |
| Medium | Muon variants (`muon_eq_r`, `normuon`) | Invariant tests verify row normalization and divergence from baseline Muon | The tests do not pin a one-step numerical fixture for the advertised optimizer formulas. | Add one-step CPU oracle tests for each public Muon variant with fixed gradients, momentum, Newton-Schulz settings, and decay flags. |
| Low | DeBERTa/GPT-BERT relative attention | GPU CPU oracle, GPT-BERT bucket-index invariant, P2C sign check, export CPU oracle, native-vs-HF parity case | Coverage now targets the bug class that escaped: bucket sizing, log bucketing, and P2C bucket direction. Remaining risk is mainly around future feature composition. | Keep the new bucket/sign tests load-bearing and include relative attention in any future native-vs-HF parity expansion. |
| Low | Plain attention core (`RoPE`, `qk_norm`, `qk_gain`, masks, windowing, GQA) | IR validation, GPU CPU oracles for RoPE details, export CPU oracle, native-vs-HF parity cases | Export parity proves Mixlab and generated HF agree; it is not always proof of paper/reference semantics. | Accept current coverage for stable features; add focused CPU oracles when adding new position or mask policies. |
| Low | MoE top-k FFN | GPU CPU oracle for top-k routing, weighted expert combination, aux loss, entropy, finite gradients, HF parity case | No external MoE implementation fixture, but v1 semantics are simple and explicitly local. | Current coverage is adequate unless capacity limits, expert parallelism, router noise, or alternate routing policies are added. |
| Low | Masked objectives and hybrid mask switching | Deterministic batch-prep tests, leak-prevention invariants, IR mask switching tests, MLX smoke tests | No external reference is needed for these Mixlab-defined semantics. | Current coverage is adequate; keep adding leakage fixtures when objective semantics change. |
| Low | LAMB | Multi-step GPU CPU oracle covers trust ratio, decoupled decay, gradient clipping, and zero-norm fallback | No known reference gap for v1 formula. | Current coverage is adequate. |

## Immediate Priority

The next audit-driven work should move to the remaining data2vec integration
gap:

1. Add a small MLX integration assertion that reads `data2vec_loss` and verifies
   masked secondary steps produce non-zero loss while hybrid causal steps
   produce exactly zero.

The recurrent block scan and full-block wiring paths now have static fixtures.
