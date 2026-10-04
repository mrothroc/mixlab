# Execution Contract Fixtures, Revision 1

These are hand-checkable fixtures, not generated model predictions.
`expected.json` records arithmetic invariants and numerical tolerances.

- `ttt.json`: D=8, H=2, head width=4, default hidden multiplier=4, hidden width=16,
  chunk size=4. Packed sizes are 128+32+128+8=296. Native stateful offsets 0
  and 3 are checked against the actual builder, including mutated bindings.
- `recurrent.json`: two rows of length three, Dscan=4, plus a pointwise GLU.
  `gpu.TestExecutionContractScanZeroResetOracle` independently evaluates the
  one-channel hand recurrence in `expected.json`, twice to verify call reset.
- Spatial fixture: reuse `examples/grid_two_stage_2.json`, rather than copy its
  shape analysis. It binds [2,8,8,2] input and [2,8,8,1] prediction, with an
  actual detached edge and frozen first-stage weights. Changing selection to
  output1 makes stage2 unreachable. The tests check both cases.

Existing TTT and grid numerical references are reused, not regenerated or
loosened for these contracts. See `docs/state-execution-contracts.md` for the
report format.
