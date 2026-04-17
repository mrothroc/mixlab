package arch

import "testing"

func parallelResidualBlocks() []BlockSpec {
	return []BlockSpec{
		{Type: "plain", Heads: 4},
		{Type: "swiglu"},
		{Type: "plain", Heads: 4},
		{Type: "swiglu"},
	}
}

func TestCountWeightsWithParallelResidual_SavesSwigluNormPerPair(t *testing.T) {
	blocks := parallelResidualBlocks()
	seq, err := CountWeights(DefaultFFNMultiplier, false, false, false, false, blocks)
	if err != nil {
		t.Fatalf("CountWeights sequential: %v", err)
	}
	par, err := CountWeightsWithBigramRecurrenceAndParallel(64, DefaultFFNMultiplier, false, false, false, false, true, 0, 0, blocks, nil)
	if err != nil {
		t.Fatalf("CountWeights parallel: %v", err)
	}
	if seq != 25 {
		t.Fatalf("sequential count=%d want 25", seq)
	}
	if par != 23 {
		t.Fatalf("parallel count=%d want 23", par)
	}
}

func TestBuildIRProgramWithParallelResidual_SharesNormAndReducesOps(t *testing.T) {
	blocks := parallelResidualBlocks()
	seq, err := BuildIRProgram(64, 256, 32, 1, DefaultFFNMultiplier, false, false, false, false, 0, blocks)
	if err != nil {
		t.Fatalf("BuildIRProgram sequential: %v", err)
	}
	par, err := BuildIRProgramWithBigramRecurrenceAndParallel(64, 256, 32, 1, DefaultFFNMultiplier, false, false, false, false, true, 0, 0, 0, blocks, nil)
	if err != nil {
		t.Fatalf("BuildIRProgram parallel: %v", err)
	}
	if par.NumWeights != 23 {
		t.Fatalf("NumWeights=%d want 23", par.NumWeights)
	}
	if got, want := countOps(par, OpRMSNorm), countOps(seq, OpRMSNorm)-2; got != want {
		t.Fatalf("parallel RMSNorm ops=%d want %d", got, want)
	}
	if len(par.Ops) >= len(seq.Ops) {
		t.Fatalf("parallel ops=%d should be fewer than sequential ops=%d", len(par.Ops), len(seq.Ops))
	}

	sharedNorm := weightInputForOutput(t, par, OpRMSNorm, "x_parallel_0_x_norm")
	if sharedNorm != "w3" {
		t.Fatalf("shared norm weight=%q want w3", sharedNorm)
	}
	swigluGate := weightInputForOutput(t, par, OpMatMul, "x_parallel_0_x_norm_parallel_swiglu_0_gate")
	if swigluGate != "w10" {
		t.Fatalf("swiglu gate weight=%q want w10", swigluGate)
	}
}

func TestBuildIRProgramWithParallelResidual_BackwardCompatFalse(t *testing.T) {
	blocks := parallelResidualBlocks()
	base, err := BuildIRProgram(64, 256, 32, 1, DefaultFFNMultiplier, false, false, false, false, 0, blocks)
	if err != nil {
		t.Fatalf("BuildIRProgram: %v", err)
	}
	withFalse, err := BuildIRProgramWithBigramRecurrenceAndParallel(64, 256, 32, 1, DefaultFFNMultiplier, false, false, false, false, false, 0, 0, 0, blocks, nil)
	if err != nil {
		t.Fatalf("BuildIRProgramWithBigramRecurrenceAndParallel false: %v", err)
	}
	if withFalse.NumWeights != base.NumWeights || len(withFalse.Ops) != len(base.Ops) {
		t.Fatalf("parallel=false changed program: weights %d/%d ops %d/%d", withFalse.NumWeights, base.NumWeights, len(withFalse.Ops), len(base.Ops))
	}
}

func TestBuildIRProgramWithParallelResidual_ComposesWithScalesResidMixAndRecurrence(t *testing.T) {
	blocks := []BlockSpec{
		{Type: "plain", Heads: 4},
		{Type: "swiglu"},
		{Type: "plain", Heads: 4},
		{Type: "swiglu"},
		{Type: "plain", Heads: 4},
		{Type: "swiglu"},
	}
	recurrence := []int{0, 1, 2, 3, 2, 3}
	prog, err := BuildIRProgramWithBigramRecurrenceAndParallel(64, 256, 32, 1, DefaultFFNMultiplier, false, true, true, false, true, 0, 0, 0, blocks, recurrence)
	if err != nil {
		t.Fatalf("BuildIRProgram parallel recurrence: %v", err)
	}
	if prog.NumWeights != 31 {
		t.Fatalf("NumWeights=%d want 31", prog.NumWeights)
	}
	origGate := weightInputForOutput(t, prog, OpMatMul, "x_parallel_2_x_norm_parallel_swiglu_2_gate")
	copyGate := weightInputForOutput(t, prog, OpMatMul, "x_parallel_4_x_norm_parallel_swiglu_4_gate")
	if origGate != "w27" || copyGate != origGate {
		t.Fatalf("swiglu recurrence gate original=%q copy=%q want both w27", origGate, copyGate)
	}
	if got := weightInputForOutput(t, prog, OpRMSNorm, "x_parallel_4_x_norm"); got != "w18" {
		t.Fatalf("plain recurrence shared norm=%q want w18", got)
	}
}

func TestBuildIRProgramWithParallelResidual_RejectsInvalidPairs(t *testing.T) {
	blocks := []BlockSpec{{Type: "plain", Heads: 4}, {Type: "plain", Heads: 4}}
	_, err := BuildIRProgramWithBigramRecurrenceAndParallel(64, 256, 32, 1, DefaultFFNMultiplier, false, false, false, false, true, 0, 0, 0, blocks, nil)
	if err == nil {
		t.Fatal("expected invalid pair error")
	}
}

func TestCollectWeightShapesWithParallelResidual_OmitsSwigluNorm(t *testing.T) {
	blocks := parallelResidualBlocks()
	metas, err := CollectWeightShapesWithBigramRecurrenceAndParallel(64, 256, 32, DefaultFFNMultiplier, false, false, false, false, true, 0, 0, blocks, nil)
	if err != nil {
		t.Fatalf("CollectWeightShapesWithBigramRecurrenceAndParallel: %v", err)
	}
	if len(metas) != 23 {
		t.Fatalf("len(metas)=%d want 23", len(metas))
	}
	if metas[10].Name != "w_gate" {
		t.Fatalf("first parallel swiglu weight=%q want w_gate", metas[10].Name)
	}
	for _, idx := range []int{10, 21} {
		if metas[idx].Name == "ffn_norm_scale" {
			t.Fatalf("parallel swiglu retained omitted norm at index %d", idx)
		}
	}
}

func TestCollectWeightShapesWithParallelResidual_ComposesWithRecurrence(t *testing.T) {
	blocks := []BlockSpec{
		{Type: "plain", Heads: 4},
		{Type: "swiglu"},
		{Type: "plain", Heads: 4},
		{Type: "swiglu"},
		{Type: "plain", Heads: 4},
		{Type: "swiglu"},
	}
	recurrence := []int{0, 1, 2, 3, 2, 3}
	metas, err := CollectWeightShapesWithBigramRecurrenceAndParallel(64, 256, 32, DefaultFFNMultiplier, false, true, true, false, true, 0, 0, blocks, recurrence)
	if err != nil {
		t.Fatalf("CollectWeightShapesWithBigramRecurrenceAndParallel: %v", err)
	}
	if len(metas) != 31 {
		t.Fatalf("len(metas)=%d want 31", len(metas))
	}
	if metas[27].Name != "w_gate" {
		t.Fatalf("second original swiglu starts with %q want w_gate", metas[27].Name)
	}
}

func TestParallelResidualInternalHelpers_ErrorPaths(t *testing.T) {
	if _, err := countBlockRangeWeightsWithRecurrenceAndParallel(parallelResidualBlocks(), []int{0, 1, 2, 3}, 1, 3, false, false, true); err == nil {
		t.Fatal("expected unaligned range error")
	}
	bad := []BlockSpec{{Type: "plain", Heads: 4}, {Type: "swiglu"}, {Type: "plain", Heads: 4}}
	if _, err := countStreamWeightsWithRecurrenceAndParallel(bad, nil, false, false, true); err == nil {
		t.Fatal("expected invalid stream error")
	}
	if _, err := parallelBlockWeightCount(BlockSpec{Type: "bogus"}, 0, false, false); err == nil {
		t.Fatal("expected invalid block type error")
	}
}
