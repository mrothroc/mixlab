package arch

import "testing"

func countOpCode(prog *Program, code int) int {
	count := 0
	for _, op := range prog.Ops {
		if op.Code == code {
			count++
		}
	}
	return count
}

func recurrenceActivationProgramConfig() *ArchConfig {
	return &ArchConfig{
		Name:       "recurrence_activation",
		ModelDim:   64,
		VocabSize:  256,
		SeqLen:     32,
		Recurrence: []int{0, 1, 2, 3, 2, 3},
		Blocks: []BlockSpec{
			{Type: "plain", Heads: 4},
			{Type: "swiglu"},
			{Type: "plain", Heads: 4},
			{Type: "swiglu"},
			{Type: "plain", Heads: 4},
			{Type: "swiglu"},
		},
		Training: TrainingSpec{Steps: 1000, BatchTokens: 32, RecurrenceActivationStep: 500},
	}
}

func TestBuildPreActivationIRProgramFromConfig_UsesUniqueRecurrenceBlocks(t *testing.T) {
	cfg := recurrenceActivationProgramConfig()
	full, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}
	pre, err := BuildPreActivationIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildPreActivationIRProgramFromConfig: %v", err)
	}
	if pre.NumWeights != full.NumWeights {
		t.Fatalf("pre NumWeights=%d want full NumWeights=%d", pre.NumWeights, full.NumWeights)
	}
	if len(pre.Ops) >= len(full.Ops) {
		t.Fatalf("pre op count=%d want less than full op count=%d", len(pre.Ops), len(full.Ops))
	}
	if got, want := countOpCode(pre, OpRMSNorm), 5; got != want {
		t.Fatalf("pre RMSNorm ops=%d want %d", got, want)
	}
	if got, want := countOpCode(full, OpRMSNorm), 7; got != want {
		t.Fatalf("full RMSNorm ops=%d want %d", got, want)
	}
	if got := weightInputForOutput(t, pre, OpRMSNorm, "x_attn_2_x_norm"); got != "w14" {
		t.Fatalf("pre block 2 norm weight=%q want w14", got)
	}
	if got := weightInputForOutput(t, full, OpRMSNorm, "x_attn_4_x_norm"); got != "w14" {
		t.Fatalf("full repeated block 4 norm weight=%q want w14", got)
	}
}

func TestBuildPreActivationIRProgramFromConfig_BackwardCompatWithoutRecurrence(t *testing.T) {
	cfg := recurrenceActivationProgramConfig()
	cfg.Recurrence = nil
	full, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}
	pre, err := BuildPreActivationIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildPreActivationIRProgramFromConfig: %v", err)
	}
	if pre.NumWeights != full.NumWeights || len(pre.Ops) != len(full.Ops) {
		t.Fatalf("pre changed non-recurrent program: weights %d/%d ops %d/%d", pre.NumWeights, full.NumWeights, len(pre.Ops), len(full.Ops))
	}
}

func TestBuildTrainingIRProgramForRecurrencePhaseFromConfig(t *testing.T) {
	cfg := recurrenceActivationProgramConfig()
	cfg.RecurrencePhases = []RecurrencePhaseSpec{
		{Frac: 0, Order: []int{0, 1, 2, 3}},
		{Frac: 0.5, Order: []int{0, 1, 2, 3, 4, 5}},
	}
	phase0, err := BuildTrainingIRProgramForRecurrencePhaseFromConfig(cfg, 0, TrainingProgramState{RecurrenceActive: true})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramForRecurrencePhaseFromConfig phase0: %v", err)
	}
	phase1, err := BuildTrainingIRProgramForRecurrencePhaseFromConfig(cfg, 1, TrainingProgramState{RecurrenceActive: true})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramForRecurrencePhaseFromConfig phase1: %v", err)
	}
	if phase0.NumWeights != phase1.NumWeights {
		t.Fatalf("phase weights mismatch: phase0=%d phase1=%d", phase0.NumWeights, phase1.NumWeights)
	}
	if len(phase0.Ops) >= len(phase1.Ops) {
		t.Fatalf("phase0 ops=%d want less than phase1 ops=%d", len(phase0.Ops), len(phase1.Ops))
	}
	full, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}
	if len(full.Ops) != len(phase1.Ops) {
		t.Fatalf("full ops=%d want max-cost phase ops=%d", len(full.Ops), len(phase1.Ops))
	}
}
