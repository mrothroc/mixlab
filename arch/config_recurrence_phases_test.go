package arch

import (
	"encoding/json"
	"strings"
	"testing"
)

func parsePhaseConfig(t *testing.T, cfg ArchConfig) (*ArchConfig, error) {
	t.Helper()
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return ParseArchConfig(data, "recurrence_phases")
}

func validRecurrencePhaseConfig() ArchConfig {
	cfg := recurrenceTestConfig([]int{0, 1, 0, 1})
	cfg.Training = TrainingSpec{Steps: 1000}
	cfg.RecurrencePhases = []RecurrencePhaseSpec{
		{Frac: 0, Order: []int{0, 1}},
		{Frac: 0.5, Order: []int{0, 1, 2, 3}},
	}
	return cfg
}

func TestParseArchConfig_ValidRecurrencePhases(t *testing.T) {
	cfg := validRecurrencePhaseConfig()
	got, err := parsePhaseConfig(t, cfg)
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	if len(got.RecurrencePhases) != 2 {
		t.Fatalf("phases=%d want 2", len(got.RecurrencePhases))
	}
	if steps := got.PhaseStartSteps(); len(steps) != 2 || steps[0] != 0 || steps[1] != 500 {
		t.Fatalf("PhaseStartSteps=%v want [0 500]", steps)
	}
	out, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("round-trip marshal: %v", err)
	}
	if !strings.Contains(string(out), "recurrence_phases") {
		t.Fatalf("round-trip JSON missing recurrence_phases: %s", out)
	}
}

func TestParseArchConfig_RejectsBadRecurrencePhases(t *testing.T) {
	tests := []struct {
		name       string
		mutate     func(*ArchConfig)
		wantErrSub string
	}{
		{
			name: "missing zero frac",
			mutate: func(cfg *ArchConfig) {
				cfg.RecurrencePhases[0].Frac = 0.1
			},
			wantErrSub: "recurrence_phases[0].frac=0.1 must be 0.0",
		},
		{
			name: "misordered frac",
			mutate: func(cfg *ArchConfig) {
				cfg.RecurrencePhases = append(cfg.RecurrencePhases, RecurrencePhaseSpec{Frac: 0.25, Order: []int{0, 1, 2, 3}})
			},
			wantErrSub: "must be greater than previous",
		},
		{
			name: "collapsed integer start",
			mutate: func(cfg *ArchConfig) {
				cfg.Training.Steps = 2
				cfg.RecurrencePhases[1].Frac = 0.1
			},
			wantErrSub: "starts at step 0",
		},
		{
			name: "duplicate order position",
			mutate: func(cfg *ArchConfig) {
				cfg.RecurrencePhases[0].Order = []int{0, 1, 1}
			},
			wantErrSub: "repeats block position 1",
		},
		{
			name: "out of bounds",
			mutate: func(cfg *ArchConfig) {
				cfg.RecurrencePhases[0].Order = []int{0, 4}
			},
			wantErrSub: "out of range",
		},
		{
			name: "root after copy",
			mutate: func(cfg *ArchConfig) {
				cfg.RecurrencePhases[1].Order = []int{2, 0, 1, 3}
			},
			wantErrSub: "reuses weights from root block 0",
		},
		{
			name: "legacy activation conflict",
			mutate: func(cfg *ArchConfig) {
				cfg.Training.RecurrenceActivationStep = 5
			},
			wantErrSub: "cannot set recurrence_phases with training.recurrence_activation_frac or training.recurrence_activation_step",
		},
		{
			name: "schema 2 conflict",
			mutate: func(cfg *ArchConfig) {
				cfg.ExecutionOrder = []int{0, 1, 2, 3}
			},
			wantErrSub: "cannot set recurrence_phases with execution_order or recurrence_phase_activations",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := validRecurrencePhaseConfig()
			tc.mutate(&cfg)
			_, err := parsePhaseConfig(t, cfg)
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tc.wantErrSub) {
				t.Fatalf("error %q does not contain %q", err, tc.wantErrSub)
			}
		})
	}
}

func TestParseArchConfig_RejectsEmptyRecurrencePhasesField(t *testing.T) {
	_, err := ParseArchConfig([]byte(`{
		"name": "empty_phases",
		"model_dim": 64,
		"vocab_size": 256,
		"seq_len": 32,
		"blocks": [{"type": "plain", "heads": 4}],
		"recurrence_phases": []
	}`), "empty_phases")
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "recurrence_phases must contain at least one phase") {
		t.Fatalf("error %q does not mention empty recurrence_phases", err)
	}
}

func TestParseArchConfig_RejectsRecurrencePhaseKVSourceAfterDependent(t *testing.T) {
	cfg := ArchConfig{
		Name:      "kv_phase",
		ModelDim:  64,
		VocabSize: 256,
		SeqLen:    32,
		Blocks: []BlockSpec{
			{Type: "plain", Heads: 4},
			{Type: "plain", Heads: 4, KVSource: 1},
		},
		Training: TrainingSpec{Steps: 100},
		RecurrencePhases: []RecurrencePhaseSpec{
			{Frac: 0, Order: []int{1}},
		},
	}
	_, err := parsePhaseConfig(t, cfg)
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "kv_source=1") {
		t.Fatalf("error %q does not mention kv_source", err)
	}
}

func TestParseArchConfig_RejectsRecurrencePhaseParallelResidualSplit(t *testing.T) {
	cfg := ArchConfig{
		Name:             "parallel_phase",
		ModelDim:         64,
		VocabSize:        256,
		SeqLen:           32,
		ParallelResidual: true,
		Blocks: []BlockSpec{
			{Type: "plain", Heads: 4},
			{Type: "swiglu"},
		},
		Training: TrainingSpec{Steps: 100},
		RecurrencePhases: []RecurrencePhaseSpec{
			{Frac: 0, Order: []int{1, 0}},
		},
	}
	_, err := parsePhaseConfig(t, cfg)
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "parallel_residual pair [0,1]") {
		t.Fatalf("error %q does not mention split pair", err)
	}
}

func TestParseArchConfig_RejectsRecurrencePhasesWithBackoutOmittingSaveLayer(t *testing.T) {
	cfg := validRecurrencePhaseConfig()
	cfg.Backout = &BackoutSpec{SaveLayer: 2, LambdaInit: -1, saveLayerSet: true, lambdaInitSet: true}
	cfg.RecurrencePhases[0].Order = []int{0, 1}
	_, err := parsePhaseConfig(t, cfg)
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "backout.save_layer=2") {
		t.Fatalf("error %q does not mention backout save_layer", err)
	}
}

func TestParseArchConfig_RejectsSchema2RecurrencePhases(t *testing.T) {
	cfg := recurrenceTestConfig([]int{0, 1, 0, 1})
	cfg.Training = TrainingSpec{Steps: 100}
	cfg.ExecutionOrder = []int{0, 1, 2, 3}
	cfg.RecurrencePhaseActivations = []RecurrencePhaseActivationSpec{{Frac: 0, ExecutionPrefixLen: 2}}
	_, err := parsePhaseConfig(t, cfg)
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "schema is not implemented") {
		t.Fatalf("error %q does not mention unsupported schema", err)
	}
}
