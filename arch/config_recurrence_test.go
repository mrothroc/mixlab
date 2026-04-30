package arch

import (
	"encoding/json"
	"strings"
	"testing"
)

func recurrenceTestConfig(recurrence []int) ArchConfig {
	return ArchConfig{
		Name:      "recurrence",
		ModelDim:  64,
		VocabSize: 256,
		SeqLen:    32,
		Blocks: []BlockSpec{
			{Type: "plain", Heads: 4},
			{Type: "swiglu"},
			{Type: "plain", Heads: 4},
			{Type: "swiglu"},
		},
		Recurrence: recurrence,
	}
}

func parseRecurrenceConfig(t *testing.T, cfg ArchConfig) error {
	t.Helper()
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	_, err = ParseArchConfig(data, "recurrence")
	return err
}

func TestParseArchConfig_ValidRecurrence(t *testing.T) {
	cfg := recurrenceTestConfig([]int{0, 1, 0, 1})
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got, err := ParseArchConfig(data, "recurrence")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	if len(got.Recurrence) != 4 || got.Recurrence[2] != 0 || got.Recurrence[3] != 1 {
		t.Fatalf("unexpected recurrence: %v", got.Recurrence)
	}
}

func TestParseArchConfig_RecurrenceActivationSchedule(t *testing.T) {
	t.Run("frac", func(t *testing.T) {
		cfg := recurrenceTestConfig([]int{0, 1, 0, 1})
		cfg.Training = TrainingSpec{Steps: 4550, RecurrenceActivationFrac: 0.35}
		data, err := json.Marshal(cfg)
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		got, err := ParseArchConfig(data, "recurrence_activation_frac")
		if err != nil {
			t.Fatalf("ParseArchConfig: %v", err)
		}
		if got.Training.RecurrenceActivationFrac != 0.35 {
			t.Fatalf("recurrence_activation_frac=%g want 0.35", got.Training.RecurrenceActivationFrac)
		}
		if step := got.Training.EffectiveRecurrenceActivationStep(); step != 1592 {
			t.Fatalf("activation step=%d want 1592", step)
		}
		out, err := json.Marshal(got)
		if err != nil {
			t.Fatalf("round-trip marshal: %v", err)
		}
		if !strings.Contains(string(out), "recurrence_activation_frac") {
			t.Fatalf("round-trip JSON missing recurrence_activation_frac: %s", out)
		}
	})

	t.Run("step", func(t *testing.T) {
		cfg := recurrenceTestConfig([]int{0, 1, 0, 1})
		cfg.Training = TrainingSpec{Steps: 4550, RecurrenceActivationStep: 1592}
		data, err := json.Marshal(cfg)
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		got, err := ParseArchConfig(data, "recurrence_activation_step")
		if err != nil {
			t.Fatalf("ParseArchConfig: %v", err)
		}
		if got.Training.RecurrenceActivationStep != 1592 {
			t.Fatalf("recurrence_activation_step=%d want 1592", got.Training.RecurrenceActivationStep)
		}
		if step := got.Training.EffectiveRecurrenceActivationStep(); step != 1592 {
			t.Fatalf("activation step=%d want 1592", step)
		}
		out, err := json.Marshal(got)
		if err != nil {
			t.Fatalf("round-trip marshal: %v", err)
		}
		if !strings.Contains(string(out), "recurrence_activation_step") {
			t.Fatalf("round-trip JSON missing recurrence_activation_step: %s", out)
		}
	})
}

func TestParseArchConfig_RejectsConflictingRecurrenceActivationSchedule(t *testing.T) {
	cfg := recurrenceTestConfig([]int{0, 1, 0, 1})
	cfg.Training = TrainingSpec{
		Steps:                    4550,
		RecurrenceActivationFrac: 0.35,
		RecurrenceActivationStep: 1592,
	}
	err := parseRecurrenceConfig(t, cfg)
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "cannot set both training.recurrence_activation_frac and training.recurrence_activation_step") {
		t.Fatalf("error %q does not mention conflicting recurrence activation fields", err)
	}
}

func TestParseArchConfig_RejectsBadRecurrence(t *testing.T) {
	tests := []struct {
		name       string
		cfg        ArchConfig
		wantErrSub string
	}{
		{
			name:       "length mismatch",
			cfg:        recurrenceTestConfig([]int{0, 1}),
			wantErrSub: "recurrence length=2 must match blocks length=4",
		},
		{
			name:       "forward reference",
			cfg:        recurrenceTestConfig([]int{0, 2, 2, 3}),
			wantErrSub: "forward reference",
		},
		{
			name:       "out of range",
			cfg:        recurrenceTestConfig([]int{0, 1, 4, 3}),
			wantErrSub: "out of range",
		},
		{
			name:       "type mismatch",
			cfg:        recurrenceTestConfig([]int{0, 0, 2, 3}),
			wantErrSub: "type mismatch",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := parseRecurrenceConfig(t, tc.cfg)
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tc.wantErrSub) {
				t.Fatalf("error %q does not contain %q", err, tc.wantErrSub)
			}
		})
	}
}
