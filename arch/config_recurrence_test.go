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
