package arch

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func TestGridTrainingPhases(t *testing.T) {
	for _, tc := range []struct {
		name    string
		phases  []TrainingPhase
		horizon int
		want    string
	}{
		{"omitted", nil, 0, ""},
		{"constant", []TrainingPhase{{Steps: 6, LR: 1e-4}, {Steps: 4, LR: 1e-5}}, 0, ""},
		{"zero-steps", []TrainingPhase{{Steps: 0, LR: 1e-4}}, 0, "phases[0].steps"},
		{"negative-steps", []TrainingPhase{{Steps: -1, LR: 1e-4}}, 0, "phases[0].steps"},
		{"zero-lr", []TrainingPhase{{Steps: 1, LR: 0}}, 0, "phases[0].lr"},
		{"negative-lr", []TrainingPhase{{Steps: 1, LR: -1}}, 0, "phases[0].lr"},
		{"horizon", []TrainingPhase{{Steps: 1, LR: 1e-4}}, 2, "lr_schedule_steps"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := gridTestConfig(t)
			cfg.Training.Phases = tc.phases
			cfg.Training.LRScheduleSteps = tc.horizon
			before, err := BuildGridIRProgram(cfg, true)
			if err != nil {
				t.Fatal(err)
			}
			raw, err := json.Marshal(cfg)
			if err != nil {
				t.Fatal(err)
			}
			got, err := ParseArchConfig(raw, tc.name)
			if tc.want != "" {
				if err == nil || !strings.Contains(err.Error(), tc.want) {
					t.Fatalf("error=%v want %s", err, tc.want)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got.Training.Phases, tc.phases) {
				t.Fatal("phases lost during parse")
			}
			if len(tc.phases) > 0 && (got.Training.Steps != 10 || got.Training.WarmdownSteps != 0) {
				t.Fatal(got.Training)
			}
			after, err := BuildGridIRProgram(got, true)
			if err != nil || !reflect.DeepEqual(before, after) {
				t.Fatal("phases changed graph", err)
			}
		})
	}
}
