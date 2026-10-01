package train

import (
	"encoding/json"
	"reflect"
	"testing"
)

func gridPhaseConfig(t *testing.T) *ArchConfig {
	t.Helper()
	cfg, err := LoadArchConfig("../examples/grid_regression_tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	cfg.Training.Phases = []TrainingPhase{{Steps: 6, LR: 1e-4}, {Steps: 4, LR: 1e-5}}
	raw, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	cfg, err = ParseArchConfig(raw, "grid phases")
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

func TestGridPhaseScheduleAndResume(t *testing.T) {
	cfg := gridPhaseConfig(t)
	sched, steps := buildTrainingScheduler(cfg.Training)
	if steps != 10 {
		t.Fatal(steps)
	}
	for step := 0; step < steps; step++ {
		want := float32(1e-4)
		if step >= 6 {
			want = 1e-5
		}
		if got := sched.At(step); got != want {
			t.Fatalf("update %d lr=%g want=%g", step+1, got, want)
		}
	}
	saved, err := resumeScheduleFrom(cfg.Training, sched, steps)
	if err != nil {
		t.Fatal(err)
	}
	restored, n, err := schedulerForResume(saved, steps)
	if err != nil || n != steps {
		t.Fatal(n, err)
	}
	if !reflect.DeepEqual(sched, restored) {
		t.Fatal("restored schedule differs")
	}
	// An explicitly requested warmdown retains the common scheduler semantics.
	cfg.Training.WarmdownSteps = 2
	warm, _ := buildTrainingScheduler(cfg.Training)
	if warm.At(5) != 1e-4 || warm.At(6) != 1e-5 || warm.At(8) != 1e-5 || warm.At(9) >= 1e-5 {
		t.Fatal("warmdown changed earlier phases or was ignored")
	}
}
