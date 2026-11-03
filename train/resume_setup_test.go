package train

import (
	"fmt"
	"testing"
)

func TestResumeFinalStepAdmission(t *testing.T) {
	grid, err := LoadArchConfig("../examples/grid_two_stage_1.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name    string
		step    int
		cfg     *ArchConfig
		state   *gridResumeState
		allowed bool
	}{
		{"grid-before-end", 5, grid, &gridResumeState{Version: 1}, true},
		{"grid-pending-final", 6, grid, &gridResumeState{Version: 1, ValidationPending: true}, true},
		{"grid-completed-final", 6, grid, &gridResumeState{Version: 1}, false},
		{"grid-beyond-end", 7, grid, &gridResumeState{Version: 1, ValidationPending: true}, false},
		{"grid-no-state", 6, grid, nil, false},
		{"grid-unsupported-state", 6, grid, &gridResumeState{Version: 2, ValidationPending: true}, false},
		{"language-before-end", 5, smallSWAArtifactConfig(), nil, true},
		{"language-final", 6, smallSWAArtifactConfig(), nil, false},
		{"language-with-grid-state", 6, smallSWAArtifactConfig(), &gridResumeState{Version: 1, ValidationPending: true}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := validateResumeStep(tc.cfg, resumeManifest{GlobalStep: tc.step, Grid: tc.state}, 6)
			if tc.allowed {
				if err != nil {
					t.Fatal(err)
				}
				return
			}
			want := fmt.Sprintf("checkpoint is at step %d, at or after configured training steps 6; raise training.steps to extend", tc.step)
			if err == nil || err.Error() != want {
				t.Fatalf("got %v, want %s", err, want)
			}
		})
	}
}
