package train

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/gpu"
)

func TestPhaseEffectiveGroupRates(t *testing.T) {
	for _, tc := range []struct {
		name, fields         string
		phase, matrix, state float64
	}{
		{"inherited", "", 1e-4, 1e-4, 1e-3 / 3},
		// Overrides need an explicit base with phases; arch rejects an implicit one.
		{"matrix_override", `,"lr":0.0003,"matrix_lr":0.02`, 1e-4, 0.02 / 3, 1e-3 / 3},
		{"explicit_base", `,"lr":0.0001,"matrix_lr":0.02`, 1e-4, 0.02, 1e-3},
		{"state_override", "", 0.01, 0.01, 1e-3 * 100 / 3},
		{"zero_base", `,"lr":0,"matrix_lr":0.02`, 1e-4, 0.02, 1e-3},
		{"zero_override", `,"matrix_lr":0`, 1e-4, 0, 1e-3 / 3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg, err := ParseArchConfig([]byte(fmt.Sprintf(`{"model_dim":16,"vocab_size":32,"seq_len":4,
			"blocks":[{"type":"plain","heads":2}],"training":{"batch_tokens":4,
			"phases":[{"steps":2,"lr":%g},{"steps":2,"lr":%g}]%s}}`, tc.phase, tc.phase/10, tc.fields)), "phase rates")
			if err != nil {
				t.Fatal(err)
			}
			shapes := []WeightShape{
				{Name: "embed", Shape: []int{32, 16}},
				{Name: "wq", Shape: []int{16, 16}},
				{Name: "norm_scale", Shape: []int{16}, IsNormScale: true},
				{Name: "head", Shape: []int{16, 32}},
			}
			for _, role := range []string{"s4d_state", "ssm_state", "s4d_sobolev"} {
				shapes = append(shapes, WeightShape{Name: role, Shape: []int{16}, OptimizerRole: role, OptimizerLR: 0.001})
			}
			spec, err := buildTrainerOptimizerSpec(cfg, shapes)
			if err != nil {
				t.Fatal(err)
			}
			sched, _ := buildTrainingScheduler(cfg.Training)
			for _, step := range []int{0, 2} {
				scale := scheduledLRScale(spec.DefaultBaseLR, sched.At(step))
				for i, w := range spec.Weights {
					want := tc.phase
					switch {
					case i == 1:
						want = tc.matrix
					case i >= 4:
						want = tc.state
					case spec.DefaultBaseLR == 0:
						want = 0
					}
					if step == 2 && spec.DefaultBaseLR > 0 {
						want /= 10
					}
					g := spec.Groups[w.GroupIndex]
					got := g.LR * scale
					if math.Abs(float64(got)-want) > 3e-7*math.Max(want, 1e-4) {
						t.Fatalf("step %d %s: got %g want %g", step, shapes[i].Name, got, want)
					}
					if !strings.Contains(effectiveGroupRates(spec, sched.At(step)), fmt.Sprintf("%s=%.6g", g.ReportName, got)) {
						t.Fatal("reported rate differs from applied rate")
					}
				}
			}
		})
	}
}

type phaseReportingTrainer struct {
	GPUTrainer
	spec gpu.TrainerOptimizerSpec
}

func (t phaseReportingTrainer) effectiveLearningRates(lr float32) string {
	return effectiveGroupRates(t.spec, lr)
}

func TestPhaseRateLogging(t *testing.T) {
	cfg := gridPhaseConfig(t)
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatal(err)
	}
	spec, err := buildTrainerOptimizerSpec(cfg, shapes)
	if err != nil {
		t.Fatal(err)
	}
	sched, _ := buildTrainingScheduler(cfg.Training)
	trainer := phaseReportingTrainer{spec: spec}
	for _, start := range []int{0, 4, 6, 7} {
		out := captureStdout(t, func() {
			for step := start; step < 10; step++ {
				logPhaseRates(trainer, sched, step, start, "test")
			}
		})
		want := 1
		if start < 6 {
			want = 2
		}
		if strings.Count(out, "effective optimizer rates") != want || !strings.Contains(out, fmt.Sprintf("update=%d ", start+1)) {
			t.Fatalf("start=%d: %s", start, out)
		}
		if !strings.Contains(out, "extra:grid_kernel=") {
			t.Fatal(out)
		}
	}
	if out := captureStdout(t, func() { logPhaseRates(trainer, LRSchedule{}, 0, 0, "test") }); out != "" {
		t.Fatal(out)
	}
}
