//go:build mlx && cgo && (darwin || linux)

package train

import (
	"fmt"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestGridPhasesExactResumeAcrossLRDrop(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX required")
	}
	manifest := prepareTinyGrid(t)
	for _, optimizer := range []string{"adamw", "lamb"} {
		t.Run(optimizer, func(t *testing.T) {
			cfg := gridPhaseConfig(t)
			cfg.Training.Optimizer = optimizer
			fullDir := filepath.Join(t.TempDir(), "full")
			var full TrainResult
			log := captureStdout(t, func() {
				var err error
				full, err = runGridTrain(cfg, manifest, TrainOptions{CheckpointDir: fullDir, CheckpointEvery: 1, LogEvery: 1, ValEvery: 2})
				if err != nil {
					t.Fatal(err)
				}
			})
			for step := 1; step <= 10; step++ {
				want := float32(1e-4)
				if step > 6 {
					want = 1e-5
				}
				found := false
				for _, line := range strings.Split(log, "\n") {
					if strings.Contains(line, fmt.Sprintf("step=%d/10 ", step)) && strings.Contains(line, fmt.Sprintf(" lr=%g ", want)) {
						found = true
					}
				}
				if !found {
					t.Fatalf("missing LR for step %d in %s", step, log)
				}
			}
			final, err := resolveResumeManifest(filepath.Join(fullDir, resumeManifestFilename(10)))
			if err != nil {
				t.Fatal(err)
			}
			want, err := loadResumeState(final)
			if err != nil {
				t.Fatal(err)
			}
			shapes, err := computeWeightShapes(cfg)
			if err != nil {
				t.Fatal(err)
			}
			wantWeights, err := loadSafetensorsWeights(want.ModelPath, shapes)
			if err != nil {
				t.Fatal(err)
			}
			if want.Trainer.Optimizer.CommittedSteps != 10 || len(want.Trainer.Tensors) == 0 {
				t.Fatal("optimizer state missing or reset")
			}
			for _, step := range []int{5, 6, 7} {
				t.Run(fmt.Sprintf("after_%d", step), func(t *testing.T) {
					dir := t.TempDir()
					result, err := runGridTrain(cfg, manifest, TrainOptions{Resume: filepath.Join(fullDir, resumeManifestFilename(step)), CheckpointDir: dir, CheckpointEvery: 1, ValEvery: 2})
					if err != nil {
						t.Fatal(err)
					}
					m, err := resolveResumeManifest(filepath.Join(dir, resumeManifestFilename(10)))
					if err != nil {
						t.Fatal(err)
					}
					got, err := loadResumeState(m)
					if err != nil {
						t.Fatal(err)
					}
					weights, err := loadSafetensorsWeights(got.ModelPath, shapes)
					if err != nil {
						t.Fatal(err)
					}
					if !reflect.DeepEqual(weights, wantWeights) || !reflect.DeepEqual(got.Trainer, want.Trainer) || !reflect.DeepEqual(m.Schedule, final.Schedule) || !reflect.DeepEqual(m.Grid, final.Grid) || result.LastLoss != full.LastLoss || result.LastValLoss != full.LastValLoss {
						t.Fatal("resume changed weights, moments, schedule, replay, or metrics")
					}
				})
			}
		})
	}
}
