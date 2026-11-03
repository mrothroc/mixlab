//go:build mlx && cgo && (darwin || linux)

package train

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/data"
)

func TestGridRecoveryFinalValidation(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX required")
	}
	t.Setenv(gridMemoryPreflightEnv, "0")
	manifest := prepareTinyGrid(t)
	for _, tc := range []struct {
		cadence     int
		earlierBest bool
	}{{2, false}, {4, false}, {0, false}, {2, true}} {
		t.Run(fmt.Sprintf("cadence-%d-earlier-best-%t", tc.cadence, tc.earlierBest), func(t *testing.T) {
			cfg, err := LoadArchConfig("../examples/grid_two_stage_1.json")
			if err != nil {
				t.Fatal(err)
			}
			cfg.Training.Steps = 6
			cfg.Training.EarlyStop = &EarlyStopSpec{Patience: 100}
			controlDir, retryDir := t.TempDir(), t.TempDir()
			opts := func(dir string) TrainOptions {
				return TrainOptions{CheckpointDir: dir, CheckpointEvery: tc.cadence, ValEvery: 2, SafetensorsPath: filepath.Join(dir, "final.safetensors")}
			}
			var controlMetrics, resumedMetrics gridMetrics
			evaluator := func(dst *gridMetrics, failFinal bool) func(*ArchConfig, GPUTrainer, *data.GridDataset) (gridMetrics, error) {
				return func(c *ArchConfig, tr GPUTrainer, ds *data.GridDataset) (gridMetrics, error) {
					m, e := evaluateGridDataset(c, tr, ds)
					if e != nil {
						return m, e
					}
					stats, e := readOptimizerStats(tr)
					if e != nil {
						return m, e
					}
					if tc.earlierBest {
						// Control selection independently of convergence while still
						// exercising the real validation forward on every call.
						m.SSE = float64(m.Count) * float64(1+stats.AttemptedSteps)
					}
					if stats.AttemptedSteps == 6 {
						*dst = m
						if failFinal {
							return m, errors.New("injected final validation out of memory")
						}
					}
					return m, nil
				}
			}
			control, err := runGridTrainWithHooks(cfg, manifest, opts(controlDir), gridTrainHooks{evaluate: evaluator(&controlMetrics, false)})
			if err != nil {
				t.Fatal(err)
			}
			_, err = runGridTrainWithHooks(cfg, manifest, opts(retryDir), gridTrainHooks{evaluate: evaluator(&resumedMetrics, true)})
			if err == nil || !strings.Contains(err.Error(), "injected final validation out of memory") {
				t.Fatal("expected final validation failure", err)
			}
			noStep := func(GPUTrainer, objectiveBatch, int, float32) (float32, error) {
				t.Fatal("final-step recovery attempted a training update")
				return 0, nil
			}
			// A second failed validation must leave an equally usable bundle.
			resumeOpts := opts(retryDir)
			// Even with preflight enabled, finalization must not probe updates.
			t.Setenv(gridMemoryPreflightEnv, "1")
			for attempt := 0; attempt < 2; attempt++ {
				path := gridFinalFailureManifest(t, retryDir)
				m, e := readResumeManifest(path)
				if e != nil || m.GlobalStep != 6 || m.Grid == nil || !m.Grid.ValidationPending {
					t.Fatal("invalid final recovery bundle", m, e)
				}
				for _, file := range []string{"step_000006.st", "final.safetensors"} {
					if _, e := os.Stat(filepath.Join(retryDir, file)); !os.IsNotExist(e) {
						t.Fatalf("premature %s: %v", file, e)
					}
				}
				resumeOpts.Resume = path
				if attempt == 0 {
					_, err = runGridTrainWithHooks(cfg, manifest, resumeOpts, gridTrainHooks{step: noStep, evaluate: evaluator(&resumedMetrics, true)})
					if err == nil || !strings.Contains(err.Error(), "injected final validation out of memory") {
						t.Fatal("expected repeat validation failure", err)
					}
				}
			}
			var got TrainResult
			validationCalls := 0
			finalEvaluator := evaluator(&resumedMetrics, false)
			log := captureStdout(t, func() {
				got, err = runGridTrainWithHooks(cfg, manifest, resumeOpts, gridTrainHooks{step: noStep, evaluate: func(c *ArchConfig, tr GPUTrainer, ds *data.GridDataset) (gridMetrics, error) {
					validationCalls++
					return finalEvaluator(c, tr, ds)
				}})
			})
			if err != nil {
				t.Fatal(err)
			}
			if validationCalls != 1 || strings.Count(log, "finished from final-step bundle") != 1 || !strings.Contains(log, "validation step=6 "+controlMetrics.String(cfg.DenseRegression.EffectiveMetricScale())) || strings.Contains(log, "grid loader:") || strings.Contains(log, "preflight-start") {
				t.Fatal("missing completion/validation record or unnecessary training loader", log)
			}
			if got.FirstLoss != control.FirstLoss || got.LastLoss != control.LastLoss || got.LastValLoss != control.LastValLoss || got.HasValLoss != control.HasValLoss || got.Delta != control.Delta || resumedMetrics != controlMetrics {
				t.Fatal("final recovery changed metrics", got, control, resumedMetrics, controlMetrics)
			}
			for _, file := range []string{"best.safetensors", "final.safetensors"} {
				assertGridArtifactEqual(t, controlDir, retryDir, file)
			}
			if tc.cadence > 0 && 6%tc.cadence == 0 {
				assertGridArtifactEqual(t, controlDir, retryDir, "step_000006.st")
				assertGridArtifactEqual(t, controlDir, retryDir, resumeStateFilename(6))
				a, e := resolveResumeManifest(filepath.Join(controlDir, resumeManifestFilename(6)))
				if e != nil {
					t.Fatal(e)
				}
				b, e := resolveResumeManifest(filepath.Join(retryDir, resumeManifestFilename(6)))
				if e != nil {
					t.Fatal(e)
				}
				if b.Grid.ValidationPending || !reflect.DeepEqual(a.Grid, b.Grid) || !reflect.DeepEqual(a.EarlyStop, b.EarlyStop) || !reflect.DeepEqual(a.Schedule, b.Schedule) || a.Optimizer != b.Optimizer {
					t.Fatal("final checkpoint replay/selection/schedule/optimizer state mismatch")
				}
				if _, e = prepareResumeRun(cfg, manifest, filepath.Join(retryDir, resumeManifestFilename(6)), newEarlyStopState(cfg.Training.EarlyStop)); e == nil || !strings.Contains(e.Error(), "at or after configured training steps") {
					t.Fatal("finished checkpoint admitted", e)
				}
			} else if _, e := os.Stat(filepath.Join(retryDir, "step_000006.st")); !os.IsNotExist(e) {
				t.Fatal("off-cadence final checkpoint written", e)
			}
		})
	}
}

func gridFinalFailureManifest(t *testing.T, dir string) string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(dir, "recovery", "failure.json"))
	if err != nil {
		t.Fatal(err)
	}
	var report struct {
		Manifest string `json:"resume_manifest"`
	}
	if err = json.Unmarshal(raw, &report); err != nil {
		t.Fatal(err)
	}
	return report.Manifest
}

func assertGridArtifactEqual(t *testing.T, wantDir, gotDir, file string) {
	t.Helper()
	want, err := os.ReadFile(filepath.Join(wantDir, file))
	if err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(filepath.Join(gotDir, file))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("%s differs from uninterrupted run", file)
	}
}
