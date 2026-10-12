//go:build mlx && cgo && (darwin || linux)

package train

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestGridRecoveryFailedStepExactResume(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX required")
	}
	t.Setenv(gridMemoryPreflightEnv, "0")
	manifest := prepareTinyGrid(t)
	cfg, err := LoadArchConfig("../examples/grid_two_stage_1.json")
	if err != nil {
		t.Fatal(err)
	}
	cfg.Training.Steps = 6
	controlDir := t.TempDir()
	control, err := runGridTrain(cfg, manifest, TrainOptions{CheckpointDir: controlDir, CheckpointEvery: 2, ValEvery: 2})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		phase  string
		failAt int
	}{{"submit", 1}, {"collect-after-update", 1}, {"submit", 3}, {"collect-after-update", 3}} {
		t.Run(fmt.Sprintf("%s-%d", tc.phase, tc.failAt), func(t *testing.T) {
			phase, failAt := tc.phase, tc.failAt
			dir := t.TempDir()
			calls := 0
			_, err := runGridTrainWithStep(cfg, manifest, TrainOptions{CheckpointDir: dir, ValEvery: 2}, func(tr GPUTrainer, b objectiveBatch, bs int, lr float32) (float32, error) {
				calls++
				if calls == failAt && phase == "submit" {
					return 0, errors.New("injected submit out of memory")
				}
				if e := submitPreparedStepGPU(tr, b, bs, 0, lr); e != nil {
					return 0, e
				}
				loss, e := tr.CollectLossGPU()
				// Simulate a failed collect after GPU state has advanced. Recovery
				// must use step 2, never serialize these step-3 weights/moments.
				if calls == failAt {
					return 0, errors.New("injected collect out of memory")
				}
				return loss, e
			})
			completed := failAt - 1
			if err == nil || !strings.Contains(err.Error(), fmt.Sprintf("completed step %d", completed)) || calls != failAt {
				t.Fatal("failure replayed or not reported", calls, err)
			}
			var report struct {
				Manifest string `json:"resume_manifest"`
			}
			raw, err := os.ReadFile(filepath.Join(dir, "recovery", "failure.json"))
			if err != nil {
				t.Fatal(err)
			}
			if err = json.Unmarshal(raw, &report); err != nil {
				t.Fatal(err)
			}
			m, err := readResumeManifest(report.Manifest)
			if err != nil || m.GlobalStep != completed || !m.Grid.ValidationPending {
				t.Fatal("bad recovery point", err)
			}
			resumedDir := t.TempDir()
			got, err := runGridTrain(cfg, manifest, TrainOptions{Resume: report.Manifest, CheckpointDir: resumedDir, CheckpointEvery: 2, ValEvery: 2})
			if err != nil {
				t.Fatal(err)
			}
			if got.LastLoss != control.LastLoss || got.LastValLoss != control.LastValLoss {
				t.Fatal("recovery changed losses", got, control)
			}
			a, err := resolveResumeManifest(controlDir)
			if err != nil {
				t.Fatal(err)
			}
			b, err := resolveResumeManifest(resumedDir)
			if err != nil {
				t.Fatal(err)
			}
			x, err := loadResumeState(a)
			if err != nil {
				t.Fatal(err)
			}
			y, err := loadResumeState(b)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(x.Trainer, y.Trainer) || !reflect.DeepEqual(a.Grid, b.Grid) {
				t.Fatal("state/sampler mismatch after recovery")
			}
			shapes, err := computeWeightShapes(cfg)
			if err != nil {
				t.Fatal(err)
			}
			wx, err := loadSafetensorsWeights(x.ModelPath, shapes)
			if err != nil {
				t.Fatal(err)
			}
			wy, err := loadSafetensorsWeights(y.ModelPath, shapes)
			if err != nil || !reflect.DeepEqual(wx, wy) {
				t.Fatal("weights mismatch", err)
			}
		})
	}
}
