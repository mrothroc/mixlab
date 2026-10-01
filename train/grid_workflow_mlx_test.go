//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/data"
)

func TestGridTwoStageFreezeAndResume(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX required")
	}
	manifest := prepareTinyGrid(t)
	for _, optimizer := range []string{"adamw", "lamb"} {
		t.Run(optimizer, func(t *testing.T) {
			cfg, err := LoadArchConfig("../examples/grid_two_stage_1.json")
			if err != nil {
				t.Fatal(err)
			}
			cfg.Training.Steps = 4
			cfg.Training.Optimizer = optimizer
			cfg.Training.LR = .001
			cfg.Training.EarlyStop = &EarlyStopSpec{Patience: 100}
			dir := t.TempDir()
			first := filepath.Join(dir, "stage1.safetensors")
			if _, err = runGridTrain(cfg, manifest, TrainOptions{SafetensorsPath: first, ValEvery: 2}); err != nil {
				t.Fatal(err)
			}
			shapes, err := computeWeightShapes(cfg)
			if err != nil {
				t.Fatal(err)
			}
			before, err := loadSafetensorsWeights(first, shapes)
			if err != nil {
				t.Fatal(err)
			}
			init := initWeightData(shapes, cfg.Training.Seed, cfg.Training.WeightInit, cfg.Training.WeightInitStd)
			if !reflect.DeepEqual(before[2:], init[2:]) {
				t.Fatal("stage 1 changed unreachable W")
			}
			cfg.DenseRegression.Output = "network.output2"
			cfg.Training.Steps = 8
			cfg.Training.Freeze = []string{"network.U1.*"}
			cfg.Training.InitFrom = first
			cfg.Training.WeightDecayPolicy = "all"
			cfg.Training.WeightDecay = .05
			cfg.Training.MatrixWeightDecay = .05
			cfg.Training.ScalarWeightDecay = .05
			cfg.Training.GradClip = 1
			fullDir := filepath.Join(dir, "full")
			full, err := runGridTrain(cfg, manifest, TrainOptions{CheckpointDir: fullDir, CheckpointEvery: 1, ValEvery: 2})
			if err != nil {
				t.Fatal(err)
			}
			shapes, err = computeWeightShapes(cfg)
			if err != nil {
				t.Fatal(err)
			}
			want, err := loadSafetensorsWeights(filepath.Join(fullDir, "step_000008.st"), shapes)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(before[:2], want[:2]) || reflect.DeepEqual(before[2:], want[2:]) {
				t.Fatal("freeze failed or W did not train")
			}
			checkGridSecondStageCPU(t, cfg, before, want, manifest)
			for _, step := range []int{3, 4} { // mid-epoch and partial-batch epoch boundary
				resumeDir := filepath.Join(dir, "resume", resumeManifestFilename(step))
				cfg.Training.InitFrom = ""
				got, err := runGridTrain(cfg, manifest, TrainOptions{Resume: filepath.Join(fullDir, resumeManifestFilename(step)), CheckpointDir: resumeDir, CheckpointEvery: 1, ValEvery: 2})
				if err != nil {
					t.Fatal(err)
				}
				if got.LastLoss != full.LastLoss || got.LastValLoss != full.LastValLoss {
					t.Fatalf("resume metrics differ: %+v vs %+v", got, full)
				}
				actual, err := loadSafetensorsWeights(filepath.Join(resumeDir, "step_000008.st"), shapes)
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(actual, want) {
					t.Fatal("resume weights differ")
				}
				a, err := resolveResumeManifest(filepath.Join(fullDir, resumeManifestFilename(8)))
				if err != nil {
					t.Fatal(err)
				}
				b, err := resolveResumeManifest(filepath.Join(resumeDir, resumeManifestFilename(8)))
				if err != nil {
					t.Fatal(err)
				}
				sa, err := loadResumeState(a)
				if err != nil {
					t.Fatal(err)
				}
				sb, err := loadResumeState(b)
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(sa.Trainer, sb.Trainer) || !reflect.DeepEqual(a.Grid, b.Grid) || !reflect.DeepEqual(a.EarlyStop, b.EarlyStop) || !reflect.DeepEqual(a.Schedule, b.Schedule) {
					t.Fatal("resume state/moments/replay/schedule/metrics differ")
				}
				for _, tensor := range sb.Trainer.Tensors {
					if tensor.WeightIndex < 2 {
						t.Fatal("frozen U1 received optimizer state")
					}
				}
			}
			cfg.Training.Freeze = []string{"network.U1.bias"}
			if _, err = runGridTrain(cfg, manifest, TrainOptions{Resume: filepath.Join(fullDir, resumeManifestFilename(3)), ValEvery: 2}); err == nil || !strings.Contains(err.Error(), "config does not match") {
				t.Fatal("accepted changed freeze", err)
			}
			cfg.Training.Freeze = []string{"network.U1.*"}
			cfg.DenseRegression.Output = "network.output1"
			if _, err = runGridTrain(cfg, manifest, TrainOptions{Resume: filepath.Join(fullDir, resumeManifestFilename(3)), ValEvery: 2}); err == nil {
				t.Fatal("accepted stage transition as resume")
			}
		})
	}
}

func TestGridResumePreservesNewerBestArtifact(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX required")
	}
	manifest := prepareTinyGrid(t)
	cfg, err := LoadArchConfig("../examples/grid_regression_tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	cfg.Training.Steps = 4
	dir := t.TempDir()
	if _, err = runGridTrain(cfg, manifest, TrainOptions{CheckpointDir: dir, CheckpointEvery: 3, ValEvery: 1}); err != nil {
		t.Fatal(err)
	}
	hash, err := data.GridDatasetIdentity(manifest)
	if err != nil {
		t.Fatal(err)
	}
	b, err := newGridBestArtifact(cfg, dir, hash, true)
	if err != nil {
		t.Fatal(err)
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatal(err)
	}
	w, err := loadSafetensorsWeights(b.path, shapes)
	if err != nil {
		t.Fatal(err)
	}
	// Inject a best selection ahead of step 3 to make the crash window
	// deterministic, independent of the tiny model's loss trajectory.
	if err = b.save(cfg, fakeWeightReader{w}, shapes, 0); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(b.path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = runGridTrain(cfg, manifest, TrainOptions{Resume: dir, CheckpointDir: dir, CheckpointEvery: 3, ValEvery: 1}); err != nil {
		t.Fatal(err)
	}
	after, err := os.ReadFile(b.path)
	if err != nil || !reflect.DeepEqual(before, after) {
		t.Fatal("replayed validation overwrote newer best", err)
	}
}

// The stage-2 1x1 convolution consumes [stop_gradient(U1), raw channels].
// Check both before and after updates, including unchanged first-stage outputs.
func checkGridSecondStageCPU(t *testing.T, cfg *ArchConfig, before, after [][]float32, manifest string) {
	t.Helper()
	p, err := arch.BuildGridIRProgram(cfg, true)
	if err != nil {
		t.Fatal(err)
	}
	ds, err := data.OpenGridDataset(manifest, "val")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = ds.Close() }()
	b, err := ds.ReadBatch([]int{0, 1}, 2)
	if err != nil {
		t.Fatal(err)
	}
	for _, weights := range [][][]float32{before, after} {
		tr, err := initGPUTrainer(p, cfg, weights, nil)
		if err != nil {
			t.Fatal(err)
		}
		_, err = tr.(gpuObjectiveOutputEvaluator).EvaluateObjectiveGPUWithOutputs(objectiveBatch{grid: &b}, 2, 0, []string{"predictions"})
		if err != nil {
			tr.CloseTrainer()
			t.Fatal(err)
		}
		out, err := readTrainerOutput(tr, "predictions", []int{2, 8, 8, 1})
		tr.CloseTrainer()
		if err != nil {
			t.Fatal(err)
		}
		for row := 0; row < 2; row++ {
			for y := 0; y < 8; y++ {
				for x := 0; x < 8; x++ {
					first := float64(weights[1][0])
					for ky := 0; ky < 3; ky++ {
						for kx := 0; kx < 3; kx++ {
							iy, ix := y+ky-1, x+kx-1
							if iy < 0 || iy >= 8 || ix < 0 || ix >= 8 {
								continue
							}
							for c := 0; c < 2; c++ {
								first += float64(b.Inputs[((row*8+iy)*8+ix)*2+c]) * float64(weights[0][(ky*3+kx)*2+c])
							}
						}
					}
					pos := (row*8+y)*8 + x
					want := float64(weights[3][0]) + first*float64(weights[2][0]) + float64(b.Inputs[pos*2])*float64(weights[2][1]) + float64(b.Inputs[pos*2+1])*float64(weights[2][2])
					if math.Abs(float64(out[pos])-want) > 1e-5 {
						t.Fatalf("stage2 pixel=%d got=%g CPU=%g", pos, out[pos], want)
					}
				}
			}
		}
	}
	// Native prediction remains augmentation-free and input-only.
	dir := t.TempDir()
	config := filepath.Join(dir, "config.json")
	raw, err := os.ReadFile("../examples/grid_two_stage_2.json")
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(config, raw, 0600); err != nil {
		t.Fatal(err)
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatal(err)
	}
	checkpoint := filepath.Join(dir, "weights.safetensors")
	if err = exportSafetensors(checkpoint, cfg, shapes, after); err != nil {
		t.Fatal(err)
	}
	if err = RunPredictGrid(PredictGridOptions{ConfigPath: config, SafetensorsLoad: checkpoint, Input: manifest, Output: filepath.Join(dir, "predictions")}); err != nil {
		t.Fatal(err)
	}
}
