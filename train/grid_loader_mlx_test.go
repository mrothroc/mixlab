//go:build mlx && cgo && (darwin || linux)

package train

import (
	"path/filepath"
	"reflect"
	"testing"

	"github.com/mrothroc/mixlab/arch"
)

func TestGridLoaderTrainingResumeParity(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX required")
	}
	manifest := prepareTinyGrid(t)
	cfg, err := LoadArchConfig("../examples/grid_two_stage_1.json")
	if err != nil {
		t.Fatal(err)
	}
	cfg.Training.Steps = 6
	zero := 0
	cfg.Training.GridLoader = &arch.GridLoaderSpec{PrefetchBatches: &zero, ReadWorkers: 1}
	serialDir := t.TempDir()
	serial, err := runGridTrain(cfg, manifest, TrainOptions{CheckpointDir: serialDir, CheckpointEvery: 1, ValEvery: 2})
	if err != nil {
		t.Fatal(err)
	}
	cfg.Training.GridLoader = nil
	parallelDir := t.TempDir()
	parallel, err := runGridTrain(cfg, manifest, TrainOptions{CheckpointDir: parallelDir, CheckpointEvery: 1, ValEvery: 2})
	if err != nil {
		t.Fatal(err)
	}
	if serial.FirstLoss != parallel.FirstLoss || serial.LastLoss != parallel.LastLoss || serial.LastValLoss != parallel.LastValLoss {
		t.Fatal("prefetch changed losses")
	}
	compare := func(aDir, bDir string, step int) {
		t.Helper()
		a, err := resolveResumeManifest(filepath.Join(aDir, resumeManifestFilename(step)))
		if err != nil {
			t.Fatal(err)
		}
		b, err := resolveResumeManifest(filepath.Join(bDir, resumeManifestFilename(step)))
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
		if !reflect.DeepEqual(sa.Trainer, sb.Trainer) || !reflect.DeepEqual(a.Grid, b.Grid) {
			t.Fatalf("weights/moments/loss/sampler mismatch step=%d", step)
		}
		shapes, err := computeWeightShapes(cfg)
		if err != nil {
			t.Fatal(err)
		}
		wa, err := loadSafetensorsWeights(sa.ModelPath, shapes)
		if err != nil {
			t.Fatal(err)
		}
		wb, err := loadSafetensorsWeights(sb.ModelPath, shapes)
		if err != nil || !reflect.DeepEqual(wa, wb) {
			t.Fatalf("model weights differ step=%d: %v", step, err)
		}
	}
	for step := 1; step <= 6; step++ {
		compare(serialDir, parallelDir, step)
	}
	for _, step := range []int{1, 2, 3} {
		resumeDir := t.TempDir()
		cfg.Training.GridLoader = &arch.GridLoaderSpec{PrefetchBatches: &zero, ReadWorkers: 1}
		got, err := runGridTrain(cfg, manifest, TrainOptions{Resume: filepath.Join(parallelDir, resumeManifestFilename(step)), CheckpointDir: resumeDir, CheckpointEvery: 1, ValEvery: 2})
		if err != nil {
			t.Fatal(err)
		}
		if got.LastLoss != serial.LastLoss || got.LastValLoss != serial.LastValLoss {
			t.Fatal("resume changed losses")
		}
		compare(serialDir, resumeDir, 6)
	}
	cfg.Training.GridLoader = nil
	resumeDir := t.TempDir()
	if _, err = runGridTrain(cfg, manifest, TrainOptions{Resume: filepath.Join(serialDir, resumeManifestFilename(1)), CheckpointDir: resumeDir, CheckpointEvery: 1, ValEvery: 2}); err != nil {
		t.Fatal(err)
	}
	compare(serialDir, resumeDir, 6)
}
