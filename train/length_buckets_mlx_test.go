//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"testing"

	"github.com/mrothroc/mixlab/arch"
)

func TestLengthBucketsMLXProgramSwitchAndPartialBatch(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	cfg := discreteCodebookTrainTestConfig(t)
	cfg.Training.BatchTokens = 10
	cfg.Training.LengthBuckets = []int{2, 4}
	baseProgram, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	trainerInterface, err := initGPUTrainer(baseProgram, cfg, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	trainer := trainerInterface.(*mlxGPUTrainer)
	defer trainer.CloseTrainer()

	shortCfg := *cfg
	shortCfg.SeqLen = 2
	shortCfg.Training = cfg.Training
	shortCfg.Training.BatchTokens = 10
	shortProgram, err := BuildIRProgramFromConfig(&shortCfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := trainer.SetProgramGPU(shortProgram); err != nil {
		t.Fatal(err)
	}
	shortBatch, err := prepareObjectiveBatchWithShape(cfg, trainBatch{
		codebooks: make([]int32, 5*2*cfg.InputAdapter.NumCodebooks),
		labels:    []int32{0, 1, 2, 0, 1}, validMask: []float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
		exampleMask: []float32{1, 1, 1, 1, 0},
	}, 0, arch.ObjectiveClassification, 5, 2)
	if err != nil {
		t.Fatal(err)
	}
	shortLoss, err := trainer.TrainObjectiveStepGPU(shortBatch, 5, 2, 1e-3)
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(float64(shortLoss)) || math.IsInf(float64(shortLoss), 0) {
		t.Fatalf("short-bucket loss=%g", shortLoss)
	}

	if err := trainer.SetProgramGPU(baseProgram); err != nil {
		t.Fatal(err)
	}
	fullBatch, err := prepareObjectiveBatchWithShape(cfg, trainBatch{
		codebooks: make([]int32, 2*4*cfg.InputAdapter.NumCodebooks),
		labels:    []int32{0, 1}, validMask: []float32{1, 1, 1, 1, 1, 1, 1, 1},
		exampleMask: []float32{1, 1},
	}, 1, arch.ObjectiveClassification, 2, 4)
	if err != nil {
		t.Fatal(err)
	}
	fullLoss, err := trainer.TrainObjectiveStepGPU(fullBatch, 2, 4, 1e-3)
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(float64(fullLoss)) || math.IsInf(float64(fullLoss), 0) {
		t.Fatalf("full-bucket loss=%g", fullLoss)
	}
	if len(trainer.programCache) != 2 {
		t.Fatalf("compiled program cache entries=%d want=2", len(trainer.programCache))
	}
}
