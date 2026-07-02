//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/gpu"
)

func TestRTDMultiheadExamplesRunTwoMLXSteps(t *testing.T) {
	if !mlxAvailable() || !gpu.Available() {
		t.Skip("MLX backend not available")
	}
	for _, filename := range []string{
		"multihead_mntp_rtd_tiny.json",
		"multihead_mntp_rtd_dedicated_tiny.json",
	} {
		t.Run(filename, func(t *testing.T) {
			cfg, err := LoadArchConfig(filepath.Join("examples", filename))
			if err != nil {
				t.Fatalf("LoadArchConfig(%s): %v", filename, err)
			}
			cfg.Training.Steps = 2
			cfg.Training.WarmupSteps = 0
			cfg.Training.HoldSteps = 0
			cfg.Training.WarmdownSteps = 0

			trainDir := filepath.Join(t.TempDir(), "data")
			if err := os.MkdirAll(trainDir, 0o755); err != nil {
				t.Fatalf("MkdirAll(%s): %v", trainDir, err)
			}
			writeInferenceShard(t, filepath.Join(trainDir, "train_000.bin"), rtdSmokeTokens(cfg.VocabSize, 4096))

			result, err := runTrain(cfg, filepath.Join(trainDir, "train_*.bin"), TrainOptions{LogEvery: 0, ValEvery: 0})
			if err != nil {
				t.Fatalf("runTrain: %v", err)
			}
			for name, v := range map[string]float64{
				"first": result.FirstLoss,
				"last":  result.LastLoss,
			} {
				if v <= 0 || math.IsNaN(v) || math.IsInf(v, 0) {
					t.Fatalf("%s loss=%g, want finite positive", name, v)
				}
			}
		})
	}
}

func TestRTDDedicatedGeneratorCompiledStepCacheIsStable(t *testing.T) {
	if !mlxAvailable() || !gpu.Available() {
		t.Skip("MLX backend not available")
	}
	t.Setenv("MIXLAB_RTD_COMPILED_GENERATOR_SAMPLER", "1")

	cfg, err := LoadArchConfig(filepath.Join("examples", "multihead_mntp_rtd_dedicated_tiny.json"))
	if err != nil {
		t.Fatalf("LoadArchConfig: %v", err)
	}
	cfg.Training.Steps = 3
	cfg.Training.WarmupSteps = 0
	cfg.Training.HoldSteps = 0
	cfg.Training.WarmdownSteps = 0

	prog, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{Objective: arch.ObjectiveMultihead})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig: %v", err)
	}
	trainerIface, err := initGPUTrainer(prog, cfg, nil, nil)
	if err != nil {
		t.Fatalf("initGPUTrainer: %v", err)
	}
	defer trainerIface.CloseTrainer()
	trainer, ok := trainerIface.(*mlxGPUTrainer)
	if !ok {
		t.Fatalf("trainer type %T does not expose MLX compile stats", trainerIface)
	}

	raw := deterministicRTDRawBatch(cfg)
	batchSize := cfg.Training.BatchTokens / cfg.SeqLen
	for step := 0; step < 3; step++ {
		prepared, err := prepareObjectiveBatchWithSeqLen(cfg, raw, step, arch.ObjectiveMultihead, cfg.SeqLen)
		if err != nil {
			t.Fatalf("prepareObjectiveBatchWithSeqLen step %d: %v", step, err)
		}
		prepared, err = maybeAttachRTDCorruption(trainer, cfg, raw, step, prepared, batchSize, cfg.SeqLen, arch.ObjectiveMultihead)
		if err != nil {
			t.Fatalf("maybeAttachRTDCorruption step %d: %v", step, err)
		}
		if err := submitPreparedStepGPU(trainer, prepared, batchSize, cfg.SeqLen, float32(cfg.Training.LR)); err != nil {
			t.Fatalf("submitPreparedStepGPU step %d: %v", step, err)
		}
		loss, err := trainer.CollectLossGPU()
		if err != nil {
			t.Fatalf("CollectLossGPU step %d: %v", step, err)
		}
		if loss <= 0 || math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
			t.Fatalf("loss step %d=%g, want finite positive", step, loss)
		}
	}

	stats, err := trainer.CompileStatsGPU()
	if err != nil {
		t.Fatalf("CompileStatsGPU: %v", err)
	}
	if stats.TrainingStepCacheMisses != 1 {
		t.Fatalf("training step cache misses=%d, want 1 (hits=%d)", stats.TrainingStepCacheMisses, stats.TrainingStepCacheHits)
	}
	if stats.TrainingStepCacheHits < 2 {
		t.Fatalf("training step cache hits=%d, want at least 2", stats.TrainingStepCacheHits)
	}
	if stats.CategoricalSamplerCacheMisses != 1 {
		t.Fatalf("categorical sampler cache misses=%d, want 1 (hits=%d)", stats.CategoricalSamplerCacheMisses, stats.CategoricalSamplerCacheHits)
	}
	if stats.CategoricalSamplerCacheHits < 2 {
		t.Fatalf("categorical sampler cache hits=%d, want at least 2", stats.CategoricalSamplerCacheHits)
	}
}

func rtdSmokeTokens(vocabSize, n int) []uint16 {
	out := make([]uint16, n)
	span := vocabSize - 2
	if span <= 0 {
		span = 1
	}
	for i := range out {
		out[i] = uint16((i % span) + 2)
	}
	return out
}

func deterministicRTDRawBatch(cfg *ArchConfig) trainBatch {
	need := cfg.Training.BatchTokens
	x := make([]int, need)
	y := make([]int, need)
	span := cfg.VocabSize - 2
	if span <= 0 {
		span = 1
	}
	for i := 0; i < need; i++ {
		x[i] = (i % span) + 2
		y[i] = ((i + 1) % span) + 2
	}
	return trainBatch{x: x, y: y}
}
