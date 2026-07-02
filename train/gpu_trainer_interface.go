package train

import (
	"fmt"

	"github.com/mrothroc/mixlab/gpu"
)

// GPUTrainer and the objective step/eval adapter interfaces plus their small
// dispatch helpers. The concrete implementation is the MLX backend.

// GPUTrainer abstracts the GPU training interface.
// This is implemented by the MLX backend when available.
type GPUTrainer interface {
	TrainStepGPU(xTok, yTok []int, batchSize, seqLen int, lr float32) (float32, error)
	SubmitStepGPU(xTok, yTok []int, batchSize, seqLen int, lr float32) error
	CollectLossGPU() (float32, error)
	FlushGPU() error
	SetQATGPU(mode string) error
	SetWeightGPU(name string, data []float32) error
	EvaluateObjectiveGPU(batch objectiveBatch, batchSize, seqLen int) (float32, error)
	EvaluateGPU(xTok, yTok []int, batchSize, seqLen int) (float32, error)
	EvaluatePerTokenGPU(xTok, yTok []int, batchSize, seqLen int) ([]float32, error)
	EvaluateLoRATTTGPU(xTok, yTok []int, batchSize, seqLen, tttSteps int, tttLR float32, tttRank int) (float32, error)
	CloseTrainer()
}

type gpuWeightCopier interface {
	CopyWeightGPU(dstName, srcName string) error
}

type gpuObjectiveStepSubmitter interface {
	SubmitObjectiveStepGPU(batch objectiveBatch, batchSize, seqLen int, lr float32) error
}

type gpuObjectiveEvaluator interface {
	EvaluateObjectiveGPU(batch objectiveBatch, batchSize, seqLen int) (float32, error)
}

type gpuObjectiveCategoricalSampler interface {
	SampleObjectiveOutputCategoricalGPU(batch objectiveBatch, batchSize, seqLen int, outputName string, rows, vocab int, temperature float64, seed uint64) ([]int, error)
}

type gpuObjectiveCategoricalEagerSampler interface {
	SampleObjectiveOutputCategoricalEagerGPU(batch objectiveBatch, batchSize, seqLen int, outputName string, rows, vocab int, temperature float64, seed uint64) ([]int, error)
}

type gpuCompileStatsReader interface {
	CompileStatsGPU() (gpu.TrainerCompileStats, error)
}

func submitPreparedStepGPU(trainer GPUTrainer, batch objectiveBatch, batchSize, seqLen int, lr float32) error {
	if batch.batchSizeOverride > 0 {
		batchSize = batch.batchSizeOverride
	}
	if batch.lossMask == nil {
		return trainer.SubmitStepGPU(batch.x, batch.y, batchSize, seqLen, lr)
	}
	submitter, ok := trainer.(gpuObjectiveStepSubmitter)
	if !ok {
		return fmt.Errorf("trainer does not support masked objective batches")
	}
	return submitter.SubmitObjectiveStepGPU(batch, batchSize, seqLen, lr)
}

func evaluateTokensViaObjectiveGPU(trainer gpuObjectiveEvaluator, xTok, yTok []int, batchSize, seqLen int) (float32, error) {
	return trainer.EvaluateObjectiveGPU(objectiveBatch{x: xTok, y: yTok}, batchSize, seqLen)
}

func formatCompileStats(trainer GPUTrainer) string {
	reader, ok := trainer.(gpuCompileStatsReader)
	if !ok {
		return ""
	}
	stats, err := reader.CompileStatsGPU()
	if err != nil {
		return ""
	}
	return fmt.Sprintf(
		" compile=train_hits=%d train_misses=%d sampler_hits=%d sampler_misses=%d",
		stats.TrainingStepCacheHits,
		stats.TrainingStepCacheMisses,
		stats.CategoricalSamplerCacheHits,
		stats.CategoricalSamplerCacheMisses,
	)
}
