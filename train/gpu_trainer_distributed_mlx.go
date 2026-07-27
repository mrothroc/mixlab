//go:build mlx && cgo && (darwin || linux)

package train

import (
	"fmt"

	"github.com/mrothroc/mixlab/gpu"
)

// SubmitStepGPU submits one training step without blocking on loss readback.
func (t *mlxGPUTrainer) SubmitStepGPU(
	xTok, yTok []int,
	batchSize, seqLen int,
	lr float32,
) error {
	t.setLRScale(lr)
	inputs, err := t.makeInputs(xTok, yTok, batchSize, seqLen)
	if err != nil {
		return err
	}
	if t.distributed != nil {
		if err := gpu.TrainerSubmitStepWithNormalizer(
			t.handle,
			inputs,
			float32(batchSize*seqLen),
		); err != nil {
			return err
		}
	} else if err := gpu.TrainerSubmitStep(t.handle, inputs); err != nil {
		return err
	}
	t.trainingStep++
	return nil
}

func (t *mlxGPUTrainer) SubmitObjectiveStepGPU(
	batch objectiveBatch,
	batchSize, seqLen int,
	lr float32,
) error {
	t.setLRScale(lr)
	inputs, err := t.makeObjectiveInputs(batch, batchSize, seqLen)
	if err != nil {
		return err
	}
	if t.distributed != nil {
		if !batch.lossNormalizerSet {
			return fmt.Errorf("distributed objective batch is missing loss_normalizer")
		}
		if err := gpu.TrainerSubmitStepWithNormalizer(
			t.handle,
			inputs,
			batch.lossNormalizer,
		); err != nil {
			return err
		}
	} else if err := gpu.TrainerSubmitStep(t.handle, inputs); err != nil {
		return err
	}
	t.trainingStep++
	return nil
}

func (t *mlxGPUTrainer) DistributedContextActiveGPU() bool {
	return t != nil && t.distributed != nil
}

func (t *mlxGPUTrainer) DistributedStageTraceGPU() ([]string, error) {
	if t == nil || t.distributed == nil {
		return nil, nil
	}
	return gpu.TrainerLastStageTrace(t.handle)
}

func (t *mlxGPUTrainer) DistributedBucketMetadataGPU() (gpu.DistributedBucketMetadata, error) {
	if t == nil || t.distributed == nil {
		return gpu.DistributedBucketMetadata{}, nil
	}
	return gpu.TrainerDistributedBucketMetadata(t.handle)
}

func (t *mlxGPUTrainer) DistributedStepTelemetryGPU() (
	*distributedTrainingTelemetry,
	error,
) {
	if t == nil || t.distributed == nil {
		return nil, nil
	}
	metrics, err := gpu.TrainerDistributedStepMetrics(t.handle)
	if err != nil {
		return nil, err
	}
	buckets, err := gpu.TrainerDistributedBucketMetadata(t.handle)
	if err != nil {
		return nil, err
	}
	accumulationSteps := t.distributed.AccumulationSteps
	if accumulationSteps == 0 {
		accumulationSteps = 1
	}
	worldSize := t.distributed.GroupRuntime.WorldSize()
	effectiveTokensPerUpdate := uint64(
		t.distributedLocalBatchTokens * worldSize * accumulationSteps,
	)
	effectiveGlobalTokens := metrics.Microsteps *
		uint64(t.distributedLocalBatchTokens*worldSize)
	globalTokensPerSec := 0.0
	if metrics.TotalUS > 0 {
		globalTokensPerSec = float64(effectiveTokensPerUpdate) /
			(float64(metrics.TotalUS) / 1e6)
	}
	effectiveBandwidth := 0.0
	if metrics.GradientAllReduceUS > 0 {
		effectiveBandwidth = float64(buckets.TotalBytes) /
			(float64(metrics.GradientAllReduceUS) * 1e3)
	}
	return &distributedTrainingTelemetry{
		ComputeMS:                float64(metrics.ComputeUS) / 1e3,
		WaitMS:                   float64(metrics.WaitUS) / 1e3,
		CollectiveMS:             float64(metrics.CollectiveUS) / 1e3,
		AllReduceMS:              float64(metrics.GradientAllReduceUS) / 1e3,
		EffectiveBandwidthGBSec:  effectiveBandwidth,
		GlobalTokensPerSec:       globalTokensPerSec,
		Microsteps:               metrics.Microsteps,
		OptimizerAttempts:        metrics.OptimizerAttempts,
		EffectiveGlobalTokens:    effectiveGlobalTokens,
		EffectiveTokensPerUpdate: effectiveTokensPerUpdate,
		GradientBytes:            buckets.TotalBytes,
		BucketCount:              buckets.BucketCount,
		WorldSize:                worldSize,
		AccumulationSteps:        accumulationSteps,
	}, nil
}
