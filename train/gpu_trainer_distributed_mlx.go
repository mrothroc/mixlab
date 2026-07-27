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
