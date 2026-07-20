//go:build mlx && cgo && (darwin || linux)

package train

import (
	"fmt"

	"github.com/mrothroc/mixlab/gpu"
)

func (t *mlxGPUTrainer) SetTrainingStepGPU(step int) error {
	if step < 0 {
		return fmt.Errorf("training step must be non-negative, got %d", step)
	}
	t.trainingStep = step
	return nil
}

func (t *mlxGPUTrainer) ReadTrainerState() (gpu.TrainerStateSnapshot, error) {
	return gpu.TrainerStateSnapshotRead(t.handle)
}

func (t *mlxGPUTrainer) RestoreTrainerState(snapshot gpu.TrainerStateSnapshot) error {
	return gpu.TrainerStateSnapshotRestore(t.handle, snapshot)
}

func (t *mlxGPUTrainer) OptimizerSpec() gpu.TrainerOptimizerSpec {
	return t.optimizerSpec
}
