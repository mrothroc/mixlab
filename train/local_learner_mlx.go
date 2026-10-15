//go:build mlx && cgo && (darwin || linux)

package train

import (
	"fmt"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/gpu"
)

type mlxLocalLearner struct {
	t                 *mlxGPUTrainer
	batchSize, seqLen int
}

func makeLocalLearnerFactory(cfg *ArchConfig, program *arch.Program, seed uint64) (localLearnerFactory, error) {
	evalProgram, e := arch.BuildTrainingIRProgramFromConfig(cfg, arch.TrainingProgramState{Objective: arch.ObjectiveCausal, DropoutInactive: true})
	if e != nil {
		return nil, e
	}
	return func(weights [][]float32, evaluation bool) (localLearnerEngine, error) {
		return localLearnerOnThread(func() (localLearnerEngine, error) {
			p := program
			if evaluation {
				p = evalProgram
			}
			trainer, e := initGPUTrainer(p, cfg, weights, nil)
			if e != nil {
				return nil, e
			}
			t := trainer.(*mlxGPUTrainer)
			t.trainingSeed = seed
			if !evaluation {
				k := cfg.Training.Distributed.GradientAccumulationSteps
				if e = gpu.TrainerSetLocalAccumulation(t.handle, k); e != nil {
					trainer.CloseTrainer()
					return nil, e
				}
				t.localAccumulation = k
			}
			return &mlxLocalLearner{t, cfg.Training.BatchTokens / cfg.SeqLen, cfg.SeqLen}, nil
		})
	}, nil
}

func (l *mlxLocalLearner) step(b objectiveBatch, step int, lr float32) (float32, error) {
	if e := l.t.SetTrainingStepGPU(step); e != nil {
		return 0, e
	}
	if e := l.t.SubmitObjectiveStepGPU(b, l.batchSize, l.seqLen, lr); e != nil {
		return 0, e
	}
	return l.t.CollectLossGPU()
}
func (l *mlxLocalLearner) weights() ([][]float32, error)             { return l.t.ReadWeights() }
func (l *mlxLocalLearner) state() (gpu.TrainerStateSnapshot, error)  { return l.t.ReadTrainerState() }
func (l *mlxLocalLearner) stats() (gpu.TrainerOptimizerStats, error) { return l.t.OptimizerStatsGPU() }
func (l *mlxLocalLearner) restore(v gpu.TrainerStateSnapshot, step int) error {
	if e := l.t.RestoreTrainerState(v); e != nil {
		return e
	}
	return l.t.SetTrainingStepGPU(step)
}
func (l *mlxLocalLearner) install(w [][]float32) error {
	if len(w) != len(l.t.shapes) {
		return fmt.Errorf("incomplete global model")
	}
	handles := make([]int64, len(w))
	defer gpu.FreeHandles(handles)
	for i, shape := range l.t.shapes {
		var e error
		// The trainer's legacy scalar/vector storage is [1,N]. Preserve it.
		if len(shape.Shape) == 1 {
			handles[i], e = gpu.FromData(w[i], 1, len(w[i]))
		} else {
			handles[i], e = gpu.FromDataShape(w[i], shape.Shape)
		}
		if e != nil {
			return e
		}
	}
	return gpu.TrainerInstallWeights(l.t.handle, handles)
}
func (l *mlxLocalLearner) evaluate(b objectiveBatch) (float32, error) {
	return l.t.EvaluateObjectiveGPU(b, l.batchSize, l.seqLen)
}
func (l *mlxLocalLearner) close() { l.t.CloseTrainer() }
