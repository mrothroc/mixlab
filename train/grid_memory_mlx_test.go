//go:build mlx && cgo && (darwin || linux)

package train

import (
	"reflect"
	"runtime"
	"testing"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/data"
)

func TestGridMemoryPreflightRestoresTraining(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX required")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	for _, opt := range []string{"adamw", "lamb"} {
		t.Run(opt, func(t *testing.T) {
			cfg := gridPhaseConfig(t)
			cfg.Training.Optimizer = opt
			p, err := arch.BuildGridIRProgram(cfg, true)
			if err != nil {
				t.Fatal(err)
			}
			shapes, err := computeWeightShapes(cfg)
			if err != nil {
				t.Fatal(err)
			}
			tr, err := initGPUTrainer(p, cfg, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			defer tr.CloseTrainer()
			b := data.GridBatch{BatchSize: 2, Count: 2, Inputs: make([]float32, 256), Targets: make([]float32, 128), LossMask: make([]float32, 128)}
			for i := range b.Inputs {
				b.Inputs[i] = float32(i%13) / 13
			}
			for i := range b.Targets {
				b.Targets[i] = .3
				b.LossMask[i] = 1
			}
			// Exercise fresh and nonzero moments, as in a resumed run.
			for step := 0; step < 2; step++ {
				before, err := readTrainerWeights(tr)
				if err != nil {
					t.Fatal(err)
				}
				state, err := tr.(gpuTrainerStateReader).ReadTrainerState()
				if err != nil {
					t.Fatal(err)
				}
				if err := preflightGridTraining(tr, shapes, b, step, 1e-4); err != nil {
					t.Fatal(err)
				}
				after, err := readTrainerWeights(tr)
				if err != nil {
					t.Fatal(err)
				}
				got, err := tr.(gpuTrainerStateReader).ReadTrainerState()
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(before, after) || !reflect.DeepEqual(state, got) || tr.(*mlxGPUTrainer).trainingStep != step {
					t.Fatal("preflight changed weights, moments or counters")
				}
				if err := submitPreparedStepGPU(tr, objectiveBatch{grid: &b}, 2, 0, 1e-4); err != nil {
					t.Fatal(err)
				}
				loss, err := tr.CollectLossGPU()
				if err != nil {
					t.Fatal(err)
				}
				expected, err := readTrainerWeights(tr)
				if err != nil {
					t.Fatal(err)
				}
				control, err := initGPUTrainer(p, cfg, before, nil)
				if err != nil {
					t.Fatal(err)
				}
				defer control.CloseTrainer()
				if err := control.(gpuTrainerStateRestorer).RestoreTrainerState(state); err != nil {
					t.Fatal(err)
				}
				if err := submitPreparedStepGPU(control, objectiveBatch{grid: &b}, 2, 0, 1e-4); err != nil {
					t.Fatal(err)
				}
				controlLoss, err := control.CollectLossGPU()
				if err != nil {
					t.Fatal(err)
				}
				controlWeights, err := readTrainerWeights(control)
				if err != nil {
					t.Fatal(err)
				}
				if loss != controlLoss || !reflect.DeepEqual(expected, controlWeights) {
					t.Fatal("preflight changed subsequent update")
				}
			}
		})
	}
}
