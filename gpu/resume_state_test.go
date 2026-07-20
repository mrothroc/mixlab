//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"math"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func resumeStateProgram(t *testing.T) *Program {
	t.Helper()
	prog := ir.NewProgram(1)
	prog.DeclareInput("x", ir.TensorFloat32, []int{2, 2})
	prog.DeclareOutput("loss", ir.TensorFloat32, []int{1})
	prog.Mul("w0", "x", "scaled")
	prog.MeanAxis("scaled", 0, "mean0")
	prog.MeanAxis("mean0", 0, "loss")
	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	return gpuProg
}

func resumeStateTrainer(t *testing.T, prog *Program, weights []float32, group OptimizerGroup) TrainerHandle {
	t.Helper()
	handle, err := FromDataShape(append([]float32(nil), weights...), []int{2, 2})
	if err != nil {
		t.Fatalf("FromDataShape: %v", err)
	}
	t.Cleanup(func() { FreeHandle(handle) })
	trainer, err := CreateTrainer(prog, []int64{handle}, TrainerOptimizerSpec{
		Groups:        []OptimizerGroup{group},
		Weights:       []WeightOptimizer{{GroupIndex: 0}},
		DefaultBaseLR: group.LR,
	})
	if err != nil {
		t.Fatalf("CreateTrainer: %v", err)
	}
	t.Cleanup(func() { TrainerDestroy(trainer) })
	return trainer
}

func resumeStateWeights(t *testing.T, trainer TrainerHandle) []float32 {
	t.Helper()
	out := make([]float32, 4)
	if err := TrainerReadWeight(trainer, 0, out); err != nil {
		t.Fatalf("TrainerReadWeight: %v", err)
	}
	return out
}

func TestTrainerStateSnapshotRestoreMatchesUninterruptedOptimizerStep(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}
	prog := resumeStateProgram(t)
	defer prog.Destroy()
	inputs := []TensorInput{{Name: "x", DType: TensorFloat32, Shape: []int{2, 2}, Data: []float32{0.5, -0.25, 0.75, -1}}}
	initial := []float32{0.4, -0.2, 0.1, 0.7}
	for _, tc := range []struct {
		name       string
		group      OptimizerGroup
		stateCount int
	}{
		{name: "adamw", group: OptimizerGroup{Kind: OptimizerAdamW, LR: 1e-2, Beta1: 0.9, Beta2: 0.99, Epsilon: 1e-8}, stateCount: 2},
		{name: "lamb", group: OptimizerGroup{Kind: OptimizerLAMB, LR: 1e-2, Beta1: 0.9, Beta2: 0.99, Epsilon: 1e-8, LAMBTrustRatioCap: 10}, stateCount: 2},
		{name: "muon", group: OptimizerGroup{Kind: OptimizerMuon, LR: 1e-2, Beta1: 0.9, BackendSteps: 3, Nesterov: true}, stateCount: 1},
		{name: "normuon", group: OptimizerGroup{Kind: OptimizerMuon, LR: 1e-2, Beta1: 0.9, Beta2: 0.95, Epsilon: 1e-8, BackendSteps: 3, Nesterov: true, MuonNormalization: MuonNormalizationNorMuon}, stateCount: 2},
		{name: "sgd_momentum", group: OptimizerGroup{Kind: OptimizerSGD, LR: 1e-2, Beta1: 0.9}, stateCount: 1},
		{name: "sgd_zero_momentum", group: OptimizerGroup{Kind: OptimizerSGD, LR: 1e-2}, stateCount: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			uninterrupted := resumeStateTrainer(t, prog, initial, tc.group)
			for step := 0; step < 2; step++ {
				if _, err := TrainerStep(uninterrupted, inputs); err != nil {
					t.Fatalf("TrainerStep(%d): %v", step, err)
				}
			}
			snapshot, err := TrainerStateSnapshotRead(uninterrupted)
			if err != nil {
				t.Fatalf("TrainerStateSnapshotRead: %v", err)
			}
			if len(snapshot.Tensors) != tc.stateCount {
				t.Fatalf("state tensors=%d want=%d: %+v", len(snapshot.Tensors), tc.stateCount, snapshot.Tensors)
			}
			checkpointWeights := resumeStateWeights(t, uninterrupted)
			if _, err := TrainerStep(uninterrupted, inputs); err != nil {
				t.Fatalf("uninterrupted final step: %v", err)
			}

			resumed := resumeStateTrainer(t, prog, checkpointWeights, tc.group)
			if err := TrainerStateSnapshotRestore(resumed, snapshot); err != nil {
				t.Fatalf("TrainerStateSnapshotRestore: %v", err)
			}
			if _, err := TrainerStep(resumed, inputs); err != nil {
				t.Fatalf("resumed final step: %v", err)
			}
			got := resumeStateWeights(t, resumed)
			want := resumeStateWeights(t, uninterrupted)
			for i := range want {
				if diff := math.Abs(float64(got[i] - want[i])); diff > 1e-6 {
					t.Fatalf("weight[%d]=%g want=%g diff=%g", i, got[i], want[i], diff)
				}
			}
			stats, err := TrainerOptimizerStatsSnapshot(resumed)
			if err != nil {
				t.Fatal(err)
			}
			if stats.AttemptedSteps != 3 || stats.CommittedSteps != 3 {
				t.Fatalf("restored counters=%+v", stats)
			}
		})
	}
}
