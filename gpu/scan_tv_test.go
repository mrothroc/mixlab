//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"math"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestScanTVMatchesHandComputedReference(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}

	const (
		scanTVTestB   = 1
		scanTVTestT   = 4
		scanTVTestD   = 2
		scanTVTestTol = 1e-5
	)

	prog := ir.NewProgram(1)
	prog.DeclareInput("x", ir.TensorFloat32, []int{scanTVTestB * scanTVTestT, scanTVTestD})
	prog.DeclareInput("gate", ir.TensorFloat32, []int{scanTVTestB * scanTVTestT, scanTVTestD})
	prog.DeclareInput("targets", ir.TensorInt32, []int{scanTVTestB * scanTVTestT})
	prog.DeclareOutput("logits", ir.TensorFloat32, []int{scanTVTestB * scanTVTestT, scanTVTestD})
	prog.DeclareOutput("loss", ir.TensorFloat32, []int{1})
	prog.ScanTV("x", "gate", "logits", scanTVTestB, scanTVTestT, scanTVTestD)
	prog.CrossEntropy("logits", "targets", "loss")

	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()

	dummyWeight, err := FromData([]float32{0}, 1, 1)
	if err != nil {
		t.Fatalf("FromData dummy weight: %v", err)
	}
	defer FreeHandle(dummyWeight)

	trainer, err := CreateTrainer(gpuProg, []int64{dummyWeight}, TrainerOptimizerSpec{
		Groups: []OptimizerGroup{{
			Kind:        OptimizerAdamW,
			LR:          0.001,
			Beta1:       0.9,
			Beta2:       0.95,
			Epsilon:     1e-8,
			WeightDecay: 0.0,
		}},
		Weights:       []WeightOptimizer{{GroupIndex: 0, Decay: false}},
		DefaultBaseLR: 0.001,
	})
	if err != nil {
		t.Fatalf("CreateTrainer: %v", err)
	}
	defer TrainerDestroy(trainer)

	inputs := []TensorInput{
		{
			Name:  "x",
			DType: TensorFloat32,
			Shape: []int{scanTVTestB * scanTVTestT, scanTVTestD},
			Data: []float32{
				1, 2,
				3, 4,
				5, 6,
				7, 8,
			},
		},
		{
			Name:  "gate",
			DType: TensorFloat32,
			Shape: []int{scanTVTestB * scanTVTestT, scanTVTestD},
			Data: []float32{
				0.0, 0.25,
				0.5, 0.1,
				1.0, 0.8,
				0.25, 0.0,
			},
		},
		{
			Name:  "targets",
			DType: TensorInt32,
			Shape: []int{scanTVTestB * scanTVTestT},
			Data:  []int32{0, 1, 0, 1},
		},
	}

	if _, err := TrainerEvaluate(trainer, inputs); err != nil {
		t.Fatalf("TrainerEvaluate: %v", err)
	}

	got, err := TrainerReadOutput(trainer, "logits", []int{scanTVTestB * scanTVTestT, scanTVTestD})
	if err != nil {
		t.Fatalf("TrainerReadOutput(logits): %v", err)
	}

	want := []float32{
		1.0, 1.5,
		2.0, 3.75,
		2.0, 4.2,
		5.75, 8.0,
	}
	if len(got) != len(want) {
		t.Fatalf("logits length = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if diff := math.Abs(float64(got[i] - want[i])); diff > scanTVTestTol {
			t.Fatalf("logits[%d] = %g, want %g (diff=%g)", i, got[i], want[i], diff)
		}
	}
}
