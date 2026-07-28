//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestTiedDropoutSharesMaskAcrossSequencePositions(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}
	const (
		B = 2
		T = 4
		D = 3
	)
	prog := ir.NewProgram(1)
	prog.DeclareInput("x", ir.TensorFloat32, []int{B * T, D})
	prog.Mul("x", "w0", "scaled")
	prog.TiedDropout("scaled", 0.5, B, T, D, "tied")
	prog.Square("tied", "squared")
	prog.MeanAxis("squared", 0, "mean_channels")
	prog.MeanAxis("mean_channels", 0, "loss")
	prog.DeclareOutput("loss", ir.TensorFloat32, []int{1})
	prog.DeclareOutput("tied", ir.TensorFloat32, []int{B * T, D})
	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatal(err)
	}
	defer gpuProg.Destroy()
	weight, err := FromDataShape([]float32{1, 1, 1}, []int{D})
	if err != nil {
		t.Fatal(err)
	}
	defer FreeHandle(weight)
	trainer, err := CreateTrainer(gpuProg, []int64{weight}, TrainerOptimizerSpec{
		Groups: []OptimizerGroup{{
			Kind: OptimizerAdamW, LR: 0, Beta1: 0.9, Beta2: 0.95, Epsilon: 1e-8,
		}},
		Weights: []WeightOptimizer{{GroupIndex: 0}},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer TrainerDestroy(trainer)
	if err := TrainerSetStepOutputNames(trainer, []string{"tied"}); err != nil {
		t.Fatal(err)
	}
	x := repeatFloat32(B*T*D, 1)
	inputs := []TensorInput{
		{Name: "x", DType: TensorFloat32, Shape: []int{B * T, D}, Data: x},
		{Name: ir.DropoutKeysInput, DType: TensorInt32, Shape: []int{1, 2}, Data: []int32{123, 456}},
	}
	if _, err := TrainerStep(trainer, inputs); err != nil {
		t.Fatal(err)
	}
	got, err := TrainerReadCachedOutput(trainer, "tied", []int{B * T, D})
	if err != nil {
		t.Fatal(err)
	}
	for b := 0; b < B; b++ {
		for d := 0; d < D; d++ {
			want := got[(b*T)*D+d]
			if want != 0 && want != 2 {
				t.Fatalf("mask value batch=%d dim=%d is %g, want 0 or 2", b, d, want)
			}
			for pos := 1; pos < T; pos++ {
				if value := got[(b*T+pos)*D+d]; value != want {
					t.Fatalf("mask changed across time: batch=%d dim=%d pos0=%g pos%d=%g", b, d, want, pos, value)
				}
			}
		}
	}
}
