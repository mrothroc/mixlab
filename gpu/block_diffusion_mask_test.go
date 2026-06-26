//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"math"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestBlockDiffusionMaskMatchesOracle(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}
	const (
		B = 4
		H = 1
		T = 4
	)
	blockStarts := []int32{0, 2, 1, 0}
	blockEnds := []int32{2, 4, 2, 4}
	scores := make([]float32, B*H*T*T)
	for i := range scores {
		scores[i] = float32(i) / 10
	}

	prog := ir.NewProgram(1)
	prog.DeclareInput("scores", ir.TensorFloat32, []int{B, H, T, T})
	prog.DeclareInput("diffusion_block_start", ir.TensorInt32, []int{B})
	prog.DeclareInput("diffusion_block_end", ir.TensorInt32, []int{B})
	prog.DeclareInput("targets", ir.TensorInt32, []int{B * T * T})
	prog.DeclareOutput("masked", ir.TensorFloat32, []int{B, H, T, T})
	prog.DeclareOutput("loss", ir.TensorFloat32, []int{1})
	prog.BlockDiffusionMask("scores", "diffusion_block_start", "diffusion_block_end", T, "masked")
	prog.Reshape("masked", []int{B * T * T, H}, "masked_flat")
	prog.MatMul("masked_flat", "w0", "logits")
	prog.CrossEntropy("logits", "targets", "loss")

	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()
	w0, err := FromData([]float32{0}, 1, 1)
	if err != nil {
		t.Fatalf("FromData w0: %v", err)
	}
	defer FreeHandle(w0)

	inputs := []TensorInput{
		{Name: "scores", DType: TensorFloat32, Shape: []int{B, H, T, T}, Data: scores},
		{Name: "diffusion_block_start", DType: TensorInt32, Shape: []int{B}, Data: blockStarts},
		{Name: "diffusion_block_end", DType: TensorInt32, Shape: []int{B}, Data: blockEnds},
		{Name: "targets", DType: TensorInt32, Shape: []int{B * T * T}, Data: make([]int32, B*T*T)},
	}
	got, err := evalProgramOutput(gpuProg, []int64{w0}, inputs, "masked")
	if err != nil {
		t.Fatalf("evalProgramOutput(masked): %v", err)
	}
	want := blockDiffusionMaskOracle(scores, blockStarts, blockEnds, B, H, T)
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > causalMaskTol {
			t.Fatalf("masked[%d]=%g want %g", i, got[i], want[i])
		}
	}
}

func blockDiffusionMaskOracle(scores []float32, blockStarts, blockEnds []int32, B, H, T int) []float32 {
	out := append([]float32(nil), scores...)
	for b := 0; b < B; b++ {
		start := int(blockStarts[b])
		end := int(blockEnds[b])
		for h := 0; h < H; h++ {
			for q := 0; q < T; q++ {
				for k := 0; k < T; k++ {
					masked := k > q
					if q >= start && q < end {
						masked = k >= end
					}
					if masked {
						out[((b*H+h)*T+q)*T+k] = causalMaskMaskValue
					}
				}
			}
		}
	}
	return out
}
