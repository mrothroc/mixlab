//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"math"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestSegmentAttentionMaskMatchesOracle(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}
	const (
		B = 2
		H = 1
		T = 4
	)
	tests := []struct {
		name       string
		mode       int
		windowSize int
		causalRows []int32
	}{
		{name: "bidirectional", mode: ir.SegmentMaskModeNone},
		{name: "causal", mode: ir.SegmentMaskModeCausal},
		{name: "selective causal", mode: ir.SegmentMaskModeSelectiveCausal, causalRows: []int32{1, 0}},
		{name: "causal window", mode: ir.SegmentMaskModeCausal, windowSize: 2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			prog := ir.NewProgram(1)
			prog.DeclareInput("scores", ir.TensorFloat32, []int{B, H, T, T})
			prog.DeclareInput("segment_ids", ir.TensorInt32, []int{B, T})
			if tt.mode == ir.SegmentMaskModeSelectiveCausal {
				prog.DeclareInput("causal_rows", ir.TensorInt32, []int{B})
			}
			prog.DeclareInput("targets", ir.TensorInt32, []int{B * T * T})
			prog.DeclareOutput("masked", ir.TensorFloat32, []int{B, H, T, T})
			prog.DeclareOutput("loss", ir.TensorFloat32, []int{1})
			if tt.mode == ir.SegmentMaskModeSelectiveCausal {
				prog.SegmentAttentionMask("scores", "segment_ids", "causal_rows", T, tt.windowSize, tt.mode, "masked")
			} else {
				prog.SegmentAttentionMask("scores", "segment_ids", "", T, tt.windowSize, tt.mode, "masked")
			}
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

			scores := make([]float32, B*H*T*T)
			for i := range scores {
				scores[i] = float32(i) / 10
			}
			segmentIDs := []int32{
				0, 0, 1, 1,
				0, 1, 1, 2,
			}
			inputs := []TensorInput{
				{Name: "scores", DType: TensorFloat32, Shape: []int{B, H, T, T}, Data: scores},
				{Name: "segment_ids", DType: TensorInt32, Shape: []int{B, T}, Data: segmentIDs},
				{Name: "targets", DType: TensorInt32, Shape: []int{B * T * T}, Data: make([]int32, B*T*T)},
			}
			if tt.mode == ir.SegmentMaskModeSelectiveCausal {
				inputs = append(inputs, TensorInput{Name: "causal_rows", DType: TensorInt32, Shape: []int{B}, Data: tt.causalRows})
			}
			got, err := evalProgramOutput(gpuProg, []int64{w0}, inputs, "masked")
			if err != nil {
				t.Fatalf("evalProgramOutput(masked): %v", err)
			}
			want := segmentAttentionMaskOracle(scores, segmentIDs, tt.causalRows, B, H, T, tt.windowSize, tt.mode)
			for i := range want {
				if math.Abs(float64(got[i]-want[i])) > causalMaskTol {
					t.Fatalf("masked[%d]=%g want %g", i, got[i], want[i])
				}
			}
		})
	}
}

func segmentAttentionMaskOracle(scores []float32, segmentIDs, causalRows []int32, B, H, T, windowSize, mode int) []float32 {
	out := append([]float32(nil), scores...)
	for b := 0; b < B; b++ {
		for h := 0; h < H; h++ {
			for q := 0; q < T; q++ {
				for k := 0; k < T; k++ {
					masked := segmentIDs[b*T+q] != segmentIDs[b*T+k]
					causal := mode == ir.SegmentMaskModeCausal ||
						(mode == ir.SegmentMaskModeSelectiveCausal && len(causalRows) > b && causalRows[b] > 0)
					if causal && k > q {
						masked = true
					}
					if causal && windowSize > 0 && windowSize < T && k < q-(windowSize-1) {
						masked = true
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
