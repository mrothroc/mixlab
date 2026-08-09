//go:build mlx && cgo && linux

package gpu

import "testing"

func TestS4DSobolevCUDAForwardMatchesForcedFallback(t *testing.T) {
	lockMLXThread(t)
	if !Available() {
		t.Skip("MLX CUDA backend not available")
	}
	const B, T, D, N = 1, 5, 2, 4
	x, weights := s4dParityFixture(B, T, D, N)
	weights = append(weights, []float32{-0.5, 1.25})
	inputs := []TensorInput{{
		Name: "x", DType: TensorFloat32, Shape: []int{B * T, D}, Data: x,
	}}

	primitiveProgram := lowerS4DSobolevProgram(t, B, T, D, N)
	primitiveHandles := s4dSobolevWeightHandles(t, weights, D, N)
	primitive, err := EvalProgramOutput(
		primitiveProgram, primitiveHandles, inputs, "output",
	)
	primitiveProgram.Destroy()
	FreeHandles(primitiveHandles)
	if err != nil {
		t.Fatalf("CUDA primitive forward: %v", err)
	}

	t.Setenv("MIXLAB_S4D_SOBOLEV_DISABLE_CUDA_PRIMITIVE", "1")
	fallbackProgram := lowerS4DSobolevProgram(t, B, T, D, N)
	fallbackHandles := s4dSobolevWeightHandles(t, weights, D, N)
	fallback, err := EvalProgramOutput(
		fallbackProgram, fallbackHandles, inputs, "output",
	)
	fallbackProgram.Destroy()
	FreeHandles(fallbackHandles)
	if err != nil {
		t.Fatalf("forced MLX fallback forward: %v", err)
	}
	if diff := maxAbsDiffFloat32(primitive, fallback); diff > 3e-5 {
		t.Fatalf("CUDA primitive/fallback forward diff=%g want <=3e-5", diff)
	}
}
