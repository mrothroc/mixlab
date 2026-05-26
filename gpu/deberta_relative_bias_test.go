//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestDebertaRelativeBiasMatchesCPUOracle(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}
	const (
		B      = 2
		T      = 4
		H      = 2
		D      = 3
		window = 2
	)
	R := 2 * window
	q := patternedFloats(B*H*T*D, 0.11)
	k := patternedFloats(B*H*T*D, 0.07)
	posKey := patternedFloats(H*R*D, 0.13)
	posQuery := patternedFloats(H*R*D, 0.17)

	prog := ir.NewProgram(1)
	prog.DeclareInput("q", ir.TensorFloat32, []int{B, H, T, D})
	prog.DeclareInput("k", ir.TensorFloat32, []int{B, H, T, D})
	prog.DeclareInput("pk", ir.TensorFloat32, []int{H, R, D})
	prog.DeclareInput("pq", ir.TensorFloat32, []int{H, R, D})
	prog.DeclareOutput("bias", ir.TensorFloat32, []int{B, H, T, T})
	prog.DebertaRelativeBias("q", "k", "pk", "pq", "bias", B, T, H, D, window)

	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()
	dummy, err := FromData([]float32{0}, 1, 1)
	if err != nil {
		t.Fatalf("FromData(dummy): %v", err)
	}
	defer FreeHandle(dummy)

	got, err := EvalProgramOutput(gpuProg, []int64{dummy}, []TensorInput{
		{Name: "q", DType: TensorFloat32, Shape: []int{B, H, T, D}, Data: q},
		{Name: "k", DType: TensorFloat32, Shape: []int{B, H, T, D}, Data: k},
		{Name: "pk", DType: TensorFloat32, Shape: []int{H, R, D}, Data: posKey},
		{Name: "pq", DType: TensorFloat32, Shape: []int{H, R, D}, Data: posQuery},
	}, "bias")
	if err != nil {
		t.Fatalf("EvalProgramOutput: %v", err)
	}
	want := cpuDebertaRelativeBias(q, k, posKey, posQuery, B, T, H, D, window)
	if diff := maxAbsDiffFloat32(got, want); diff > 1e-5 {
		t.Fatalf("DebertaRelativeBias L_inf=%g, want <= 1e-5\ngot=%v\nwant=%v", diff, got, want)
	}
}

func cpuDebertaRelativeBias(q, k, posKey, posQuery []float32, B, T, H, D, window int) []float32 {
	R := 2 * window
	out := make([]float32, B*H*T*T)
	qIdx := func(b, h, t, d int) int { return (((b*H+h)*T+t)*D + d) }
	pIdx := func(h, r, d int) int { return ((h*R+r)*D + d) }
	oIdx := func(b, h, i, j int) int { return (((b*H+h)*T+i)*T + j) }
	clip := func(v int) int {
		if v < 0 {
			return 0
		}
		if v >= R {
			return R - 1
		}
		return v
	}
	for b := 0; b < B; b++ {
		for h := 0; h < H; h++ {
			for i := 0; i < T; i++ {
				for j := 0; j < T; j++ {
					c2p := clip(i - j + window)
					p2c := clip(j - i + window)
					var sum float32
					for d := 0; d < D; d++ {
						sum += q[qIdx(b, h, i, d)]*posKey[pIdx(h, c2p, d)] +
							k[qIdx(b, h, j, d)]*posQuery[pIdx(h, p2c, d)]
					}
					out[oIdx(b, h, i, j)] = sum
				}
			}
		}
	}
	return out
}
