//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestCharFeatureBagMatchesCPUOracle(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}
	const (
		B = 2
		T = 3
		K = 4
		D = 3
		V = 8
	)
	table := patternedFloats(V*D, 0.19)
	ids := []int32{
		0, 1, 2, 2,
		3, 0, 0, 4,
		5, 6, 0, 0,
		7, 1, 0, 0,
		2, 3, 4, 5,
		0, 0, 0, 0,
	}

	prog := ir.NewProgram(1)
	prog.DeclareInput("ids", ir.TensorInt32, []int{B, T, K})
	prog.DeclareOutput("out", ir.TensorFloat32, []int{B * T, D})
	prog.CharFeatureBag("w0", "ids", "out", B, T, K, D)

	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()
	tableHandle, err := FromData(table, V, D)
	if err != nil {
		t.Fatalf("FromData(table): %v", err)
	}
	defer FreeHandle(tableHandle)

	got, err := EvalProgramOutput(gpuProg, []int64{tableHandle}, []TensorInput{
		{Name: "ids", DType: TensorInt32, Shape: []int{B, T, K}, Data: ids},
	}, "out")
	if err != nil {
		t.Fatalf("EvalProgramOutput: %v", err)
	}
	want := cpuCharFeatureBag(table, ids, B, T, K, D)
	if diff := maxAbsDiffFloat32(got, want); diff > 1e-6 {
		t.Fatalf("CharFeatureBag L_inf=%g, want <= 1e-6\ngot=%v\nwant=%v", diff, got, want)
	}
}

func cpuCharFeatureBag(table []float32, ids []int32, B, T, K, D int) []float32 {
	out := make([]float32, B*T*D)
	for bt := 0; bt < B*T; bt++ {
		for k := 0; k < K; k++ {
			id := int(ids[bt*K+k])
			if id <= 0 {
				continue
			}
			for d := 0; d < D; d++ {
				out[bt*D+d] += table[id*D+d]
			}
		}
	}
	return out
}
