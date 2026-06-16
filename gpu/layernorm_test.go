//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"math"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestLayerNormMatchesCPUOracle(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}
	const (
		rows = 2
		dim  = 3
		eps  = 1e-5
	)
	x := []float32{1, 2, 4, -1, 0.5, 3}
	scale := []float32{0.5, 1.5, -2}
	bias := []float32{0.1, -0.2, 0.3}

	prog := ir.NewProgram(2)
	prog.DeclareInput("x", ir.TensorFloat32, []int{rows, dim})
	prog.DeclareOutput("affine", ir.TensorFloat32, []int{rows, dim})
	prog.DeclareOutput("plain", ir.TensorFloat32, []int{rows, dim})
	prog.LayerNorm("x", "w0", "w1", "affine", eps)
	prog.LayerNormNoAffine("x", "plain", eps)

	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()
	scaleHandle, err := FromDataShape(scale, []int{dim})
	if err != nil {
		t.Fatalf("FromDataShape(scale): %v", err)
	}
	biasHandle, err := FromDataShape(bias, []int{dim})
	if err != nil {
		FreeHandle(scaleHandle)
		t.Fatalf("FromDataShape(bias): %v", err)
	}
	defer FreeHandles([]int64{scaleHandle, biasHandle})
	inputs := []TensorInput{{Name: "x", DType: TensorFloat32, Shape: []int{rows, dim}, Data: x}}

	gotAffine, err := EvalProgramOutput(gpuProg, []int64{scaleHandle, biasHandle}, inputs, "affine")
	if err != nil {
		t.Fatalf("EvalProgramOutput(affine): %v", err)
	}
	gotPlain, err := EvalProgramOutput(gpuProg, []int64{scaleHandle, biasHandle}, inputs, "plain")
	if err != nil {
		t.Fatalf("EvalProgramOutput(plain): %v", err)
	}
	wantPlain := cpuLayerNorm(x, nil, nil, rows, dim, eps)
	wantAffine := cpuLayerNorm(x, scale, bias, rows, dim, eps)
	if diff := maxAbsDiffFloat32(gotPlain, wantPlain); diff > 1e-5 {
		t.Fatalf("LayerNormNoAffine L_inf=%g, want <= 1e-5\ngot=%v\nwant=%v", diff, gotPlain, wantPlain)
	}
	if diff := maxAbsDiffFloat32(gotAffine, wantAffine); diff > 1e-5 {
		t.Fatalf("LayerNorm affine L_inf=%g, want <= 1e-5\ngot=%v\nwant=%v", diff, gotAffine, wantAffine)
	}
}

func TestElementwisePrototypeOpsMatchCPUOracle(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}
	x := []float32{0.25, 1.0, 4.0, 9.0}
	y := []float32{-2.0, -0.5, 2.0, 10.0}

	prog := ir.NewProgram(1)
	prog.DeclareInput("x", ir.TensorFloat32, []int{4})
	prog.DeclareInput("y", ir.TensorFloat32, []int{4})
	prog.DeclareOutput("out", ir.TensorFloat32, []int{4})
	prog.Sqrt("x", "sqrt")
	prog.Log("x", "log")
	prog.PowScalar("x", 2, "pow")
	prog.PowScalar("y", 2, "powneg") // negative bases: (-2)^2, (-0.5)^2
	prog.Abs("y", "abs")
	prog.Clamp("y", -1, 1, "clamp")
	prog.Minimum("x", "y", "min")
	prog.Maximum("x", "y", "max")
	prog.GreaterThanScalar("y", 0, "positive")
	prog.Where("positive", "y", "x", "selected")
	prog.Reciprocal("x", "recip")
	prog.Add("sqrt", "log", "out")
	prog.Add("out", "pow", "out")
	prog.Add("out", "abs", "out")
	prog.Add("out", "clamp", "out")
	prog.Add("out", "min", "out")
	prog.Add("out", "max", "out")
	prog.Add("out", "selected", "out")
	prog.Add("out", "powneg", "out")
	prog.Add("out", "recip", "out")

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
		{Name: "x", DType: TensorFloat32, Shape: []int{4}, Data: x},
		{Name: "y", DType: TensorFloat32, Shape: []int{4}, Data: y},
	}, "out")
	if err != nil {
		t.Fatalf("EvalProgramOutput(out): %v", err)
	}
	want := make([]float32, len(x))
	for i := range x {
		selected := x[i]
		if y[i] > 0 {
			selected = y[i]
		}
		want[i] = float32(math.Sqrt(float64(x[i])) +
			math.Log(float64(x[i])) +
			math.Pow(float64(x[i]), 2) +
			math.Abs(float64(y[i])) +
			math.Min(math.Max(float64(y[i]), -1), 1) +
			math.Min(float64(x[i]), float64(y[i])) +
			math.Max(float64(x[i]), float64(y[i])) +
			float64(selected) +
			math.Pow(float64(y[i]), 2) +
			1/float64(x[i]))
	}
	if diff := maxAbsDiffFloat32(got, want); diff > 2e-5 {
		t.Fatalf("elementwise prototype ops L_inf=%g, want <= 2e-5\ngot=%v\nwant=%v", diff, got, want)
	}
}

func cpuLayerNorm(x, scale, bias []float32, rows, dim int, eps float64) []float32 {
	out := make([]float32, len(x))
	for r := 0; r < rows; r++ {
		row := x[r*dim : (r+1)*dim]
		var mean float64
		for _, v := range row {
			mean += float64(v)
		}
		mean /= float64(dim)
		var variance float64
		for _, v := range row {
			diff := float64(v) - mean
			variance += diff * diff
		}
		variance /= float64(dim)
		inv := 1 / math.Sqrt(variance+eps)
		for d, v := range row {
			y := (float64(v) - mean) * inv
			if scale != nil {
				y *= float64(scale[d])
			}
			if bias != nil {
				y += float64(bias[d])
			}
			out[r*dim+d] = float32(y)
		}
	}
	return out
}
