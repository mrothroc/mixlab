//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"math"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

const (
	randomNormalRows = 1024
	randomNormalCols = 1024
)

func TestRandomNormalProducesNonDegenerateSamples(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}

	out := randomNormalEvalOutput(t, buildRandomNormalEvalProgram(0.0, 1.0, "out"), "out")
	mean, stddev := randomNormalStats(out)
	t.Logf("standard normal stats: mean=%0.6f stddev=%0.6f", mean, stddev)

	if mean < -0.05 || mean > 0.05 {
		t.Fatalf("mean=%0.6f, want in [-0.05, 0.05]", mean)
	}
	if stddev < 0.95 || stddev > 1.05 {
		t.Fatalf("stddev=%0.6f, want in [0.95, 1.05]", stddev)
	}
	if randomNormalAllZero(out) {
		t.Fatal("random normal output is all zeros")
	}
}

func TestRandomNormalSuccessiveOpsProduceDistinctSamples(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}

	prog := ir.NewProgram(1)
	prog.DeclareInput("dummy", ir.TensorFloat32, []int{1})
	prog.DeclareOutput("combined", ir.TensorFloat32, []int{randomNormalRows * 2, randomNormalCols})
	prog.RandomNormal([]int{randomNormalRows, randomNormalCols}, 0.0, 1.0, "a")
	prog.RandomNormal([]int{randomNormalRows, randomNormalCols}, 0.0, 1.0, "b")
	prog.Concat("a", "b", 0, "combined")

	combined := randomNormalEvalOutput(t, prog, "combined")
	split := randomNormalRows * randomNormalCols
	a := combined[:split]
	b := combined[split:]

	equalCount := 0
	for i := range a {
		if a[i] == b[i] {
			equalCount++
		}
	}
	t.Logf("equal entries across paired samples: %d / %d", equalCount, len(a))

	if randomNormalAllClose(a, b, 1e-5, 1e-8) {
		t.Fatal("paired RandomNormal outputs were unexpectedly allclose")
	}
	if equalCount >= len(a)/100 {
		t.Fatalf("paired RandomNormal outputs matched too often: %d / %d", equalCount, len(a))
	}
}

func TestRandomNormalHonorsMeanAndStddev(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}

	out := randomNormalEvalOutput(t, buildRandomNormalEvalProgram(5.0, 2.0, "out"), "out")
	mean, stddev := randomNormalStats(out)
	t.Logf("shifted normal stats: mean=%0.6f stddev=%0.6f", mean, stddev)

	if mean < 4.9 || mean > 5.1 {
		t.Fatalf("mean=%0.6f, want in [4.9, 5.1]", mean)
	}
	if stddev < 1.9 || stddev > 2.1 {
		t.Fatalf("stddev=%0.6f, want in [1.9, 2.1]", stddev)
	}
}

func buildRandomNormalEvalProgram(mean, stddev float32, output string) *ir.Program {
	prog := ir.NewProgram(1)
	prog.DeclareInput("dummy", ir.TensorFloat32, []int{1})
	prog.DeclareOutput(output, ir.TensorFloat32, []int{randomNormalRows, randomNormalCols})
	prog.RandomNormal([]int{randomNormalRows, randomNormalCols}, mean, stddev, output)
	return prog
}

func randomNormalEvalOutput(t *testing.T, prog *ir.Program, outputName string) []float32 {
	t.Helper()

	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()

	dummyWeight := randomNormalDummyWeight(t)
	defer FreeHandle(dummyWeight)

	out, err := evalProgramOutput(
		gpuProg,
		[]int64{dummyWeight},
		[]TensorInput{{Name: "dummy", DType: TensorFloat32, Shape: []int{1}, Data: []float32{0}}},
		outputName,
	)
	if err != nil {
		t.Fatalf("evalProgramOutput(%s): %v", outputName, err)
	}
	return out
}

func randomNormalDummyWeight(t *testing.T) int64 {
	t.Helper()

	handle, err := FromData([]float32{0}, 1, 1)
	if err != nil {
		t.Fatalf("FromData dummy weight: %v", err)
	}
	return handle
}

func randomNormalStats(data []float32) (float64, float64) {
	sum := 0.0
	for _, v := range data {
		sum += float64(v)
	}
	mean := sum / float64(len(data))

	variance := 0.0
	for _, v := range data {
		diff := float64(v) - mean
		variance += diff * diff
	}
	variance /= float64(len(data))
	return mean, math.Sqrt(variance)
}

func randomNormalAllZero(data []float32) bool {
	for _, v := range data {
		if v != 0 {
			return false
		}
	}
	return true
}

func randomNormalAllClose(a, b []float32, rtol, atol float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		diff := math.Abs(float64(a[i] - b[i]))
		limit := atol + rtol*math.Abs(float64(b[i]))
		if diff > limit {
			return false
		}
	}
	return true
}
