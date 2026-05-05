//go:build mlx

package train

import (
	"math"
	"math/rand"
	"testing"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/gpu"
)

func BenchmarkMamba3SelectiveScanForwardBackward(b *testing.B) {
	benchmarkMamba3SelectiveScanForwardBackward(b, true)
}

func BenchmarkMamba3SelectiveScanForwardBackwardColdCompile(b *testing.B) {
	benchmarkMamba3SelectiveScanForwardBackward(b, false)
}

func benchmarkMamba3SelectiveScanForwardBackward(b *testing.B, warmup bool) {
	if !gpu.Available() {
		b.Skip("MLX backend not available")
	}

	const (
		B = 2
		T = 1024
		D = 128
		N = 16
		G = 4
	)

	prog := arch.NewProgram(7)
	prog.DeclareInput("dummy", arch.TensorFloat32, []int{1})
	prog.DeclareOutput("loss", arch.TensorFloat32, []int{1})
	prog.Mamba3SelectiveScan("w0", "w1", "w2", "w3", "w4", "w5", "w6", "y", B, T, D, N, G)
	prog.MeanAxis("y", 1, "loss_rows")
	prog.MeanAxis("loss_rows", 0, "loss")

	gpuProg, err := gpu.LowerIRProgram(prog)
	if err != nil {
		b.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()

	rng := rand.New(rand.NewSource(20260505))
	x := seededFloats(rng, B*T*D, 0.2)
	dt := seededFloats(rng, B*T*D, 0.05)
	lambdaInput := seededFloats(rng, B*T*D, 0.05)
	theta := seededFloats(rng, B*T*D*(N/2), 0.05)
	aLog := make([]float32, D*N)
	for d := 0; d < D; d++ {
		for n := 0; n < N; n++ {
			aLog[d*N+n] = float32(math.Log(float64(n+1))) - 2.0
		}
	}
	bProj := seededFloats(rng, B*T*G*N, 0.1)
	cProj := seededFloats(rng, B*T*G*N, 0.1)

	weights := [][]float32{x, dt, lambdaInput, theta, aLog, bProj, cProj}
	shapes := [][2]int{
		{B * T, D},
		{B * T, D},
		{B * T, D},
		{B * T, D * (N / 2)},
		{D, N},
		{B * T, G * N},
		{B * T, G * N},
	}

	handles := make([]int64, len(weights))
	for i := range weights {
		handles[i], err = gpu.FromData(weights[i], shapes[i][0], shapes[i][1])
		if err != nil {
			b.Fatalf("FromData(%d): %v", i, err)
		}
		defer gpu.FreeHandle(handles[i])
	}

	inputs := []gpu.TensorInput{{Name: "dummy", DType: gpu.TensorFloat32, Shape: []int{1}, Data: []float32{0}}}
	if warmup {
		if _, _, err := gpu.EvalProgramGradientsForOutput(gpuProg, handles, inputs, "loss"); err != nil {
			b.Fatalf("warmup EvalProgramGradientsForOutput: %v", err)
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := gpu.EvalProgramGradientsForOutput(gpuProg, handles, inputs, "loss"); err != nil {
			b.Fatalf("EvalProgramGradientsForOutput iteration %d: %v", i, err)
		}
	}
}
