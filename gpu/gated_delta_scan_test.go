//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"math"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func buildGatedDeltaScanParityProgram(chunkSize, batchSize, seqLen, heads, dk, dv, modelDim, classes int) *ir.Program {
	prog := ir.NewProgram(6)
	prog.DeclareInput("x", ir.TensorFloat32, []int{batchSize * seqLen, modelDim})
	prog.DeclareInput("targets", ir.TensorInt32, []int{batchSize * seqLen * heads})
	prog.DeclareOutput("scan", ir.TensorFloat32, []int{batchSize * seqLen * heads, dv})
	prog.DeclareOutput("loss", ir.TensorFloat32, []int{1})

	prog.MatMul("x", "w0", "q_flat")
	prog.Reshape("q_flat", []int{batchSize, seqLen, heads, dk}, "q")

	prog.MatMul("x", "w1", "k_flat")
	prog.Reshape("k_flat", []int{batchSize, seqLen, heads, dk}, "k")

	prog.MatMul("x", "w2", "v_flat")
	prog.Reshape("v_flat", []int{batchSize, seqLen, heads, dv}, "v")

	prog.MatMul("x", "w3", "beta_logits")
	prog.Sigmoid("beta_logits", "beta_flat")
	prog.Reshape("beta_flat", []int{batchSize, seqLen, heads}, "beta")

	prog.MatMul("x", "w4", "gate_logits")
	prog.Softplus("gate_logits", "gate_softplus")
	prog.ScalarMul("gate_softplus", -1.0, "gate_neg")
	prog.Exp("gate_neg", "gate_flat")
	prog.Reshape("gate_flat", []int{batchSize, seqLen, heads}, "gate")

	prog.GatedDeltaScan("q", "k", "v", "beta", "gate", "scan", batchSize, seqLen, heads, dk, dv, chunkSize)
	prog.MatMul("scan", "w5", "logits")
	prog.CrossEntropy("logits", "targets", "loss")
	return prog
}

func makeGatedDeltaScanTestWeights(modelDim, heads, dk, dv, classes int) [][]float32 {
	shapes := [][2]int{
		{modelDim, heads * dk},
		{modelDim, heads * dk},
		{modelDim, heads * dv},
		{modelDim, heads},
		{modelDim, heads},
		{dv, classes},
	}
	weights := make([][]float32, len(shapes))
	for wi, shape := range shapes {
		n := shape[0] * shape[1]
		data := make([]float32, n)
		scale := float32(0.05)
		if wi == 2 || wi == 5 {
			scale = 0.035
		}
		for i := range data {
			angle := float64((wi + 1) * (i%17 + 1))
			data[i] = scale * float32(math.Sin(angle)*0.7+math.Cos(angle*0.37)*0.3)
		}
		weights[wi] = data
	}
	return weights
}

func makeGatedDeltaScanTestInputs(batchSize, seqLen, heads, modelDim, classes int) []TensorInput {
	x := make([]float32, batchSize*seqLen*modelDim)
	for i := range x {
		x[i] = 0.1 * float32(math.Sin(float64(i%29))*0.6+math.Cos(float64(i%11))*0.4)
	}
	targets := make([]int32, batchSize*seqLen*heads)
	for i := range targets {
		targets[i] = int32((3*i + 1) % classes)
	}
	return []TensorInput{
		{Name: "x", DType: TensorFloat32, Shape: []int{batchSize * seqLen, modelDim}, Data: x},
		{Name: "targets", DType: TensorInt32, Shape: []int{batchSize * seqLen * heads}, Data: targets},
	}
}

func makeWeightHandles(t *testing.T, weights [][]float32, modelDim, heads, dk, dv, classes int) []int64 {
	t.Helper()

	shapes := [][2]int{
		{modelDim, heads * dk},
		{modelDim, heads * dk},
		{modelDim, heads * dv},
		{modelDim, heads},
		{modelDim, heads},
		{dv, classes},
	}
	handles := make([]int64, len(weights))
	for i, weight := range weights {
		handle, err := FromData(append([]float32(nil), weight...), shapes[i][0], shapes[i][1])
		if err != nil {
			FreeHandles(handles[:i])
			t.Fatalf("FromData(weight %d): %v", i, err)
		}
		handles[i] = handle
	}
	return handles
}

func makeWeightHandlesForBenchmark(weights [][]float32, modelDim, heads, dk, dv, classes int) ([]int64, error) {
	shapes := [][2]int{
		{modelDim, heads * dk},
		{modelDim, heads * dk},
		{modelDim, heads * dv},
		{modelDim, heads},
		{modelDim, heads},
		{dv, classes},
	}
	handles := make([]int64, len(weights))
	for i, weight := range weights {
		handle, err := FromData(append([]float32(nil), weight...), shapes[i][0], shapes[i][1])
		if err != nil {
			FreeHandles(handles[:i])
			return nil, err
		}
		handles[i] = handle
	}
	return handles, nil
}

func createGatedDeltaScanTrainer(
	t *testing.T,
	chunkSize, batchSize, seqLen, heads, dk, dv, modelDim, classes int,
	weights [][]float32,
) TrainerHandle {
	t.Helper()

	gpuProg, err := LowerIRProgram(buildGatedDeltaScanParityProgram(chunkSize, batchSize, seqLen, heads, dk, dv, modelDim, classes))
	if err != nil {
		t.Fatalf("LowerIRProgram(chunk=%d): %v", chunkSize, err)
	}
	handles := makeWeightHandles(t, weights, modelDim, heads, dk, dv, classes)
	trainer, err := CreateTrainer(gpuProg, handles, TrainerOptimizerSpec{
		Groups: []OptimizerGroup{{
			Kind:        OptimizerAdamW,
			LR:          0.0,
			Beta1:       0.9,
			Beta2:       0.95,
			Epsilon:     1e-8,
			WeightDecay: 0.0,
		}},
		Weights: []WeightOptimizer{
			{GroupIndex: 0, Decay: false},
			{GroupIndex: 0, Decay: false},
			{GroupIndex: 0, Decay: false},
			{GroupIndex: 0, Decay: false},
			{GroupIndex: 0, Decay: false},
			{GroupIndex: 0, Decay: false},
		},
		DefaultBaseLR: 0.0,
	})
	if err != nil {
		gpuProg.Destroy()
		FreeHandles(handles)
		t.Fatalf("CreateTrainer(chunk=%d): %v", chunkSize, err)
	}
	t.Cleanup(func() {
		_ = TrainerFlush(trainer)
		TrainerDestroy(trainer)
		FreeHandles(handles)
		gpuProg.Destroy()
	})
	return trainer
}

func TestGatedDeltaScanChunkedMatchesNaiveForwardAndGrad(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}

	const (
		batchSize = 2
		seqLen    = 97
		heads     = 3
		dk        = 8
		dv        = 12
		modelDim  = 16
		classes   = 7
		tolOut    = 1e-4
		tolGrad   = 1e-6
	)

	weights := makeGatedDeltaScanTestWeights(modelDim, heads, dk, dv, classes)
	inputs := makeGatedDeltaScanTestInputs(batchSize, seqLen, heads, modelDim, classes)

	naiveProg, err := LowerIRProgram(buildGatedDeltaScanParityProgram(0, batchSize, seqLen, heads, dk, dv, modelDim, classes))
	if err != nil {
		t.Fatalf("LowerIRProgram(naive): %v", err)
	}
	defer naiveProg.Destroy()
	chunkedProg, err := LowerIRProgram(buildGatedDeltaScanParityProgram(64, batchSize, seqLen, heads, dk, dv, modelDim, classes))
	if err != nil {
		t.Fatalf("LowerIRProgram(chunked): %v", err)
	}
	defer chunkedProg.Destroy()

	naiveHandles := makeWeightHandles(t, weights, modelDim, heads, dk, dv, classes)
	defer FreeHandles(naiveHandles)
	chunkedHandles := makeWeightHandles(t, weights, modelDim, heads, dk, dv, classes)
	defer FreeHandles(chunkedHandles)

	naiveOut, err := evalProgramOutput(naiveProg, naiveHandles, inputs, "scan")
	if err != nil {
		t.Fatalf("evalProgramOutput(naive): %v", err)
	}
	chunkedOut, err := evalProgramOutput(chunkedProg, chunkedHandles, inputs, "scan")
	if err != nil {
		t.Fatalf("evalProgramOutput(chunked): %v", err)
	}
	forwardDiff := maxAbsDiffFloat32(naiveOut, chunkedOut)
	t.Logf("forward L_inf=%g", forwardDiff)
	if forwardDiff > tolOut {
		t.Fatalf("forward max diff=%g, want <= %g", forwardDiff, tolOut)
	}

	naiveTrainer := createGatedDeltaScanTrainer(t, 0, batchSize, seqLen, heads, dk, dv, modelDim, classes, weights)
	chunkedTrainer := createGatedDeltaScanTrainer(t, 64, batchSize, seqLen, heads, dk, dv, modelDim, classes, weights)

	naiveLoss, err := TrainerComputeMeanSquareGrads(naiveTrainer, inputs, "scan")
	if err != nil {
		t.Fatalf("TrainerComputeMeanSquareGrads(naive): %v", err)
	}
	chunkedLoss, err := TrainerComputeMeanSquareGrads(chunkedTrainer, inputs, "scan")
	if err != nil {
		t.Fatalf("TrainerComputeMeanSquareGrads(chunked): %v", err)
	}
	if diff := math.Abs(float64(naiveLoss - chunkedLoss)); diff > tolGrad {
		t.Fatalf("mean-square grad loss diff=%g, want <= %g", diff, tolGrad)
	}

	var maxGradDiff float32
	for weightIdx := range weights {
		size, err := TrainerWeightSize(naiveTrainer, weightIdx)
		if err != nil {
			t.Fatalf("TrainerWeightSize(%d): %v", weightIdx, err)
		}
		naiveGrad := make([]float32, size)
		if err := TrainerReadGrad(naiveTrainer, weightIdx, naiveGrad); err != nil {
			t.Fatalf("TrainerReadGrad(naive, %d): %v", weightIdx, err)
		}
		chunkedGrad := make([]float32, size)
		if err := TrainerReadGrad(chunkedTrainer, weightIdx, chunkedGrad); err != nil {
			t.Fatalf("TrainerReadGrad(chunked, %d): %v", weightIdx, err)
		}
		diff := maxAbsDiffFloat32(naiveGrad, chunkedGrad)
		if diff > maxGradDiff {
			maxGradDiff = diff
		}
		if diff > tolGrad {
			t.Fatalf("grad weight %d max diff=%g, want <= %g", weightIdx, diff, tolGrad)
		}
	}
	t.Logf("grad L_inf=%g", maxGradDiff)
}

func BenchmarkGatedDeltaScanForward(b *testing.B) {
	if !Available() {
		b.Skip("MLX backend not available")
	}

	const (
		batchSize = 4
		seqLen    = 1024
		heads     = 4
		dk        = 64
		dv        = 128
		modelDim  = 192
		classes   = 11
	)

	weights := makeGatedDeltaScanTestWeights(modelDim, heads, dk, dv, classes)
	inputs := makeGatedDeltaScanTestInputs(batchSize, seqLen, heads, modelDim, classes)

	for _, tc := range []struct {
		name      string
		chunkSize int
	}{
		{name: "naive", chunkSize: 0},
		{name: "chunked", chunkSize: 64},
	} {
		b.Run(tc.name, func(b *testing.B) {
			gpuProg, err := LowerIRProgram(buildGatedDeltaScanParityProgram(tc.chunkSize, batchSize, seqLen, heads, dk, dv, modelDim, classes))
			if err != nil {
				b.Fatalf("LowerIRProgram(chunk=%d): %v", tc.chunkSize, err)
			}
			defer gpuProg.Destroy()
			handles, err := makeWeightHandlesForBenchmark(weights, modelDim, heads, dk, dv, classes)
			if err != nil {
				b.Fatalf("FromData(chunk=%d): %v", tc.chunkSize, err)
			}
			defer FreeHandles(handles)

			if _, err := evalProgramOutput(gpuProg, handles, inputs, "scan"); err != nil {
				b.Fatalf("warmup evalProgramOutput(chunk=%d): %v", tc.chunkSize, err)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := evalProgramOutput(gpuProg, handles, inputs, "scan"); err != nil {
					b.Fatalf("evalProgramOutput(chunk=%d): %v", tc.chunkSize, err)
				}
			}
			b.StopTimer()
			b.ReportMetric(float64(batchSize*seqLen*b.N)/b.Elapsed().Seconds(), "tok/s")
		})
	}
}

func maxAbsDiffFloat32(a, b []float32) float32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var maxDiff float32
	for i := 0; i < n; i++ {
		diff := float32(math.Abs(float64(a[i] - b[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	return maxDiff
}
