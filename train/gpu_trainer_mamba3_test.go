//go:build mlx

package train

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/gpu"
)

func TestMamba3SelectiveScanGrad(t *testing.T) {
	if !gpu.Available() {
		t.Skip("MLX backend not available")
	}

	for _, groups := range []int{1, 2} {
		for _, chunkSize := range []int{0, 3} {
			t.Run(fmt.Sprintf("G%d_chunk%d", groups, chunkSize), func(t *testing.T) {
				testMamba3SelectiveScanGrad(t, groups, chunkSize)
			})
		}
	}
}

func TestMamba3SelectiveScanGradChannelChunked(t *testing.T) {
	if !gpu.Available() {
		t.Skip("MLX backend not available")
	}
	t.Setenv("MIXLAB_MAMBA3_CHANNEL_CHUNK", "2")

	for _, groups := range []int{1, 2} {
		t.Run(fmt.Sprintf("G%d", groups), func(t *testing.T) {
			testMamba3SelectiveScanGrad(t, groups, 3)
		})
	}
}

func TestMamba3SelectiveScanGradForwardV2(t *testing.T) {
	if !gpu.Available() {
		t.Skip("MLX backend not available")
	}
	t.Setenv("MIXLAB_MAMBA3_SCAN_FWD", "v2")
	t.Setenv("MIXLAB_MAMBA3_SCAN_FWD_CHUNK", "3")

	testMamba3SelectiveScanGrad(t, 2, 3)
}

func TestMamba3SelectiveScanGradForwardV3(t *testing.T) {
	if !gpu.Available() {
		t.Skip("MLX backend not available")
	}
	t.Setenv("MIXLAB_MAMBA3_SCAN_FWD", "v3")
	t.Setenv("MIXLAB_MAMBA3_SCAN_FWD_CHUNK", "3")

	testMamba3SelectiveScanGrad(t, 2, 3)
}

func TestMamba3SelectiveScanGradBackwardV2(t *testing.T) {
	if !gpu.Available() {
		t.Skip("MLX backend not available")
	}
	t.Setenv("MIXLAB_MAMBA3_SCAN_BWD", "v2")

	testMamba3SelectiveScanGrad(t, 2, 3)
}

func TestMamba3SelectiveScanGradForwardBackwardV2(t *testing.T) {
	if !gpu.Available() {
		t.Skip("MLX backend not available")
	}
	t.Setenv("MIXLAB_MAMBA3_SCAN_FWD", "v2")
	t.Setenv("MIXLAB_MAMBA3_SCAN_FWD_CHUNK", "3")
	t.Setenv("MIXLAB_MAMBA3_SCAN_BWD", "v2")

	testMamba3SelectiveScanGrad(t, 2, 3)
}

func TestMamba3SelectiveScanGradForwardV3BackwardV2(t *testing.T) {
	if !gpu.Available() {
		t.Skip("MLX backend not available")
	}
	t.Setenv("MIXLAB_MAMBA3_SCAN_FWD", "v3")
	t.Setenv("MIXLAB_MAMBA3_SCAN_FWD_CHUNK", "3")
	t.Setenv("MIXLAB_MAMBA3_SCAN_BWD", "v2")

	testMamba3SelectiveScanGrad(t, 2, 3)
}

func TestDepthwiseConv1DGrad(t *testing.T) {
	if !gpu.Available() {
		t.Skip("MLX backend not available")
	}

	const (
		B = 2
		T = 5
		D = 3
		K = 4
	)
	prog := arch.NewProgram(2)
	prog.DeclareInput("dummy", arch.TensorFloat32, []int{1})
	prog.DeclareOutput("loss", arch.TensorFloat32, []int{1})
	prog.DepthwiseConv1D("w0", "w1", "y", B, T, D, K)
	prog.MeanAxis("y", 1, "loss_rows")
	prog.MeanAxis("loss_rows", 0, "loss")

	gpuProg, err := gpu.LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()

	rng := rand.New(rand.NewSource(5678))
	x := seededFloats(rng, B*T*D, 0.4)
	w := seededFloats(rng, D*K, 0.3)
	handles := make([]int64, 2)
	handles[0], err = gpu.FromData(x, B*T, D)
	if err != nil {
		t.Fatalf("FromData(x): %v", err)
	}
	defer gpu.FreeHandle(handles[0])
	handles[1], err = gpu.FromData(w, D, K)
	if err != nil {
		t.Fatalf("FromData(w): %v", err)
	}
	defer gpu.FreeHandle(handles[1])

	loss, gotGrads, err := gpu.EvalProgramGradientsForOutput(
		gpuProg,
		handles,
		[]gpu.TensorInput{{Name: "dummy", DType: gpu.TensorFloat32, Shape: []int{1}, Data: []float32{0}}},
		"loss",
	)
	if err != nil {
		t.Fatalf("EvalProgramGradientsForOutput: %v", err)
	}
	wantLoss, wantGradX, wantGradW := depthwiseConv1DCPUForwardBackward(x, w, B, T, D, K)
	if diff := math.Abs(float64(loss - wantLoss)); diff > 1e-6 {
		t.Fatalf("loss mismatch: mlx=%g cpu=%g diff=%g", loss, wantLoss, diff)
	}
	if maxRel, maxAbs := maxGradientError(gotGrads[0], wantGradX); maxRel > 1e-5 && maxAbs > 1e-6 {
		t.Fatalf("dL/dx mismatch: max_relative_error=%g max_absolute_error=%g", maxRel, maxAbs)
	}
	if maxRel, maxAbs := maxGradientError(gotGrads[1], wantGradW); maxRel > 1e-5 && maxAbs > 1e-6 {
		t.Fatalf("dL/dw mismatch: max_relative_error=%g max_absolute_error=%g", maxRel, maxAbs)
	}
}

func TestMamba3CanonicalBlockGradMatchesExpanded(t *testing.T) {
	if !gpu.Available() {
		t.Skip("MLX backend not available")
	}

	const (
		B          = 2
		T          = 4
		D          = 8
		inner      = 8
		stateSize  = 4
		nGroups    = 2
		dtRank     = 2
		convKernel = 3
		scanChunk  = 3
	)

	expanded := arch.NewProgram(20)
	expanded.DeclareInput("x_in", arch.TensorFloat32, []int{B * T, D})
	expanded.DeclareOutput("loss", arch.TensorFloat32, []int{1})
	emitExpandedMamba3CanonicalTestIR(expanded, B, T, D, inner, stateSize, nGroups, convKernel, scanChunk)

	fused := arch.NewProgram(20)
	fused.DeclareInput("x_in", arch.TensorFloat32, []int{B * T, D})
	fused.DeclareOutput("loss", arch.TensorFloat32, []int{1})
	blockInputs := []string{"x_in"}
	for i := 0; i < 20; i++ {
		blockInputs = append(blockInputs, fmt.Sprintf("w%d", i))
	}
	fused.Mamba3CanonicalBlock(blockInputs, "out", B, T, true, scanChunk)
	addMeanLoss(fused, "out")

	expandedGPU, err := gpu.LowerIRProgram(expanded)
	if err != nil {
		t.Fatalf("LowerIRProgram(expanded): %v", err)
	}
	defer expandedGPU.Destroy()
	fusedGPU, err := gpu.LowerIRProgram(fused)
	if err != nil {
		t.Fatalf("LowerIRProgram(fused): %v", err)
	}
	defer fusedGPU.Destroy()

	rng := rand.New(rand.NewSource(20260508))
	xInput := seededFloats(rng, B*T*D, 0.2)
	weights, shapes := seededCanonicalBlockWeights(rng, D, inner, stateSize, nGroups, dtRank, convKernel)
	handles := make([]int64, len(weights))
	for i := range weights {
		handles[i], err = gpu.FromData(weights[i], shapes[i][0], shapes[i][1])
		if err != nil {
			t.Fatalf("FromData(%d): %v", i, err)
		}
		defer gpu.FreeHandle(handles[i])
	}
	inputs := []gpu.TensorInput{{Name: "x_in", DType: gpu.TensorFloat32, Shape: []int{B * T, D}, Data: xInput}}
	expandedLoss, expandedGrads, err := gpu.EvalProgramGradientsForOutput(expandedGPU, handles, inputs, "loss")
	if err != nil {
		t.Fatalf("expanded EvalProgramGradientsForOutput: %v", err)
	}
	fusedLoss, fusedGrads, err := gpu.EvalProgramGradientsForOutput(fusedGPU, handles, inputs, "loss")
	if err != nil {
		t.Fatalf("fused EvalProgramGradientsForOutput: %v", err)
	}
	if diff := math.Abs(float64(expandedLoss - fusedLoss)); diff > 2e-5 {
		t.Fatalf("loss mismatch: expanded=%g fused=%g diff=%g", expandedLoss, fusedLoss, diff)
	}
	if len(expandedGrads) != len(fusedGrads) {
		t.Fatalf("gradient count mismatch: expanded=%d fused=%d", len(expandedGrads), len(fusedGrads))
	}
	for i := range expandedGrads {
		maxRel, maxAbs := maxGradientError(fusedGrads[i], expandedGrads[i])
		t.Logf("w%d max_relative_error=%g max_absolute_error=%g", i, maxRel, maxAbs)
		if maxRel > 5e-3 && maxAbs > 5e-5 {
			t.Fatalf("w%d gradient mismatch: max_relative_error=%g max_absolute_error=%g", i, maxRel, maxAbs)
		}
	}
}

func testMamba3SelectiveScanGrad(t *testing.T, groups, chunkSize int) {
	const (
		B = 2
		T = 8
		D = 4
		N = 8
	)
	if D%groups != 0 {
		t.Fatalf("test requires D divisible by groups: D=%d G=%d", D, groups)
	}

	prog := arch.NewProgram(7)
	prog.DeclareInput("dummy", arch.TensorFloat32, []int{1})
	prog.DeclareOutput("loss", arch.TensorFloat32, []int{1})
	prog.Mamba3SelectiveScanChunked("w0", "w1", "w2", "w3", "w4", "w5", "w6", "y", B, T, D, N, groups, chunkSize)
	prog.MeanAxis("y", 1, "loss_rows")
	prog.MeanAxis("loss_rows", 0, "loss")

	gpuProg, err := gpu.LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()

	rng := rand.New(rand.NewSource(1234))
	x := seededFloats(rng, B*T*D, 0.35)
	dt := seededFloats(rng, B*T*D, 0.2)
	lambdaInput := seededFloats(rng, B*T*D, 0.2)
	theta := seededFloats(rng, B*T*D*(N/2), 0.5)
	aLog := make([]float32, D*N)
	for d := 0; d < D; d++ {
		for n := 0; n < N; n++ {
			aLog[d*N+n] = float32(math.Log(float64(n+1))) - 1.5
		}
	}
	bProj := seededFloats(rng, B*T*groups*N, 0.25)
	cProj := seededFloats(rng, B*T*groups*N, 0.25)
	weights := [][]float32{x, dt, lambdaInput, theta, aLog, bProj, cProj}
	shapes := [][2]int{{B * T, D}, {B * T, D}, {B * T, D}, {B * T, D * (N / 2)}, {D, N}, {B * T, groups * N}, {B * T, groups * N}}

	handles := make([]int64, len(weights))
	for i := range weights {
		handles[i], err = gpu.FromData(weights[i], shapes[i][0], shapes[i][1])
		if err != nil {
			t.Fatalf("FromData(%d): %v", i, err)
		}
		defer gpu.FreeHandle(handles[i])
	}

	loss, gotGrads, err := gpu.EvalProgramGradientsForOutput(
		gpuProg,
		handles,
		[]gpu.TensorInput{{Name: "dummy", DType: gpu.TensorFloat32, Shape: []int{1}, Data: []float32{0}}},
		"loss",
	)
	if err != nil {
		t.Fatalf("EvalProgramGradientsForOutput: %v", err)
	}
	wantLoss, wantGrads := mamba3MIMOCPUForwardBackward(x, dt, lambdaInput, theta, aLog, bProj, cProj, B, T, D, N, groups)
	if diff := math.Abs(float64(loss - wantLoss)); diff > 2e-5 {
		t.Fatalf("loss mismatch: mlx=%g cpu=%g diff=%g", loss, wantLoss, diff)
	}

	names := []string{"dL/dx", "dL/dDeltaRaw", "dL/dLambdaInput", "dL/dTheta", "dL/dA_log", "dL/dB", "dL/dC"}
	for i, name := range names {
		maxRel, maxAbs := maxGradientError(gotGrads[i], wantGrads[i])
		t.Logf("%s max_relative_error=%g max_absolute_error=%g", name, maxRel, maxAbs)
		if maxRel > 3e-3 && maxAbs > 3e-5 {
			t.Fatalf("%s gradient mismatch: max_relative_error=%g max_absolute_error=%g", name, maxRel, maxAbs)
		}
	}

	if groups == 1 {
		thetaZero := make([]float32, len(theta))
		phase4Loss, phase4Grads := mamba3MIMOCPUForwardBackward(x, dt, lambdaInput, thetaZero, aLog, bProj, cProj, B, T, D, N, groups)
		phase2Loss, phase2Grads := mamba3SISOCPUForwardBackwardPhase2(x, dt, lambdaInput, aLog, bProj, cProj, B, T, D, N)
		if diff := math.Abs(float64(phase4Loss - phase2Loss)); diff > 1e-7 {
			t.Fatalf("G=1 theta-zero loss mismatch with Phase 2: phase4=%g phase2=%g diff=%g", phase4Loss, phase2Loss, diff)
		}
		phase2Names := []string{"dL/dx", "dL/dDeltaRaw", "dL/dLambdaInput", "dL/dA_log", "dL/dB", "dL/dC"}
		phase4Idx := []int{0, 1, 2, 4, 5, 6}
		for i, name := range phase2Names {
			maxRel, maxAbs := maxGradientError(phase4Grads[phase4Idx[i]], phase2Grads[i])
			t.Logf("G=1 theta-zero %s max_relative_error=%g max_absolute_error=%g", name, maxRel, maxAbs)
			if maxRel > 1e-7 && maxAbs > 1e-7 {
				t.Fatalf("G=1 theta-zero %s mismatch with Phase 2: max_relative_error=%g max_absolute_error=%g", name, maxRel, maxAbs)
			}
		}
	}
}

func TestMamba3CanonicalSmokeLossDecreases(t *testing.T) {
	if !gpu.Available() {
		t.Skip("MLX backend not available")
	}
	useConv := true
	cfg := &ArchConfig{
		Name:          "mamba3_canonical_smoketest",
		ModelDim:      16,
		VocabSize:     32,
		SeqLen:        8,
		TieEmbeddings: false,
		Blocks: []BlockSpec{{
			Type:       "mamba3-canonical",
			InnerDim:   16,
			StateSize:  8,
			NGroups:    2,
			DTRank:     2,
			ConvKernel: 4,
			UseConv:    &useConv,
		}},
		Training: TrainingSpec{
			Optimizer:         "adamw",
			Steps:             10,
			LR:                3e-3,
			EmbedLR:           3e-3,
			MatrixLR:          3e-3,
			ScalarLR:          3e-3,
			HeadLR:            3e-3,
			Beta1:             0.9,
			Beta2:             0.95,
			Epsilon:           1e-8,
			BatchTokens:       16,
			Seed:              7,
			GradClip:          1,
			MuonMomentum:      0.9,
			MuonBackendSteps:  5,
			EmbedWeightDecay:  0,
			MatrixWeightDecay: 0,
			ScalarWeightDecay: 0,
			HeadWeightDecay:   0,
		},
	}
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}
	trainer, err := initGPUTrainer(prog, cfg, nil, nil)
	if err != nil {
		t.Fatalf("initGPUTrainer: %v", err)
	}
	defer trainer.CloseTrainer()

	xTok := []int{0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8}
	yTok := []int{1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 4, 5, 6, 7, 8, 9}
	weightReader, ok := trainer.(interface {
		ReadWeights() ([][]float32, error)
	})
	if !ok {
		t.Fatalf("trainer does not expose ReadWeights")
	}
	beforeWeights, err := weightReader.ReadWeights()
	if err != nil {
		t.Fatalf("ReadWeights before training: %v", err)
	}
	losses := make([]float32, 10)
	for step := range losses {
		if err := trainer.SubmitStepGPU(xTok, yTok, 2, cfg.SeqLen, float32(cfg.Training.LR)); err != nil {
			t.Fatalf("SubmitStepGPU step %d: %v", step, err)
		}
		loss, err := trainer.CollectLossGPU()
		if err != nil {
			t.Fatalf("CollectLossGPU step %d: %v", step, err)
		}
		if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
			t.Fatalf("step %d non-finite loss %g", step, loss)
		}
		losses[step] = loss
	}
	t.Logf("smoke loss curve: %s", formatLosses(losses))
	if losses[len(losses)-1] >= losses[0] {
		t.Fatalf("loss did not decrease: first=%g last=%g curve=%s", losses[0], losses[len(losses)-1], formatLosses(losses))
	}
	afterWeights, err := weightReader.ReadWeights()
	if err != nil {
		t.Fatalf("ReadWeights after training: %v", err)
	}
	maxWeightDelta := float64(0)
	for i := range beforeWeights {
		for j := range beforeWeights[i] {
			if delta := math.Abs(float64(afterWeights[i][j] - beforeWeights[i][j])); delta > maxWeightDelta {
				maxWeightDelta = delta
			}
		}
	}
	t.Logf("smoke max weight delta after multi-step training: %g", maxWeightDelta)
	if maxWeightDelta == 0 {
		t.Fatalf("weights did not update across compiled multi-step training")
	}
}

func emitExpandedMamba3CanonicalTestIR(prog *arch.Program, B, T, D, inner, stateSize, nGroups, convKernel, scanChunk int) {
	prog.RMSNorm("x_in", "w0", "x_norm", 1e-5)
	prog.MatMul("x_norm", "w1", "x_proj")
	prog.DepthwiseConv1D("x_proj", "w2", "x_conv", B, T, inner, convKernel)
	prog.MatMul("x_conv", "w3", "dt_low")
	prog.MatMul("dt_low", "w4", "dt_raw")
	prog.Add("dt_raw", "w16", "dt")
	prog.MatMul("x_conv", "w5", "lambda_low")
	prog.MatMul("lambda_low", "w6", "lambda")
	prog.MatMul("x_conv", "w7", "theta_low")
	prog.MatMul("theta_low", "w8", "theta")
	prog.MatMul("x_conv", "w9", "b_proj")
	prog.Reshape("b_proj", []int{B * T * nGroups, stateSize}, "b_group")
	prog.RMSNorm("b_group", "w11", "b_norm", 1e-5)
	prog.Reshape("b_norm", []int{B * T, nGroups * stateSize}, "b_flat")
	prog.Add("b_flat", "w13", "b_biased")
	prog.MatMul("x_conv", "w10", "c_proj")
	prog.Reshape("c_proj", []int{B * T * nGroups, stateSize}, "c_group")
	prog.RMSNorm("c_group", "w12", "c_norm", 1e-5)
	prog.Reshape("c_norm", []int{B * T, nGroups * stateSize}, "c_flat")
	prog.Add("c_flat", "w14", "c_biased")
	prog.Mamba3SelectiveScanChunked("x_conv", "dt", "lambda", "theta", "w15", "b_biased", "c_biased", "y", B, T, inner, stateSize, nGroups, scanChunk)
	prog.RMSNorm("y", "w17", "y_norm", 1e-5)
	prog.MatMul("x_norm", "w18", "z")
	prog.SiLU("z", "z_act")
	prog.Mul("y_norm", "z_act", "y_gated")
	prog.MatMul("y_gated", "w19", "out_proj")
	prog.Add("x_in", "out_proj", "out")
	addMeanLoss(prog, "out")
}

func addMeanLoss(prog *arch.Program, output string) {
	prog.MeanAxis(output, 1, "loss_rows")
	prog.MeanAxis("loss_rows", 0, "loss")
}

func seededCanonicalBlockWeights(rng *rand.Rand, D, inner, stateSize, nGroups, dtRank, convKernel int) ([][]float32, [][2]int) {
	groupState := nGroups * stateSize
	shapes := [][2]int{
		{1, D},
		{D, inner},
		{inner, convKernel},
		{inner, dtRank},
		{dtRank, inner},
		{inner, dtRank},
		{dtRank, inner},
		{inner, dtRank},
		{dtRank, inner * (stateSize / 2)},
		{inner, groupState},
		{inner, groupState},
		{1, stateSize},
		{1, stateSize},
		{1, groupState},
		{1, groupState},
		{inner, stateSize},
		{1, inner},
		{1, inner},
		{D, inner},
		{inner, D},
	}
	weights := make([][]float32, len(shapes))
	for i, shape := range shapes {
		weights[i] = seededFloats(rng, shape[0]*shape[1], 0.08)
	}
	for _, idx := range []int{0, 11, 12, 17} {
		for j := range weights[idx] {
			weights[idx][j] = 1 + weights[idx][j]
		}
	}
	for _, idx := range []int{13, 14} {
		for j := range weights[idx] {
			weights[idx][j] = 1 + weights[idx][j]
		}
	}
	for d := 0; d < inner; d++ {
		for n := 0; n < stateSize; n++ {
			weights[15][d*stateSize+n] = float32(math.Log(float64(n+1))) - 2.0
		}
	}
	return weights, shapes
}

func seededFloats(rng *rand.Rand, n int, scale float64) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32((rng.Float64()*2 - 1) * scale)
	}
	return out
}

func depthwiseConv1DCPUForwardBackward(x, w []float32, B, T, D, K int) (float32, []float32, []float32) {
	denom := float32(B * T * D)
	gradX := make([]float32, len(x))
	gradW := make([]float32, len(w))
	var loss float32
	for b := 0; b < B; b++ {
		for t := 0; t < T; t++ {
			for d := 0; d < D; d++ {
				var y float32
				for k := 0; k < K; k++ {
					srcT := t - k
					if srcT < 0 {
						continue
					}
					xIdx := (b*T+srcT)*D + d
					wIdx := d*K + k
					y += x[xIdx] * w[wIdx]
					gradX[xIdx] += w[wIdx] / denom
					gradW[wIdx] += x[xIdx] / denom
				}
				loss += y / denom
			}
		}
	}
	return loss, gradX, gradW
}

func mamba3MIMOCPUForwardBackward(x, dtRaw, lambdaInput, theta, aLog, bProj, cProj []float32, B, T, D, N, G int) (float32, [][]float32) {
	K := N / 2
	channelsPerGroup := D / G
	totalY := float64(0)
	hBefore := make([]float64, B*T*D*N)
	hAfter := make([]float64, B*T*D*N)
	alpha := make([]float64, B*T*D*N)
	delta := make([]float64, B*T*D)
	lambda := make([]float64, B*T*D)
	phi := make([]float64, B*T*D*K)
	bRot := make([]float64, B*T*D*N)
	cRot := make([]float64, B*T*D*N)
	A := make([]float64, D*N)
	h := make([]float64, B*D*N)
	for d := 0; d < D; d++ {
		for n := 0; n < N; n++ {
			A[d*N+n] = -math.Exp(float64(aLog[d*N+n]))
		}
	}
	for b := 0; b < B; b++ {
		for t := 0; t < T; t++ {
			for d := 0; d < D; d++ {
				idx := ((b*T + t) * D) + d
				delta[idx] = scanSoftplus64(float64(dtRaw[idx]))
				lambda[idx] = sigmoid64(float64(lambdaInput[idx]))
				for k := 0; k < K; k++ {
					thetaIdx := (((b*T+t)*D)+d)*K + k
					prevPhi := float64(0)
					if t > 0 {
						prevPhi = phi[(((b*T+t-1)*D)+d)*K+k]
					}
					angle := prevPhi + delta[idx]*float64(theta[thetaIdx])
					phi[thetaIdx] = angle
					cosv := math.Cos(angle)
					sinv := math.Sin(angle)
					g := d / channelsPerGroup
					groupBase := ((b*T+t)*G + g) * N
					b0 := float64(bProj[groupBase+2*k])
					b1 := float64(bProj[groupBase+2*k+1])
					c0 := float64(cProj[groupBase+2*k])
					c1 := float64(cProj[groupBase+2*k+1])
					full := (((b*T+t)*D)+d)*N + 2*k
					// Prop. 4's transformed scalar SSM applies cumulative R^T to B/C.
					bRot[full] = cosv*b0 + sinv*b1
					bRot[full+1] = -sinv*b0 + cosv*b1
					cRot[full] = cosv*c0 + sinv*c1
					cRot[full+1] = -sinv*c0 + cosv*c1
				}
			}
			for d := 0; d < D; d++ {
				idx := (b*T+t)*D + d
				xt := float64(x[idx])
				dt := delta[idx]
				lam := lambda[idx]
				for n := 0; n < N; n++ {
					state := (b*D+d)*N + n
					full := (((b*T+t)*D)+d)*N + n
					hBefore[full] = h[state]
					a := math.Exp(dt * A[d*N+n])
					alpha[full] = a
					previous := float64(0)
					if t > 0 {
						previous = (1 - lam) * dt * a * bRot[(((b*T+t-1)*D)+d)*N+n] * float64(x[(b*T+t-1)*D+d])
					}
					current := lam * dt * bRot[full] * xt
					h[state] = a*h[state] + previous + current
					hAfter[full] = h[state]
				}
			}
			for d := 0; d < D; d++ {
				for n := 0; n < N; n++ {
					totalY += cRot[(((b*T+t)*D)+d)*N+n] * h[((b*D+d)*N)+n]
				}
			}
		}
	}

	gradX := make([]float32, len(x))
	gradDelta := make([]float64, len(dtRaw))
	gradDt := make([]float32, len(dtRaw))
	gradLambda := make([]float32, len(lambdaInput))
	gradTheta := make([]float32, len(theta))
	gradA := make([]float32, len(aLog))
	gradB := make([]float32, len(bProj))
	gradC := make([]float32, len(cProj))
	gradBRot := make([]float64, len(bRot))
	gradCRot := make([]float64, len(cRot))
	gradPhi := make([]float64, len(phi))
	dhNext := make([]float64, B*D*N)
	dy := 1.0 / float64(B*T*D)
	for b := 0; b < B; b++ {
		for t := T - 1; t >= 0; t-- {
			dh := make([]float64, D*N)
			for d := 0; d < D; d++ {
				for n := 0; n < N; n++ {
					full := (((b*T+t)*D)+d)*N + n
					state := (b*D+d)*N + n
					dh[d*N+n] = dhNext[state] + dy*cRot[full]
					gradCRot[full] += dy * hAfter[full]
				}
			}
			for d := 0; d < D; d++ {
				idx := (b*T+t)*D + d
				xt := float64(x[idx])
				ddelta := float64(0)
				dlambda := float64(0)
				dx := float64(0)
				dt := delta[idx]
				lam := lambda[idx]
				for n := 0; n < N; n++ {
					full := (((b*T+t)*D)+d)*N + n
					state := (b*D+d)*N + n
					upstream := dh[d*N+n]
					bv := bRot[full]
					a := alpha[full]
					dx += lam * dt * bv * upstream
					gradBRot[full] += lam * dt * xt * upstream
					deltaTerm := A[d*N+n] * a * hBefore[full]
					aLogTerm := dt * a * A[d*N+n] * hBefore[full]
					if t > 0 {
						prevX := float64(x[(b*T+t-1)*D+d])
						prevFull := (((b*T+t-1)*D)+d)*N + n
						prevB := bRot[prevFull]
						prevInput := prevB * prevX
						betaNoInput := (1 - lam) * dt * a
						gradX[(b*T+t-1)*D+d] += float32(betaNoInput * prevB * upstream)
						gradBRot[prevFull] += betaNoInput * prevX * upstream
						dlambda += (-dt*a*prevInput + dt*bv*xt) * upstream
						deltaTerm += (1 - lam) * (a + dt*A[d*N+n]*a) * prevInput
						aLogTerm += (1 - lam) * dt * dt * a * A[d*N+n] * prevInput
					} else {
						dlambda += dt * bv * xt * upstream
					}
					ddelta += (deltaTerm + lam*bv*xt) * upstream
					gradA[d*N+n] += float32(aLogTerm * upstream)
					dhNext[state] = alpha[full] * upstream
				}
				gradX[idx] += float32(dx)
				gradDelta[idx] += ddelta
				gradLambda[idx] = float32(dlambda * lam * (1 - lam))
			}
		}
	}

	for b := 0; b < B; b++ {
		for t := 0; t < T; t++ {
			for d := 0; d < D; d++ {
				for k := 0; k < K; k++ {
					phiIdx := (((b*T+t)*D)+d)*K + k
					angle := phi[phiIdx]
					cosv := math.Cos(angle)
					sinv := math.Sin(angle)
					full := (((b*T+t)*D)+d)*N + 2*k

					gb0 := gradBRot[full]
					gb1 := gradBRot[full+1]
					gc0 := gradCRot[full]
					gc1 := gradCRot[full+1]

					g := d / channelsPerGroup
					groupBase := ((b*T+t)*G + g) * N
					gradB[groupBase+2*k] += float32(cosv*gb0 - sinv*gb1)
					gradB[groupBase+2*k+1] += float32(sinv*gb0 + cosv*gb1)
					gradC[groupBase+2*k] += float32(cosv*gc0 - sinv*gc1)
					gradC[groupBase+2*k+1] += float32(sinv*gc0 + cosv*gc1)

					bRot0 := bRot[full]
					bRot1 := bRot[full+1]
					cRot0 := cRot[full]
					cRot1 := cRot[full+1]
					gradPhi[phiIdx] += bRot1*gb0 - bRot0*gb1 + cRot1*gc0 - cRot0*gc1
				}
			}
		}
	}

	for b := 0; b < B; b++ {
		for d := 0; d < D; d++ {
			for k := 0; k < K; k++ {
				carry := float64(0)
				for t := T - 1; t >= 0; t-- {
					phiIdx := (((b*T+t)*D)+d)*K + k
					stepIdx := (b*T+t)*D + d
					carry += gradPhi[phiIdx]
					gradTheta[phiIdx] = float32(carry * delta[stepIdx])
					gradDelta[stepIdx] += carry * float64(theta[phiIdx])
				}
			}
		}
	}
	for i := range gradDt {
		gradDt[i] = float32(gradDelta[i] * sigmoid64(float64(dtRaw[i])))
	}

	loss := float32(totalY / float64(B*T*D))
	return loss, [][]float32{gradX, gradDt, gradLambda, gradTheta, gradA, gradB, gradC}
}

func mamba3SISOCPUForwardBackwardPhase2(x, dtRaw, lambdaInput, aLog, bProj, cProj []float32, B, T, D, N int) (float32, [][]float32) {
	totalY := float64(0)
	hBefore := make([]float64, B*T*D*N)
	hAfter := make([]float64, B*T*D*N)
	alpha := make([]float64, B*T*D*N)
	delta := make([]float64, B*T*D)
	lambda := make([]float64, B*T*D)
	A := make([]float64, D*N)
	h := make([]float64, B*D*N)
	for d := 0; d < D; d++ {
		for n := 0; n < N; n++ {
			A[d*N+n] = -math.Exp(float64(aLog[d*N+n]))
		}
	}
	for b := 0; b < B; b++ {
		for t := 0; t < T; t++ {
			for d := 0; d < D; d++ {
				idx := ((b*T + t) * D) + d
				delta[idx] = scanSoftplus64(float64(dtRaw[idx]))
				lambda[idx] = sigmoid64(float64(lambdaInput[idx]))
			}
			for d := 0; d < D; d++ {
				idx := (b*T+t)*D + d
				xt := float64(x[idx])
				dt := delta[idx]
				lam := lambda[idx]
				for n := 0; n < N; n++ {
					state := (b*D+d)*N + n
					full := (((b*T+t)*D)+d)*N + n
					hBefore[full] = h[state]
					a := math.Exp(dt * A[d*N+n])
					alpha[full] = a
					previous := float64(0)
					if t > 0 {
						previous = (1 - lam) * dt * a * float64(bProj[(b*T+t-1)*N+n]) * float64(x[(b*T+t-1)*D+d])
					}
					current := lam * dt * float64(bProj[(b*T+t)*N+n]) * xt
					h[state] = a*h[state] + previous + current
					hAfter[full] = h[state]
				}
			}
			for d := 0; d < D; d++ {
				for n := 0; n < N; n++ {
					totalY += float64(cProj[(b*T+t)*N+n]) * h[((b*D+d)*N)+n]
				}
			}
		}
	}

	gradX := make([]float32, len(x))
	gradDt := make([]float32, len(dtRaw))
	gradLambda := make([]float32, len(lambdaInput))
	gradA := make([]float32, len(aLog))
	gradB := make([]float32, len(bProj))
	gradC := make([]float32, len(cProj))
	dhNext := make([]float64, B*D*N)
	dy := 1.0 / float64(B*T*D)
	for b := 0; b < B; b++ {
		for t := T - 1; t >= 0; t-- {
			dh := make([]float64, D*N)
			for d := 0; d < D; d++ {
				for n := 0; n < N; n++ {
					full := (((b*T+t)*D)+d)*N + n
					state := (b*D+d)*N + n
					dh[d*N+n] = dhNext[state] + dy*float64(cProj[(b*T+t)*N+n])
					gradC[(b*T+t)*N+n] += float32(dy * hAfter[full])
				}
			}
			for d := 0; d < D; d++ {
				idx := (b*T+t)*D + d
				xt := float64(x[idx])
				ddelta := float64(0)
				dlambda := float64(0)
				dx := float64(0)
				dt := delta[idx]
				lam := lambda[idx]
				for n := 0; n < N; n++ {
					full := (((b*T+t)*D)+d)*N + n
					state := (b*D+d)*N + n
					upstream := dh[d*N+n]
					bv := float64(bProj[(b*T+t)*N+n])
					a := alpha[full]
					dx += lam * dt * bv * upstream
					gradB[(b*T+t)*N+n] += float32(lam * dt * xt * upstream)
					deltaTerm := A[d*N+n] * a * hBefore[full]
					aLogTerm := dt * a * A[d*N+n] * hBefore[full]
					if t > 0 {
						prevX := float64(x[(b*T+t-1)*D+d])
						prevB := float64(bProj[(b*T+t-1)*N+n])
						prevInput := prevB * prevX
						betaNoInput := (1 - lam) * dt * a
						gradX[(b*T+t-1)*D+d] += float32(betaNoInput * prevB * upstream)
						gradB[(b*T+t-1)*N+n] += float32(betaNoInput * prevX * upstream)
						dlambda += (-dt*a*prevInput + dt*bv*xt) * upstream
						deltaTerm += (1 - lam) * (a + dt*A[d*N+n]*a) * prevInput
						aLogTerm += (1 - lam) * dt * dt * a * A[d*N+n] * prevInput
					} else {
						dlambda += dt * bv * xt * upstream
					}
					ddelta += (deltaTerm + lam*bv*xt) * upstream
					gradA[d*N+n] += float32(aLogTerm * upstream)
					dhNext[state] = alpha[full] * upstream
				}
				gradX[idx] += float32(dx)
				gradDt[idx] = float32(ddelta * sigmoid64(float64(dtRaw[idx])))
				gradLambda[idx] = float32(dlambda * lam * (1 - lam))
			}
		}
	}

	loss := float32(totalY / float64(B*T*D))
	return loss, [][]float32{gradX, gradDt, gradLambda, gradA, gradB, gradC}
}

func scanSoftplus64(x float64) float64 {
	return math.Log1p(math.Exp(x))
}

func sigmoid64(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func maxGradientError(got, want []float32) (float64, float64) {
	maxRel := float64(0)
	maxAbs := float64(0)
	for i := range got {
		absErr := math.Abs(float64(got[i] - want[i]))
		denom := math.Max(1e-4, math.Max(math.Abs(float64(got[i])), math.Abs(float64(want[i]))))
		rel := absErr / denom
		if rel > maxRel {
			maxRel = rel
		}
		if absErr > maxAbs {
			maxAbs = absErr
		}
	}
	return maxRel, maxAbs
}

func formatLosses(losses []float32) string {
	out := ""
	for i, loss := range losses {
		if i > 0 {
			out += ", "
		}
		out += fmt.Sprintf("%.6f", loss)
	}
	return out
}
