//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"math"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestBatchNormTrainingMatchesCPUForwardBackwardAndUpdatesBuffers(t *testing.T) {
	lockMLXThread(t)
	if !Available() {
		t.Skip("MLX backend not available")
	}
	const (
		rows = 4
		dim  = 2
		eps  = 1e-5
	)
	x := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	scale := []float32{1.25, -0.75}
	bias := []float32{0.1, -0.2}
	runningMean := []float32{0, 0}
	runningVar := []float32{1, 1}
	targets := []int32{0, 1, 0, 1}

	prog := ir.NewProgram(5)
	prog.DeclareInput("targets", ir.TensorInt32, []int{rows})
	prog.DeclareOutput("loss", ir.TensorFloat32, []int{1})
	prog.DeclareOutput("bn_out", ir.TensorFloat32, []int{rows, dim})
	prog.BatchNorm(
		"w0", "w1", "w2", "w3", "w4",
		"bn_out", "bn_mean_update", "bn_var_update",
		eps, 0.1,
	)
	prog.CrossEntropy("bn_out", "targets", "loss")

	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatal(err)
	}
	defer gpuProg.Destroy()
	weightData := [][]float32{x, scale, bias, runningMean, runningVar}
	weightShapes := [][]int{{rows, dim}, {dim}, {dim}, {dim}, {dim}}
	handles := make([]int64, len(weightData))
	for i := range weightData {
		handles[i], err = FromDataShape(append([]float32(nil), weightData[i]...), weightShapes[i])
		if err != nil {
			FreeHandles(handles[:i])
			t.Fatal(err)
		}
	}
	defer FreeHandles(handles)
	spec := TrainerOptimizerSpec{
		Groups: []OptimizerGroup{{
			Kind: OptimizerSGD, LR: 1e-4,
		}},
		Weights: []WeightOptimizer{
			{GroupIndex: 0}, {GroupIndex: 0}, {GroupIndex: 0},
			{Frozen: true}, {Frozen: true},
		},
		DefaultBaseLR: 1e-4,
	}
	trainer, err := CreateTrainer(gpuProg, handles, spec)
	if err != nil {
		t.Fatal(err)
	}
	defer TrainerDestroy(trainer)
	if err := TrainerSetStepOutputNames(trainer, []string{"bn_out"}); err != nil {
		t.Fatal(err)
	}
	inputs := []TensorInput{{
		Name: "targets", DType: TensorInt32, Shape: []int{rows}, Data: targets,
	}}
	if _, err := TrainerComputeMeanSquareGrads(trainer, inputs, "loss"); err != nil {
		t.Fatalf("TrainerComputeMeanSquareGrads: %v", err)
	}
	wantOut, wantDX, wantDScale, wantDBias := batchNormCrossEntropyCPU(
		x, scale, bias, targets, rows, dim, eps,
	)
	// TrainerComputeMeanSquareGrads differentiates mean(square(loss)); for the
	// scalar CE loss this scales the ordinary CE gradient by 2*loss.
	gradScale := float32(2 * crossEntropyMeanCPU(wantOut, targets, rows, dim))
	for _, values := range [][]float32{wantDX, wantDScale, wantDBias} {
		for i := range values {
			values[i] *= gradScale
		}
	}
	// Fixed torch.nn.BatchNorm1d fixture (PyTorch 2.12.1, input transposed
	// from [B,T,D] to [B,D,T], loss=CrossEntropy(logits)^2).
	torchOut := []float32{
		-1.5770493, 0.80622953, -0.45901656, 0.13540983,
		0.65901637, -0.5354098, 1.7770491, -1.2062296,
	}
	torchDX := []float32{
		-0.08819537, -0.052917223, 0.24573869, 0.14744323,
		-0.22689348, -0.13613608, 0.06935016, 0.041610077,
	}
	torchDScale := []float32{1.7401016, -1.7401015}
	torchDBias := []float32{0.1237154, -0.1237154}
	for name, pair := range map[string][2][]float32{
		"forward": {wantOut, torchOut},
		"dx":      {wantDX, torchDX},
		"dscale":  {wantDScale, torchDScale},
		"dbias":   {wantDBias, torchDBias},
	} {
		if diff := maxAbsDiffFloat32(pair[0], pair[1]); diff > 3e-6 {
			t.Fatalf("CPU oracle vs PyTorch %s L_inf=%g", name, diff)
		}
	}
	for i, want := range [][]float32{wantDX, wantDScale, wantDBias} {
		got := make([]float32, len(want))
		if err := TrainerReadGrad(trainer, i, got); err != nil {
			t.Fatalf("TrainerReadGrad(%d): %v", i, err)
		}
		if diff := maxAbsDiffFloat32(got, want); diff > 3e-5 {
			t.Fatalf("BatchNorm grad[%d] L_inf=%g\ngot=%v\nwant=%v", i, diff, got, want)
		}
	}

	if _, err := TrainerStep(trainer, inputs); err != nil {
		t.Fatalf("TrainerStep: %v", err)
	}

	gotOut, err := TrainerReadOutput(trainer, "bn_out", []int{rows, dim})
	if err != nil {
		t.Fatal(err)
	}
	if diff := maxAbsDiffFloat32(gotOut, wantOut); diff > 2e-5 {
		t.Fatalf("BatchNorm forward L_inf=%g\ngot=%v\nwant=%v", diff, gotOut, wantOut)
	}
	gotMean := make([]float32, dim)
	gotVar := make([]float32, dim)
	if err := TrainerReadWeight(trainer, 3, gotMean); err != nil {
		t.Fatal(err)
	}
	if err := TrainerReadWeight(trainer, 4, gotVar); err != nil {
		t.Fatal(err)
	}
	wantMean := []float32{0.4, 0.5}
	wantVar := []float32{1.5666667, 1.5666667}
	if diff := maxAbsDiffFloat32(gotMean, wantMean); diff > 1e-6 {
		t.Fatalf("running mean=%v want %v", gotMean, wantMean)
	}
	if diff := maxAbsDiffFloat32(gotVar, wantVar); diff > 2e-6 {
		t.Fatalf("running var=%v want %v", gotVar, wantVar)
	}
	if _, err := TrainerEvaluateWithOutputs(trainer, inputs, []string{"bn_out"}); err != nil {
		t.Fatal(err)
	}
	evalOut, err := TrainerReadOutput(trainer, "bn_out", []int{rows, dim})
	if err != nil {
		t.Fatal(err)
	}
	if diff := maxAbsDiffFloat32(evalOut, gotOut); diff < 0.1 {
		t.Fatalf("train/eval BatchNorm outputs unexpectedly agree: L_inf=%g", diff)
	}
}

func crossEntropyMeanCPU(logits []float32, targets []int32, rows, dim int) float64 {
	var loss float64
	for r := 0; r < rows; r++ {
		maxLogit := math.Inf(-1)
		for d := 0; d < dim; d++ {
			maxLogit = math.Max(maxLogit, float64(logits[r*dim+d]))
		}
		var denom float64
		for d := 0; d < dim; d++ {
			denom += math.Exp(float64(logits[r*dim+d]) - maxLogit)
		}
		target := float64(logits[r*dim+int(targets[r])])
		loss += math.Log(denom) + maxLogit - target
	}
	return loss / float64(rows)
}

func TestBatchNormEvaluationUsesRunningStatsNotBatchComposition(t *testing.T) {
	lockMLXThread(t)
	if !Available() {
		t.Skip("MLX backend not available")
	}
	const (
		rows = 4
		dim  = 2
	)
	prog := ir.NewProgram(5)
	prog.DeclareInput("targets", ir.TensorInt32, []int{rows})
	prog.DeclareOutput("loss", ir.TensorFloat32, []int{1})
	prog.DeclareOutput("bn_out", ir.TensorFloat32, []int{rows, dim})
	prog.BatchNorm("w0", "w1", "w2", "w3", "w4", "bn_out", "mean_update", "var_update", 1e-5, 0.1)
	prog.CrossEntropy("bn_out", "targets", "loss")
	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatal(err)
	}
	defer gpuProg.Destroy()

	weights := [][]float32{
		{1, 2, 3, 4, 5, 6, 7, 8},
		{1, 1},
		{0, 0},
		{4, 5},
		{4, 9},
	}
	shapes := [][]int{{rows, dim}, {dim}, {dim}, {dim}, {dim}}
	handles := make([]int64, len(weights))
	for i := range weights {
		handles[i], err = FromDataShape(weights[i], shapes[i])
		if err != nil {
			FreeHandles(handles[:i])
			t.Fatal(err)
		}
	}
	defer FreeHandles(handles)
	trainer, err := CreateTrainer(gpuProg, handles, TrainerOptimizerSpec{
		Groups: []OptimizerGroup{{Kind: OptimizerSGD, LR: 1e-4}},
		Weights: []WeightOptimizer{
			{GroupIndex: 0}, {GroupIndex: 0}, {GroupIndex: 0},
			{Frozen: true}, {Frozen: true},
		},
		DefaultBaseLR: 1e-4,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer TrainerDestroy(trainer)
	inputs := []TensorInput{{Name: "targets", DType: TensorInt32, Shape: []int{rows}, Data: []int32{0, 0, 0, 0}}}

	if _, err := TrainerEvaluateWithOutputs(trainer, inputs, []string{"bn_out"}); err != nil {
		t.Fatal(err)
	}
	first, err := TrainerReadOutput(trainer, "bn_out", []int{rows, dim})
	if err != nil {
		t.Fatal(err)
	}
	if err := TrainerSetWeight(trainer, 0, []float32{1, 2, 30, 40, -50, 60, 70, -80}); err != nil {
		t.Fatal(err)
	}
	if _, err := TrainerEvaluateWithOutputs(trainer, inputs, []string{"bn_out"}); err != nil {
		t.Fatal(err)
	}
	second, err := TrainerReadOutput(trainer, "bn_out", []int{rows, dim})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(first[0]-second[0])) > 1e-7 || math.Abs(float64(first[1]-second[1])) > 1e-7 {
		t.Fatalf("same sample changed with evaluation batch composition: first=%v second=%v", first[:2], second[:2])
	}
}

func batchNormCrossEntropyCPU(
	x, scale, bias []float32,
	targets []int32,
	rows, dim int,
	eps float64,
) (out, dx, dScale, dBias []float32) {
	mean := make([]float64, dim)
	for r := 0; r < rows; r++ {
		for d := 0; d < dim; d++ {
			mean[d] += float64(x[r*dim+d])
		}
	}
	for d := range mean {
		mean[d] /= float64(rows)
	}
	variance := make([]float64, dim)
	for r := 0; r < rows; r++ {
		for d := 0; d < dim; d++ {
			diff := float64(x[r*dim+d]) - mean[d]
			variance[d] += diff * diff
		}
	}
	inv := make([]float64, dim)
	for d := range variance {
		variance[d] /= float64(rows)
		inv[d] = 1 / math.Sqrt(variance[d]+eps)
	}
	z := make([]float64, rows*dim)
	y := make([]float64, rows*dim)
	out = make([]float32, rows*dim)
	for r := 0; r < rows; r++ {
		for d := 0; d < dim; d++ {
			i := r*dim + d
			z[i] = (float64(x[i]) - mean[d]) * inv[d]
			y[i] = z[i]*float64(scale[d]) + float64(bias[d])
			out[i] = float32(y[i])
		}
	}
	dy := make([]float64, rows*dim)
	for r := 0; r < rows; r++ {
		maxLogit := math.Inf(-1)
		for d := 0; d < dim; d++ {
			maxLogit = math.Max(maxLogit, y[r*dim+d])
		}
		var denom float64
		for d := 0; d < dim; d++ {
			denom += math.Exp(y[r*dim+d] - maxLogit)
		}
		for d := 0; d < dim; d++ {
			p := math.Exp(y[r*dim+d]-maxLogit) / denom
			if int32(d) == targets[r] {
				p--
			}
			dy[r*dim+d] = p / float64(rows)
		}
	}
	dx = make([]float32, rows*dim)
	dScale = make([]float32, dim)
	dBias = make([]float32, dim)
	for d := 0; d < dim; d++ {
		var sumDZ, sumDZZ float64
		for r := 0; r < rows; r++ {
			i := r*dim + d
			dScale[d] += float32(dy[i] * z[i])
			dBias[d] += float32(dy[i])
			dz := dy[i] * float64(scale[d])
			sumDZ += dz
			sumDZZ += dz * z[i]
		}
		for r := 0; r < rows; r++ {
			i := r*dim + d
			dz := dy[i] * float64(scale[d])
			dx[i] = float32(inv[d] * (float64(rows)*dz - sumDZ - z[i]*sumDZZ) / float64(rows))
		}
	}
	return out, dx, dScale, dBias
}
