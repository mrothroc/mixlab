//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"math"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestMuonEqROneStepMatchesCPUOracle(t *testing.T) {
	runMuonVariantOneStepOracle(t, MuonNormalizationRowL2)
}

func TestNorMuonOneStepMatchesCPUOracle(t *testing.T) {
	runMuonVariantOneStepOracle(t, MuonNormalizationNorMuon)
}

func runMuonVariantOneStepOracle(t *testing.T, normalization MuonNormalization) {
	t.Helper()
	if !Available() {
		t.Skip("MLX backend not available")
	}
	const (
		rows         = 2
		cols         = 3
		lr           = float32(0.025)
		beta1        = float32(0.9)
		beta2        = float32(0.95)
		backendSteps = 5
	)
	weight := []float32{
		0.42, -0.17, 0.31,
		-0.28, 0.53, -0.09,
	}
	driver := []float32{
		0.60, -0.30, 0.15,
		-0.45, 0.25, 0.70,
	}

	prog := ir.NewProgram(1)
	prog.DeclareInput("driver", ir.TensorFloat32, []int{rows, cols})
	prog.DeclareOutput("loss", ir.TensorFloat32, []int{1})
	prog.Mul("w0", "driver", "prod")
	prog.MeanAxis("prod", 0, "mean_cols")
	prog.MeanAxis("mean_cols", 0, "loss")
	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()

	handle, err := FromData(append([]float32(nil), weight...), rows, cols)
	if err != nil {
		t.Fatalf("FromData(weight): %v", err)
	}
	handles := []int64{handle}
	trainer, err := CreateTrainer(gpuProg, handles, TrainerOptimizerSpec{
		Groups: []OptimizerGroup{{
			Kind:              OptimizerMuon,
			LR:                lr,
			Beta1:             beta1,
			Beta2:             beta2,
			Epsilon:           1e-8,
			BackendSteps:      backendSteps,
			Nesterov:          true,
			MuonNormalization: normalization,
		}},
		Weights:       []WeightOptimizer{{GroupIndex: 0, Decay: false}},
		DefaultBaseLR: lr,
	})
	if err != nil {
		FreeHandles(handles)
		t.Fatalf("CreateTrainer: %v", err)
	}
	defer func() {
		TrainerDestroy(trainer)
		FreeHandles(handles)
	}()

	loss, err := TrainerStep(trainer, []TensorInput{
		{Name: "driver", DType: TensorFloat32, Shape: []int{rows, cols}, Data: driver},
	})
	if err != nil {
		t.Fatalf("TrainerStep: %v", err)
	}
	if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
		t.Fatalf("loss=%g, want finite", loss)
	}
	got := make([]float32, len(weight))
	if err := TrainerReadWeight(trainer, 0, got); err != nil {
		t.Fatalf("TrainerReadWeight: %v", err)
	}
	want := muonVariantOneStepOracle(weight, driver, rows, cols, lr, beta1, beta2, backendSteps, normalization)
	requireCloseSlice(t, got, want, 2e-4)
}

func muonVariantOneStepOracle(weight, driver []float32, rows, cols int, lr, beta1, beta2 float32, steps int, normalization MuonNormalization) []float32 {
	grad := make([]float32, len(weight))
	scale := float32(1.0 / float64(rows*cols))
	for i := range grad {
		grad[i] = driver[i] * scale
	}
	update := make([]float32, len(grad))
	for i := range update {
		momentum := grad[i]
		update[i] = grad[i] + beta1*momentum
	}
	update = muonZeroPowerCPU(update, rows, cols, steps)
	aspect := float32(math.Sqrt(math.Max(1.0, float64(rows)/float64(cols))))
	for i := range update {
		update[i] *= aspect
	}
	switch normalization {
	case MuonNormalizationRowL2:
		muonRowL2NormalizeCPU(update, rows, cols)
	case MuonNormalizationNorMuon:
		muonNorMuonNormalizeCPU(update, rows, cols, beta2)
	}
	out := append([]float32(nil), weight...)
	for i := range out {
		out[i] -= lr * update[i]
	}
	return out
}

func muonZeroPowerCPU(grad []float32, rows, cols, steps int) []float32 {
	x := make([]float32, len(grad))
	var norm float64
	for i, v := range grad {
		q := roundFloat32ToBF16(v)
		x[i] = q
		norm += float64(q) * float64(q)
	}
	invNorm := float32(1.0 / (math.Sqrt(norm) + 1e-7))
	for i := range x {
		x[i] *= invNorm
	}
	if rows > cols {
		x = transposeMatrixCPU(x, rows, cols)
		rows, cols = cols, rows
	}
	const (
		a = float32(3.4445)
		b = float32(-4.7750)
		c = float32(2.0315)
	)
	for step := 0; step < steps; step++ {
		xxT := matMulCPU(x, transposeMatrixCPU(x, rows, cols), rows, cols, rows)
		xxT2 := matMulCPU(xxT, xxT, rows, rows, rows)
		bmat := make([]float32, rows*rows)
		for i := range bmat {
			bmat[i] = b*xxT[i] + c*xxT2[i]
		}
		bx := matMulCPU(bmat, x, rows, rows, cols)
		for i := range x {
			x[i] = a*x[i] + bx[i]
		}
	}
	if len(x) != len(grad) {
		x = transposeMatrixCPU(x, rows, cols)
	}
	return x
}

func muonRowL2NormalizeCPU(x []float32, rows, cols int) {
	for r := 0; r < rows; r++ {
		var sum float64
		for c := 0; c < cols; c++ {
			v := x[r*cols+c]
			sum += float64(v) * float64(v)
		}
		inv := float32(1.0 / (math.Sqrt(sum) + 1e-7))
		for c := 0; c < cols; c++ {
			x[r*cols+c] *= inv
		}
	}
}

func muonNorMuonNormalizeCPU(x []float32, rows, cols int, beta2 float32) {
	normalized := make([]float32, len(x))
	if rows >= cols {
		for r := 0; r < rows; r++ {
			var meanSq float64
			for c := 0; c < cols; c++ {
				v := x[r*cols+c]
				meanSq += float64(v) * float64(v)
			}
			second := float64(1-beta2) * meanSq / float64(cols)
			scale := float32(1.0 / math.Sqrt(math.Max(second, 1e-10)))
			for c := 0; c < cols; c++ {
				normalized[r*cols+c] = x[r*cols+c] * scale
			}
		}
	} else {
		for c := 0; c < cols; c++ {
			var meanSq float64
			for r := 0; r < rows; r++ {
				v := x[r*cols+c]
				meanSq += float64(v) * float64(v)
			}
			second := float64(1-beta2) * meanSq / float64(rows)
			scale := float32(1.0 / math.Sqrt(math.Max(second, 1e-10)))
			for r := 0; r < rows; r++ {
				normalized[r*cols+c] = x[r*cols+c] * scale
			}
		}
	}
	oldNorm := frobeniusNormCPU(x)
	newNorm := frobeniusNormCPU(normalized)
	rescale := float32(oldNorm / math.Max(newNorm, 1e-10))
	for i := range x {
		x[i] = normalized[i] * rescale
	}
}

func matMulCPU(a, b []float32, rows, inner, cols int) []float32 {
	out := make([]float32, rows*cols)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			var sum float32
			for k := 0; k < inner; k++ {
				sum += a[r*inner+k] * b[k*cols+c]
			}
			out[r*cols+c] = sum
		}
	}
	return out
}

func transposeMatrixCPU(x []float32, rows, cols int) []float32 {
	out := make([]float32, len(x))
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			out[c*rows+r] = x[r*cols+c]
		}
	}
	return out
}

func frobeniusNormCPU(x []float32) float64 {
	var sum float64
	for _, v := range x {
		sum += float64(v) * float64(v)
	}
	return math.Sqrt(sum)
}

func roundFloat32ToBF16(v float32) float32 {
	bits := math.Float32bits(v)
	round := uint32(0x7fff) + ((bits >> 16) & 1)
	return math.Float32frombits((bits + round) & 0xffff0000)
}
