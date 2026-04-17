//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"math"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestAdamParityWithReferenceAdamTrainer(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}

	prog := ir.NewProgram(2)
	prog.DeclareInput("tokens", ir.TensorInt32, []int{1, 4})
	prog.DeclareInput("targets", ir.TensorInt32, []int{1, 4})
	prog.DeclareOutput("loss", ir.TensorFloat32, []int{1})
	prog.Embed("w0", "tokens", "emb")
	prog.Reshape("emb", []int{4, 2}, "emb_flat")
	prog.MatMul("emb_flat", "w1", "logits")
	prog.CrossEntropy("logits", "targets", "loss")

	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()

	w0 := []float32{
		0.1, 0.2,
		0.3, 0.4,
		0.5, 0.6,
		0.7, 0.8,
	}
	w1 := []float32{
		0.2, -0.1, 0.0, 0.3,
		0.4, 0.1, -0.2, 0.5,
	}
	referenceHandles := make([]int64, 2)
	v2Handles := make([]int64, 2)
	for i, data := range [][]float32{w0, w1} {
		rows, cols := 4, 2
		if i == 1 {
			rows, cols = 2, 4
		}
		referenceHandles[i], err = FromData(append([]float32(nil), data...), rows, cols)
		if err != nil {
			t.Fatalf("FromData reference %d: %v", i, err)
		}
		defer FreeHandle(referenceHandles[i])
		v2Handles[i], err = FromData(append([]float32(nil), data...), rows, cols)
		if err != nil {
			t.Fatalf("FromData v2 %d: %v", i, err)
		}
		defer FreeHandle(v2Handles[i])
	}

	const (
		lr          = float32(0.001)
		beta1       = float32(0.9)
		beta2       = float32(0.95)
		eps         = float32(1e-8)
		wd          = float32(0.01)
		maxGradNorm = float32(0.0)
	)
	referenceTrainer, err := createLegacyAdamTrainer(gpuProg, referenceHandles, []bool{false, true}, lr, beta1, beta2, eps, wd, maxGradNorm)
	if err != nil {
		t.Fatalf("createLegacyAdamTrainer: %v", err)
	}
	defer TrainerDestroy(referenceTrainer)

	v2Trainer, err := CreateTrainer(gpuProg, v2Handles, TrainerOptimizerSpec{
		Groups: []OptimizerGroup{{
			Kind:        OptimizerAdamW,
			LR:          lr,
			Beta1:       beta1,
			Beta2:       beta2,
			Epsilon:     eps,
			WeightDecay: wd,
		}},
		Weights: []WeightOptimizer{
			{GroupIndex: 0, Decay: false},
			{GroupIndex: 0, Decay: true},
		},
		MaxGradNorm:   maxGradNorm,
		DefaultBaseLR: lr,
	})
	if err != nil {
		t.Fatalf("CreateTrainer(v2): %v", err)
	}
	defer TrainerDestroy(v2Trainer)

	inputs := []TensorInput{
		{Name: "tokens", DType: TensorInt32, Shape: []int{1, 4}, Data: []int32{0, 1, 2, 3}},
		{Name: "targets", DType: TensorInt32, Shape: []int{4}, Data: []int32{1, 2, 3, 0}},
	}

	for step := 0; step < 2; step++ {
		referenceLoss, err := TrainerStep(referenceTrainer, inputs)
		if err != nil {
			t.Fatalf("reference TrainerStep step %d: %v", step, err)
		}
		v2Loss, err := TrainerStep(v2Trainer, inputs)
		if err != nil {
			t.Fatalf("v2 TrainerStep step %d: %v", step, err)
		}
		if diff := math.Abs(float64(referenceLoss - v2Loss)); diff > 1e-6 {
			t.Fatalf("step %d loss mismatch: reference=%g v2=%g diff=%g", step, referenceLoss, v2Loss, diff)
		}
	}

	for weightIdx := 0; weightIdx < 2; weightIdx++ {
		referenceSize, err := TrainerWeightSize(referenceTrainer, weightIdx)
		if err != nil {
			t.Fatalf("reference TrainerWeightSize(%d): %v", weightIdx, err)
		}
		v2Size, err := TrainerWeightSize(v2Trainer, weightIdx)
		if err != nil {
			t.Fatalf("v2 TrainerWeightSize(%d): %v", weightIdx, err)
		}
		if referenceSize != v2Size {
			t.Fatalf("weight %d size mismatch: reference=%d v2=%d", weightIdx, referenceSize, v2Size)
		}
		referenceData := make([]float32, referenceSize)
		if err := TrainerReadWeight(referenceTrainer, weightIdx, referenceData); err != nil {
			t.Fatalf("reference TrainerReadWeight(%d): %v", weightIdx, err)
		}
		v2Data := make([]float32, v2Size)
		if err := TrainerReadWeight(v2Trainer, weightIdx, v2Data); err != nil {
			t.Fatalf("v2 TrainerReadWeight(%d): %v", weightIdx, err)
		}
		for i := range referenceData {
			if diff := math.Abs(float64(referenceData[i] - v2Data[i])); diff > 1e-6 {
				t.Fatalf("weight %d elem %d mismatch: reference=%g v2=%g diff=%g", weightIdx, i, referenceData[i], v2Data[i], diff)
			}
		}
	}
}
