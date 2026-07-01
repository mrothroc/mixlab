//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"math"
	"reflect"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestTrainerSampleCategoricalOutputHandlesNonFiniteRows(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}
	const (
		rows  = 4
		vocab = 4
	)
	prog := ir.NewProgram(1)
	prog.DeclareInput("logits", ir.TensorFloat32, []int{rows, vocab})
	prog.DeclareOutput("sample_logits", ir.TensorFloat32, []int{rows, vocab})
	prog.StopGradient("logits", "sample_logits")

	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()
	dummy, err := FromData([]float32{0}, 1, 1)
	if err != nil {
		t.Fatalf("FromData: %v", err)
	}
	defer FreeHandle(dummy)
	trainer, err := CreateTrainer(gpuProg, []int64{dummy}, TrainerOptimizerSpec{
		Groups: []OptimizerGroup{{
			Kind:    OptimizerAdamW,
			LR:      1e-3,
			Beta1:   0.9,
			Beta2:   0.99,
			Epsilon: 1e-8,
		}},
		Weights:       []WeightOptimizer{{GroupIndex: 0}},
		DefaultBaseLR: 1e-3,
	})
	if err != nil {
		t.Fatalf("CreateTrainer: %v", err)
	}
	defer TrainerDestroy(trainer)

	logits := []float32{
		float32(math.NaN()), float32(math.NaN()), float32(math.NaN()), float32(math.NaN()),
		float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)),
		0, float32(math.Inf(1)), float32(math.Inf(1)), -3,
		float32(math.NaN()), -100, 100, -100,
	}
	inputs := []TensorInput{{
		Name:  "logits",
		DType: TensorFloat32,
		Shape: []int{rows, vocab},
		Data:  logits,
	}}
	first, err := TrainerSampleCategoricalOutput(trainer, inputs, "sample_logits", rows, vocab, 1, 1234)
	if err != nil {
		t.Fatalf("TrainerSampleCategoricalOutput(first): %v", err)
	}
	second, err := TrainerSampleCategoricalOutput(trainer, inputs, "sample_logits", rows, vocab, 1, 1234)
	if err != nil {
		t.Fatalf("TrainerSampleCategoricalOutput(second): %v", err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("fixed seed samples are not deterministic: %v vs %v", first, second)
	}
	for row, sample := range first {
		if sample < 0 || sample >= vocab {
			t.Fatalf("sample row %d=%d outside vocab [0,%d)", row, sample, vocab)
		}
	}
	if first[2] != 1 && first[2] != 2 {
		t.Fatalf("+Inf row sample=%d, want one of positive-infinity logits 1 or 2", first[2])
	}
	if first[3] != 2 {
		t.Fatalf("finite row with NaNs sample=%d, want dominant finite token 2", first[3])
	}
}
