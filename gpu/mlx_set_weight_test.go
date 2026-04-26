//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"math"
	"strings"
	"testing"
)

func compareFloat32Slices(t *testing.T, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("slice length mismatch: got=%d want=%d", len(got), len(want))
	}
	for i := range got {
		if diff := math.Abs(float64(got[i] - want[i])); diff > tol {
			t.Fatalf("value[%d] mismatch: got=%g want=%g diff=%g", i, got[i], want[i], diff)
		}
	}
}

func TestTrainerSetWeightRoundTrip(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}

	gpuProg, err := LowerIRProgram(makePipelineTestProgram())
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()

	trainer := createPipelineTestTrainer(t, gpuProg)
	want := []float32{
		1.5, 1.25,
		1.0, 0.75,
		0.5, 0.25,
		0.0, -0.25,
	}

	if err := TrainerSetWeight(trainer, 0, want); err != nil {
		t.Fatalf("TrainerSetWeight: %v", err)
	}

	got := make([]float32, len(want))
	if err := TrainerReadWeight(trainer, 0, got); err != nil {
		t.Fatalf("TrainerReadWeight: %v", err)
	}

	compareFloat32Slices(t, got, want, 1e-6)
}

func TestTrainerSetWeightSizeMismatch(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}

	gpuProg, err := LowerIRProgram(makePipelineTestProgram())
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()

	trainer := createPipelineTestTrainer(t, gpuProg)
	before := readAllTrainerWeights(t, trainer)[0]

	err = TrainerSetWeight(trainer, 0, before[:len(before)-1])
	if err == nil {
		t.Fatal("TrainerSetWeight size mismatch: expected error")
	}
	if !strings.Contains(err.Error(), "expects") {
		t.Fatalf("TrainerSetWeight size mismatch error = %q, want size mismatch context", err)
	}

	after := make([]float32, len(before))
	if err := TrainerReadWeight(trainer, 0, after); err != nil {
		t.Fatalf("TrainerReadWeight: %v", err)
	}
	compareFloat32Slices(t, after, before, 1e-6)
}

func TestTrainerSetWeightInvalidIndex(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}

	gpuProg, err := LowerIRProgram(makePipelineTestProgram())
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()

	trainer := createPipelineTestTrainer(t, gpuProg)
	replacement := make([]float32, len(pipelineTestWeights()[0]))

	err = TrainerSetWeight(trainer, -1, replacement)
	if err == nil || !strings.Contains(err.Error(), "invalid weight index") {
		t.Fatalf("TrainerSetWeight(-1) error = %v, want invalid weight index", err)
	}

	err = TrainerSetWeight(trainer, 99, replacement)
	if err == nil {
		t.Fatal("TrainerSetWeight(out of range): expected error")
	}
}

func TestTrainerSetWeightAffectsForward(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}

	gpuProg, err := LowerIRProgram(makePipelineTestProgram())
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()

	trainer := createPipelineTestTrainer(t, gpuProg)
	inputs := pipelineTestInputs(0)

	baseline, err := TrainerEvaluate(trainer, inputs)
	if err != nil {
		t.Fatalf("TrainerEvaluate baseline: %v", err)
	}

	zeros := make([]float32, len(pipelineTestWeights()[0]))
	if err := TrainerSetWeight(trainer, 0, zeros); err != nil {
		t.Fatalf("TrainerSetWeight: %v", err)
	}

	after, err := TrainerEvaluate(trainer, inputs)
	if err != nil {
		t.Fatalf("TrainerEvaluate after set: %v", err)
	}
	if math.Abs(float64(after-baseline)) <= 1e-6 {
		t.Fatalf("loss unchanged after replacing weight: baseline=%g after=%g", baseline, after)
	}
}

func TestTrainerSetWeightLeavesOtherWeightsAlone(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}

	gpuProg, err := LowerIRProgram(makePipelineTestProgram())
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()

	trainer := createPipelineTestTrainer(t, gpuProg)
	before := readAllTrainerWeights(t, trainer)
	replacement := []float32{
		9.0, 8.0,
		7.0, 6.0,
		5.0, 4.0,
		3.0, 2.0,
	}

	if err := TrainerSetWeight(trainer, 0, replacement); err != nil {
		t.Fatalf("TrainerSetWeight: %v", err)
	}

	after := readAllTrainerWeights(t, trainer)
	compareFloat32Slices(t, after[0], replacement, 1e-6)
	for i := 1; i < len(before); i++ {
		compareFloat32Slices(t, after[i], before[i], 1e-6)
	}
}
