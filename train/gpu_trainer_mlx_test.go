//go:build mlx && cgo && (darwin || linux)

package train

import (
	"reflect"
	"testing"
)

func TestMLXGPUTrainerMakeInputs_ExtendedTargets(t *testing.T) {
	trainer := &mlxGPUTrainer{
		declaredTargetSize: 40,
		tokBuf:             make([]int32, 32),
		tgtBuf:             make([]int32, 32),
	}

	xTok := make([]int, 32)
	yTok := make([]int, 32)
	for i := range xTok {
		xTok[i] = 100 + i
		yTok[i] = i + 1
	}

	inputs, err := trainer.makeInputs(xTok, yTok, 2, 16)
	if err != nil {
		t.Fatalf("makeInputs: %v", err)
	}
	if len(inputs) != 2 {
		t.Fatalf("len(inputs) = %d, want 2", len(inputs))
	}

	targets, ok := inputs[1].Data.([]int32)
	if !ok {
		t.Fatalf("targets data type = %T, want []int32", inputs[1].Data)
	}
	want := []int32{
		1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
		1, 5, 9, 13,
		17, 21, 25, 29,
	}
	if !reflect.DeepEqual(inputs[1].Shape, []int{40}) {
		t.Fatalf("targets shape = %v, want [40]", inputs[1].Shape)
	}
	if !reflect.DeepEqual(targets, want) {
		t.Fatalf("targets = %v, want %v", targets, want)
	}
}

func TestMLXGPUTrainerMakeInputs_StandardTargetsUnchanged(t *testing.T) {
	trainer := &mlxGPUTrainer{
		declaredTargetSize: 0,
		tokBuf:             make([]int32, 8),
		tgtBuf:             make([]int32, 8),
	}

	xTok := []int{10, 11, 12, 13, 14, 15, 16, 17}
	yTok := []int{1, 2, 3, 4, 5, 6, 7, 8}

	inputs, err := trainer.makeInputs(xTok, yTok, 2, 4)
	if err != nil {
		t.Fatalf("makeInputs: %v", err)
	}

	targets, ok := inputs[1].Data.([]int32)
	if !ok {
		t.Fatalf("targets data type = %T, want []int32", inputs[1].Data)
	}
	want := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	if !reflect.DeepEqual(inputs[1].Shape, []int{8}) {
		t.Fatalf("targets shape = %v, want [8]", inputs[1].Shape)
	}
	if !reflect.DeepEqual(targets, want) {
		t.Fatalf("targets = %v, want %v", targets, want)
	}
}

func TestMLXGPUTrainerPrepareTargets_InvalidExtendedShape(t *testing.T) {
	trainer := &mlxGPUTrainer{
		declaredTargetSize: 19,
		tgtBuf:             make([]int32, 16),
	}

	if _, _, err := trainer.prepareTargets(2, 8, 16); err == nil {
		t.Fatal("prepareTargets succeeded, want error")
	}
}
