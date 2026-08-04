package train

import (
	"math"
	"reflect"
	"testing"
)

func TestPyTorchLinearInitUsesPairedFanInForWeightAndBias(t *testing.T) {
	shapes := []WeightShape{
		{Name: "proj", Shape: []int{1, 512}, PyTorchLinearFanIn: 1},
		{Name: "bias", Shape: []int{512}, InitZero: true, PyTorchLinearFanIn: 1},
		{Name: "unmarked_embedding", Shape: []int{3, 512}},
	}
	got := initWeightData(shapes, 17, "pytorch_linear", 0)
	for i := 0; i < 2; i++ {
		assertValuesWithin(t, shapes[i].Name, got[i], 1)
		assertAnyNonZero(t, shapes[i].Name, got[i])
	}

	// Unmarked matrices retain Mixlab's Xavier fallback under this mode.
	xavierBound := math.Sqrt(6.0 / float64(3+512))
	assertValuesWithin(t, shapes[2].Name, got[2], xavierBound)
	baseline := initWeightData(shapes, 17, "xavier_uniform", 0)
	if !reflect.DeepEqual(got[2], baseline[2]) {
		t.Fatal("unmarked matrix did not retain Xavier initialization")
	}
}

func TestPyTorchLinearMetadataDoesNotChangeExistingModes(t *testing.T) {
	base := []WeightShape{
		{Name: "proj", Shape: []int{8, 16}},
		{Name: "bias", Shape: []int{16}, InitZero: true},
	}
	marked := []WeightShape{
		{Name: "proj", Shape: []int{8, 16}, PyTorchLinearFanIn: 8},
		{Name: "bias", Shape: []int{16}, InitZero: true, PyTorchLinearFanIn: 8},
	}
	for _, mode := range []string{"", "xavier_uniform", "normal", "gptbert", "gpt2"} {
		t.Run(mode, func(t *testing.T) {
			want := initWeightData(base, 42, mode, 0.02)
			got := initWeightData(marked, 42, mode, 0.02)
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("mode %q changed when affine metadata was added", mode)
			}
		})
	}
}

func TestPyTorchLinearPreservesS4DSpecialInitializerStream(t *testing.T) {
	shapes := []WeightShape{
		{Name: "input_adapter_proj", Shape: []int{1, 8}, PyTorchLinearFanIn: 1},
		{Name: "input_adapter_bias", Shape: []int{8}, InitZero: true, PyTorchLinearFanIn: 1},
		{Name: "s4d_log_dt", Shape: []int{8}, InitMode: "s4d_log_dt", DtMin: 0.001, DtMax: 0.1},
		{Name: "s4d_log_A_real", Shape: []int{2, 3}, InitValue: float32(math.Log(0.5))},
		{Name: "s4d_A_imag", Shape: []int{2, 3}, InitMode: "s4d_A_imag_lin"},
		{Name: "s4d_B_real", Shape: []int{2, 3}, InitMode: "s4d_B_one"},
		{Name: "s4d_B_imag", Shape: []int{2, 3}, InitZero: true},
		{Name: "s4d_C_real", Shape: []int{8, 3}, InitMode: "s4d_C_normal"},
		{Name: "s4d_C_imag", Shape: []int{8, 3}, InitMode: "s4d_C_normal"},
		{Name: "s4d_D", Shape: []int{8}, InitMode: "s4d_D_normal"},
	}
	baseline := initWeightData(shapes, 2222, "xavier_uniform", 0)
	got := initWeightData(shapes, 2222, "pytorch_linear", 0)
	for i := 2; i < len(shapes); i++ {
		if !reflect.DeepEqual(got[i], baseline[i]) {
			t.Fatalf("special initializer %s changed under pytorch_linear", shapes[i].Name)
		}
	}
	if reflect.DeepEqual(got[0], baseline[0]) {
		t.Fatal("input projection unexpectedly matched Xavier initialization")
	}
	assertAnyNonZero(t, "input_adapter_bias", got[1])
}

func assertValuesWithin(t *testing.T, name string, values []float32, bound float64) {
	t.Helper()
	for i, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || math.Abs(float64(value)) > bound {
			t.Fatalf("%s[%d]=%g outside finite bound +/- %g", name, i, value, bound)
		}
	}
}

func assertAnyNonZero(t *testing.T, name string, values []float32) {
	t.Helper()
	for _, value := range values {
		if value != 0 {
			return
		}
	}
	t.Fatalf("%s initialized entirely to zero", name)
}
