package train

import (
	"math"
	"testing"
)

func TestS4DSpecialWeightInitialization(t *testing.T) {
	shapes := []WeightShape{
		{Name: "s4d_log_dt", Shape: []int{256}, InitMode: "s4d_log_dt", DtMin: 0.001, DtMax: 0.1},
		{Name: "s4d_log_A_real", Shape: []int{2, 3}, InitValue: float32(math.Log(0.5))},
		{Name: "s4d_A_imag", Shape: []int{2, 3}, InitMode: "s4d_A_imag_lin"},
		{Name: "s4d_B_real", Shape: []int{2, 3}, InitMode: "s4d_B_one"},
		{Name: "s4d_B_imag", Shape: []int{2, 3}, InitZero: true},
		{Name: "s4d_C_real", Shape: []int{2, 3}, InitMode: "s4d_C_normal"},
		{Name: "s4d_C_imag", Shape: []int{2, 3}, InitMode: "s4d_C_normal"},
		{Name: "s4d_D", Shape: []int{2}, InitMode: "s4d_D_normal"},
	}
	weights := initWeightData(shapes, 17, "", 0)
	for _, value := range weights[0] {
		dt := math.Exp(float64(value))
		if dt < 0.001 || dt > 0.1 {
			t.Fatalf("exp(log_dt)=%g outside [0.001,0.1]", dt)
		}
	}
	for _, value := range weights[1] {
		if math.Abs(float64(value)-math.Log(0.5)) > 1e-7 {
			t.Fatalf("log_A_real=%g want log(0.5)", value)
		}
	}
	wantImag := []float32{0, math.Pi, 2 * math.Pi, 0, math.Pi, 2 * math.Pi}
	for i, want := range wantImag {
		if math.Abs(float64(weights[2][i]-want)) > 1e-6 {
			t.Fatalf("A_imag[%d]=%g want %g", i, weights[2][i], want)
		}
	}
	for i, value := range weights[3] {
		if value != 1 {
			t.Fatalf("B_real[%d]=%g want 1", i, value)
		}
	}
	for i, value := range weights[4] {
		if value != 0 {
			t.Fatalf("B_imag[%d]=%g want 0", i, value)
		}
	}
	for i := 5; i < len(weights); i++ {
		nonzero := false
		for _, value := range weights[i] {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatalf("%s contains non-finite value", shapes[i].Name)
			}
			nonzero = nonzero || value != 0
		}
		if !nonzero {
			t.Fatalf("%s initialized entirely to zero", shapes[i].Name)
		}
	}
}
