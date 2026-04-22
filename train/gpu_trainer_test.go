package train

import (
	"math"
	"testing"
)

// TestShouldDecayWeight verifies that 2-D+ weights get decay but 1-D does not.
func TestShouldDecayWeight(t *testing.T) {
	tests := []struct {
		name  string
		shape []int
		want  bool
	}{
		{"1D_norm", []int{256}, false},
		{"1D_bias", []int{64}, false},
		{"1D_single", []int{1}, false},
		{"2D_matrix", []int{64, 128}, true},
		{"2D_square", []int{32, 32}, true},
		{"3D_tensor", []int{4, 8, 16}, true},
		{"empty_shape", []int{}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := shouldDecayWeight(tt.shape)
			if got != tt.want {
				t.Errorf("shouldDecayWeight(%v) = %v, want %v", tt.shape, got, tt.want)
			}
		})
	}
}

func TestInitWeightData_InitOneScalars(t *testing.T) {
	weights := initWeightData([]WeightShape{
		{Name: "attn_scale", Shape: []int{4}, InitOne: true},
		{Name: "mlp_scale", Shape: []int{4}, InitOne: true},
		{Name: "bigram_scale", Shape: []int{1}, InitOne: true},
		{Name: "skip_weight_0", Shape: []int{4}, InitOne: true},
		{Name: "final_norm", Shape: []int{4}, IsNormScale: true, InitOne: true},
		{Name: "bias_like", Shape: []int{4}},
	}, 123, "", 0)

	for i := 0; i < 5; i++ {
		for _, v := range weights[i] {
			if v != 1.0 {
				t.Fatalf("weight %d expected init 1.0, got %v", i, weights[i])
			}
		}
	}
	for _, v := range weights[5] {
		if v != 0.0 {
			t.Fatalf("bias_like expected zero init, got %v", weights[5])
		}
	}
}

func TestInitWeightData_InitValue(t *testing.T) {
	weights := initWeightData([]WeightShape{
		{Name: "qk_gain", Shape: []int{4}, InitValue: 5.25},
	}, 123, "", 0)
	for _, v := range weights[0] {
		if v != 5.25 {
			t.Fatalf("qk_gain expected init 5.25, got %v", weights[0])
		}
	}
}

func TestInitWeightData_NormalInit(t *testing.T) {
	shapes := []WeightShape{
		{Name: "norm", Shape: []int{8}, IsNormScale: true},
		{Name: "wq", Shape: []int{8, 8}},
	}

	// Normal init with std=0.02
	weights := initWeightData(shapes, 42, "normal", 0.02)

	// Norm scale should still be ones
	for _, v := range weights[0] {
		if v != 1.0 {
			t.Fatalf("norm scale should be 1.0 with normal init, got %v", v)
		}
	}

	// Matrix weights should be non-zero and small (drawn from N(0, 0.02))
	var sum, sumSq float64
	for _, v := range weights[1] {
		sum += float64(v)
		sumSq += float64(v) * float64(v)
	}
	n := float64(len(weights[1]))
	mean := sum / n
	variance := sumSq/n - mean*mean
	std := math.Sqrt(variance)

	// Mean should be near zero, std should be near 0.02
	if math.Abs(mean) > 0.05 {
		t.Errorf("normal init mean = %g, expected near 0", mean)
	}
	if math.Abs(std-0.02) > 0.02 {
		t.Errorf("normal init std = %g, expected near 0.02", std)
	}
}

func TestInitWeightData_NormalInitDefaultStd(t *testing.T) {
	shapes := []WeightShape{
		{Name: "wq", Shape: []int{64, 64}},
	}

	// std=0 should default to 0.02
	weights := initWeightData(shapes, 42, "normal", 0)

	var sumSq float64
	for _, v := range weights[0] {
		sumSq += float64(v) * float64(v)
	}
	n := float64(len(weights[0]))
	std := math.Sqrt(sumSq / n)

	if math.Abs(std-0.02) > 0.01 {
		t.Errorf("default normal init std = %g, expected near 0.02", std)
	}
}

func TestInitWeightData_XavierIsDefault(t *testing.T) {
	shapes := []WeightShape{
		{Name: "wq", Shape: []int{8, 8}},
	}

	xavier := initWeightData(shapes, 42, "", 0)
	xavierExplicit := initWeightData(shapes, 42, "xavier_uniform", 0)

	// Empty string and "xavier_uniform" should produce identical results
	for i := range xavier[0] {
		if xavier[0][i] != xavierExplicit[0][i] {
			t.Fatalf("xavier default != xavier_uniform at %d: %g vs %g", i, xavier[0][i], xavierExplicit[0][i])
		}
	}
}
