package train

import "testing"

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
	}, 123)

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
