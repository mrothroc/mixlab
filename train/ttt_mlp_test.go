package train

import (
	"math"
	"testing"
)

func TestTTTMLPSpecialInitializersMatchReferenceDistributions(t *testing.T) {
	weights := initWeightData([]WeightShape{
		{Name: "linear", Shape: []int{4096}, InitMode: "ttt_normal_0_02"},
		{Name: "conv", Shape: []int{4096}, InitMode: "ttt_conv_uniform_4"},
	}, 1234, "normal", 1)

	var sum, sq float64
	for _, value := range weights[0] {
		sum += float64(value)
	}
	mean := sum / float64(len(weights[0]))
	for _, value := range weights[0] {
		delta := float64(value) - mean
		sq += delta * delta
	}
	std := math.Sqrt(sq / float64(len(weights[0])))
	if math.Abs(mean) > 0.003 || std < 0.018 || std > 0.022 {
		t.Fatalf("linear normal mean=%g std=%g, want N(0,0.02)", mean, std)
	}
	for i, value := range weights[1] {
		if value < -0.5 || value > 0.5 {
			t.Fatalf("conv[%d]=%g outside PyTorch depthwise Conv1d bound", i, value)
		}
	}
}
