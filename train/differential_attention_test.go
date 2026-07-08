package train

import (
	"math"
	"testing"
)

func TestDifferentialLambdaInitModeUsesNormalPointOne(t *testing.T) {
	weights := initWeightData([]WeightShape{{
		Name:     "diff_lambda_q1",
		Shape:    []int{4096},
		InitMode: "diff_lambda_normal_0_1",
	}}, 1234, "gptbert", 0)
	if len(weights) != 1 {
		t.Fatalf("weights len=%d want 1", len(weights))
	}
	var sum float64
	for _, v := range weights[0] {
		sum += float64(v)
	}
	mean := sum / float64(len(weights[0]))
	var sq float64
	for _, v := range weights[0] {
		d := float64(v) - mean
		sq += d * d
	}
	std := math.Sqrt(sq / float64(len(weights[0])))
	if math.Abs(mean) > 0.01 {
		t.Fatalf("mean=%g want near 0", mean)
	}
	if std < 0.085 || std > 0.115 {
		t.Fatalf("std=%g want near 0.1", std)
	}
}
