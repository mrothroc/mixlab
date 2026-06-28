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
		{Name: "backout_lambda", Shape: []int{1}, InitValue: -1.0},
	}, 123, "", 0)
	for _, v := range weights[0] {
		if v != 5.25 {
			t.Fatalf("qk_gain expected init 5.25, got %v", weights[0])
		}
	}
	for _, v := range weights[1] {
		if v != -1.0 {
			t.Fatalf("backout_lambda expected init -1.0, got %v", weights[1])
		}
	}
}

func TestInitWeightData_DWAAlphaInitializesToLastState(t *testing.T) {
	weights := initWeightData([]WeightShape{
		{Name: "dwa_alpha_0", Shape: []int{2}, InitMode: "dwa_alpha"},
		{Name: "dwa_alpha_1", Shape: []int{3}, InitMode: "dwa_alpha"},
	}, 123, "", 0)
	for i, got := range weights {
		for j, v := range got {
			want := float32(0)
			if j == len(got)-1 {
				want = 1
			}
			if v != want {
				t.Fatalf("weights[%d][%d]=%g, want %g; full=%v", i, j, v, want, got)
			}
		}
	}
}

func TestInitWeightData_Mamba3CanonicalSpecialInits(t *testing.T) {
	weights := initWeightData([]WeightShape{
		{Name: "A_log", Shape: []int{2, 4}, InitLogArange: true},
		{Name: "dt_bias", Shape: []int{8}, InitDtBias: true, DtMin: 0.001, DtMax: 0.1},
	}, 123, "", 0)
	wantALog := []float32{0, float32(math.Log(2)), float32(math.Log(3)), float32(math.Log(4)), 0, float32(math.Log(2)), float32(math.Log(3)), float32(math.Log(4))}
	for i, want := range wantALog {
		if math.Abs(float64(weights[0][i]-want)) > 1e-6 {
			t.Fatalf("A_log[%d]=%g want %g", i, weights[0][i], want)
		}
	}
	for i, raw := range weights[1] {
		dt := math.Log1p(math.Exp(float64(raw)))
		if dt < 0.001 || dt > 0.1 {
			t.Fatalf("dt_bias[%d] softplus=%g out of range", i, dt)
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

func TestComputeWeightShapesMarksGPTBERTOutputProjections(t *testing.T) {
	cfg, err := ParseArchConfig([]byte(`{
		"name": "gptbert_init_shapes",
		"model_dim": 32,
		"vocab_size": 128,
		"seq_len": 8,
		"blocks": [
			{"type": "plain", "heads": 4},
			{"type": "geglu"}
		],
		"training": {"steps": 10, "lr": 0.001, "batch_tokens": 16, "weight_init": "gptbert"}
	}`), "gptbert_init_shapes")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatalf("computeWeightShapes: %v", err)
	}
	scales := map[string]float32{}
	for _, ws := range shapes {
		if ws.GPTBERTScale > 0 {
			scales[ws.Name] = ws.GPTBERTScale
		}
	}
	wantPlainScale := float32(math.Sqrt(1.0 / 2.0))
	wantGEGLUScale := float32(math.Sqrt(1.0 / 4.0))
	if math.Abs(float64(scales["wo"]-wantPlainScale)) > 1e-6 {
		t.Fatalf("wo GPTBERTScale=%g, want %g", scales["wo"], wantPlainScale)
	}
	if math.Abs(float64(scales["ff2"]-wantPlainScale)) > 1e-6 {
		t.Fatalf("ff2 GPTBERTScale=%g, want %g", scales["ff2"], wantPlainScale)
	}
	if math.Abs(float64(scales["w_down"]-wantGEGLUScale)) > 1e-6 {
		t.Fatalf("w_down GPTBERTScale=%g, want %g", scales["w_down"], wantGEGLUScale)
	}
	if _, ok := scales["w_gate"]; ok {
		t.Fatalf("w_gate should not be depth-scaled")
	}
}

func TestComputeWeightShapesMarksGPT2ResidualProjectionScale(t *testing.T) {
	cfg, err := ParseArchConfig([]byte(`{
		"name": "gpt2_init_shapes",
		"model_dim": 32,
		"vocab_size": 128,
		"seq_len": 8,
		"blocks": [
			{"type": "plain", "heads": 4},
			{"type": "plain", "heads": 4},
			{"type": "plain", "heads": 4}
		],
		"training": {"steps": 10, "lr": 0.001, "batch_tokens": 16, "weight_init": "gpt2"}
	}`), "gpt2_init_shapes")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatalf("computeWeightShapes: %v", err)
	}
	want := float32(1.0 / math.Sqrt(2.0*3.0))
	residuals := 0
	for _, ws := range shapes {
		switch ws.Name {
		case "wo", "ff2":
			residuals++
			if math.Abs(float64(ws.GPT2Scale-want)) > 1e-6 {
				t.Fatalf("%s GPT2Scale=%g, want %g", ws.Name, ws.GPT2Scale, want)
			}
		case "wq", "wk", "wv", "ff1":
			if ws.GPT2Scale != 0 {
				t.Fatalf("%s unexpectedly had GPT2Scale=%g", ws.Name, ws.GPT2Scale)
			}
		}
	}
	if residuals != 6 {
		t.Fatalf("marked GPT-2 residual projections=%d, want 6", residuals)
	}
}

func TestInitWeightData_GPTBERTTruncatedDepthScaled(t *testing.T) {
	shapes := []WeightShape{
		{Name: "wq", Shape: []int{512, 512}, ModelDim: 512},
		{Name: "wo", Shape: []int{512, 512}, ModelDim: 512, GPTBERTScale: 0.5},
	}
	weights := initWeightData(shapes, 42, "gptbert", 0)
	baseStd := math.Sqrt(2.0 / (5.0 * 512.0))
	for i, v := range weights[0] {
		if math.Abs(float64(v)) > 2*baseStd+1e-7 {
			t.Fatalf("unscaled gptbert weight[%d]=%g outside truncated bound", i, v)
		}
	}
	for i, v := range weights[1] {
		if math.Abs(float64(v)) > baseStd+1e-7 {
			t.Fatalf("scaled gptbert weight[%d]=%g outside scaled truncated bound", i, v)
		}
	}
	ratio := rms(weights[1]) / rms(weights[0])
	if math.Abs(ratio-0.5) > 0.04 {
		t.Fatalf("scaled/unscaled RMS ratio=%g, want about 0.5", ratio)
	}
}

func TestInitWeightData_GPT2NormalAndResidualScaled(t *testing.T) {
	scale := float32(1.0 / math.Sqrt(2.0*12.0))
	shapes := []WeightShape{
		{Name: "norm", Shape: []int{64}, IsNormScale: true},
		{Name: "bias", Shape: []int{64}},
		{Name: "embed", Shape: []int{512, 512}},
		{Name: "wq", Shape: []int{512, 512}},
		{Name: "wo", Shape: []int{512, 512}, GPT2Scale: scale},
	}
	weights := initWeightData(shapes, 42, "gpt2", 0)
	for _, v := range weights[0] {
		if v != 1.0 {
			t.Fatalf("norm scale should be 1.0 with gpt2 init, got %v", v)
		}
	}
	for _, v := range weights[1] {
		if v != 0 {
			t.Fatalf("bias should be zero with gpt2 init, got %v", v)
		}
	}
	for _, idx := range []int{2, 3} {
		if got := rms(weights[idx]); math.Abs(got-0.02) > 0.0015 {
			t.Fatalf("%s RMS=%g, want about 0.02", shapes[idx].Name, got)
		}
	}
	ratio := rms(weights[4]) / rms(weights[3])
	if math.Abs(ratio-float64(scale)) > 0.006 {
		t.Fatalf("residual/base RMS ratio=%g, want about %g", ratio, scale)
	}
}

func rms(xs []float32) float64 {
	var sumSq float64
	for _, x := range xs {
		sumSq += float64(x) * float64(x)
	}
	return math.Sqrt(sumSq / float64(len(xs)))
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
