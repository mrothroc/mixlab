package arch

import (
	"reflect"
	"testing"
)

// TestBlockWeightShapes_CountMatchesBlockWeightCount verifies that
// blockWeightShapes returns the same number of weights as BlockWeightCount
// for every supported block type.
func TestBlockWeightShapes_CountMatchesBlockWeightCount(t *testing.T) {
	D := 64
	T := 128
	B := 1
	V := 256

	specs := []BlockSpec{
		{Type: "plain", Heads: 4},
		{Type: "plain", Heads: 4, QKGain: 5.25},
		{Type: "plain", Heads: 8, SparseAttnGate: true},
		{Type: "swiglu"},
		{Type: "mlp"},
		{Type: "mamba", InnerDim: 32},
		{Type: "mamba"}, // default inner = D
		{Type: "mamba3", InnerDim: 32},
		{Type: "mamba3"}, // default inner = D
		{Type: "rwkv"},
		{Type: "token_blend"},
		{Type: "perceiver", Heads: 4, NumLatents: 16},
		{Type: "bottleneck", Heads: 4, NumLatents: 4},
		{Type: "retnet", Heads: 4},
		{Type: "cross_attention", Heads: 4, SourceStream: "other"},
		{Type: "custom", Name: "test", Weights: []WeightSpec{
			{Name: "w1", Shape: []string{"D", "D"}},
			{Name: "w2", Shape: []string{"D"}},
		}, Ops: []OpSpec{
			{Op: "matmul", Inputs: []string{"x", "w1"}, Output: "y"},
		}},
	}

	for _, spec := range specs {
		t.Run(spec.Type, func(t *testing.T) {
			wantCount, err := BlockWeightCount(spec, false, false)
			if err != nil {
				t.Fatalf("BlockWeightCount: %v", err)
			}

			metas, err := blockWeightShapes(spec, D, T, B, V, DefaultFFNMultiplier, false, false)
			if err != nil {
				t.Fatalf("blockWeightShapes: %v", err)
			}

			if len(metas) != wantCount {
				t.Errorf("blockWeightShapes returned %d weights, BlockWeightCount says %d",
					len(metas), wantCount)
			}

			// Every shape should have positive dimensions
			for i, m := range metas {
				if len(m.Shape) == 0 {
					t.Errorf("weight %d (%s) has empty shape", i, m.Name)
					continue
				}
				for _, d := range m.Shape {
					if d <= 0 {
						t.Errorf("weight %d (%s) has non-positive dimension %d in shape %v",
							i, m.Name, d, m.Shape)
					}
				}
			}
		})
	}
}

func TestBlockWeightShapes_MLP(t *testing.T) {
	metas, err := blockWeightShapes(BlockSpec{Type: "mlp"}, 64, 32, 1, 256, 2.0, false, false)
	if err != nil {
		t.Fatalf("blockWeightShapes: %v", err)
	}
	want := []WeightMeta{
		{Name: "ffn_norm_scale", Shape: []int{64}, IsNormScale: true, InitOne: true},
		{Name: "w_up", Shape: []int{64, 128}},
		{Name: "w_down", Shape: []int{128, 64}},
	}
	if !reflect.DeepEqual(metas, want) {
		t.Fatalf("mlp weight shapes = %+v, want %+v", metas, want)
	}
}

func TestBlockWeightShapes_QKGainMeta(t *testing.T) {
	metas, err := blockWeightShapes(BlockSpec{Type: "plain", Heads: 4, QKGain: 5.25}, 64, 32, 1, 256, DefaultFFNMultiplier, false, false)
	if err != nil {
		t.Fatalf("blockWeightShapes: %v", err)
	}
	if len(metas) != 8 {
		t.Fatalf("weight count = %d, want 8", len(metas))
	}
	if got, want := metas[4].Name, "qk_gain"; got != want {
		t.Fatalf("weight[4].Name = %q, want %q", got, want)
	}
	if got, want := metas[4].Shape, []int{4}; !reflect.DeepEqual(got, want) {
		t.Fatalf("qk_gain shape = %v, want %v", got, want)
	}
	if got, want := metas[4].InitValue, float32(5.25); got != want {
		t.Fatalf("qk_gain InitValue = %g, want %g", got, want)
	}
	if metas[4].IsNormScale || metas[4].InitOne {
		t.Fatalf("qk_gain should use InitValue only, got %+v", metas[4])
	}
}

func TestBlockWeightShapes_SparseAttnGateMeta(t *testing.T) {
	metas, err := blockWeightShapes(BlockSpec{Type: "plain", Heads: 8, SparseAttnGate: true}, 384, 32, 1, 256, DefaultFFNMultiplier, false, false)
	if err != nil {
		t.Fatalf("blockWeightShapes: %v", err)
	}
	if len(metas) != 8 {
		t.Fatalf("weight count = %d, want 8", len(metas))
	}
	if got, want := metas[4].Name, "attn_gate_w"; got != want {
		t.Fatalf("weight[4].Name = %q, want %q", got, want)
	}
	if got, want := metas[4].Shape, []int{8, 12}; !reflect.DeepEqual(got, want) {
		t.Fatalf("attn_gate_w shape = %v, want %v", got, want)
	}
	if !metas[4].InitZero {
		t.Fatalf("attn_gate_w should be zero-initialized, got %+v", metas[4])
	}
	total := 1
	for _, d := range metas[4].Shape {
		total *= d
	}
	if total != 96 {
		t.Fatalf("attn_gate_w params = %d, want 96", total)
	}
}

func TestBlockWeightShapes_KVSourceOmitsWKAndWV(t *testing.T) {
	metas, err := blockWeightShapes(BlockSpec{Type: "plain", Heads: 8, KVHeads: 4, KVSource: 1}, 128, 32, 1, 256, DefaultFFNMultiplier, false, false)
	if err != nil {
		t.Fatalf("blockWeightShapes: %v", err)
	}
	if len(metas) != 5 {
		t.Fatalf("weight count=%d want 5", len(metas))
	}
	for _, meta := range metas {
		if meta.Name == "wk" || meta.Name == "wv" {
			t.Fatalf("unexpected shared-KV weight %q in %+v", meta.Name, metas)
		}
	}
}

// TestCollectWeightShapes_CountMatchesCountWeights verifies the total count
// matches for several architecture configurations.
func TestCollectWeightShapes_CountMatchesCountWeights(t *testing.T) {
	tests := []struct {
		name   string
		blocks []BlockSpec
	}{
		{
			name:   "plain_3L",
			blocks: []BlockSpec{{Type: "plain", Heads: 4}, {Type: "plain", Heads: 4}, {Type: "plain", Heads: 4}},
		},
		{
			name:   "mixed",
			blocks: []BlockSpec{{Type: "plain", Heads: 4}, {Type: "swiglu"}, {Type: "mamba3"}},
		},
		{
			name:   "rwkv",
			blocks: []BlockSpec{{Type: "rwkv"}},
		},
		{
			name:   "retnet",
			blocks: []BlockSpec{{Type: "retnet", Heads: 4}},
		},
	}

	D := 64
	V := 256
	T := 128

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			wantCount, err := CountWeights(DefaultFFNMultiplier, false, false, false, false, tc.blocks)
			if err != nil {
				t.Fatalf("CountWeights: %v", err)
			}

			metas, err := CollectWeightShapes(D, V, T, DefaultFFNMultiplier, false, false, false, false, tc.blocks)
			if err != nil {
				t.Fatalf("CollectWeightShapes: %v", err)
			}

			if len(metas) != wantCount {
				t.Errorf("CollectWeightShapes returned %d, CountWeights says %d",
					len(metas), wantCount)
			}
		})
	}
}

// TestCollectWeightShapes_NormScaleFlags verifies that IsNormScale is set
// correctly for known weight types.
func TestCollectWeightShapes_NormScaleFlags(t *testing.T) {
	D := 64
	V := 256
	T := 128

	metas, err := CollectWeightShapes(D, V, T, DefaultFFNMultiplier, false, false, false, false,
		[]BlockSpec{{Type: "plain", Heads: 4}, {Type: "swiglu"}})
	if err != nil {
		t.Fatalf("CollectWeightShapes: %v", err)
	}

	// Check specific weights
	normScaleNames := map[string]bool{
		"final_norm":     true,
		"norm_scale":     true,
		"ffn_norm_scale": true,
	}

	for _, m := range metas {
		if normScaleNames[m.Name] && !m.IsNormScale {
			t.Errorf("weight %q should have IsNormScale=true", m.Name)
		}
		if m.Name == "wq" && m.IsNormScale {
			t.Errorf("weight %q should have IsNormScale=false", m.Name)
		}
	}
}

// TestCollectWeightShapes_Errors verifies error cases.
func TestCollectWeightShapes_Errors(t *testing.T) {
	_, err := CollectWeightShapes(0, 256, 128, DefaultFFNMultiplier, false, false, false, false,
		[]BlockSpec{{Type: "plain", Heads: 4}})
	if err == nil {
		t.Error("expected error for model_dim=0")
	}

	_, err = CollectWeightShapes(64, 0, 128, DefaultFFNMultiplier, false, false, false, false,
		[]BlockSpec{{Type: "plain", Heads: 4}})
	if err == nil {
		t.Error("expected error for vocab_size=0")
	}

	_, err = CollectWeightShapes(64, 256, 128, DefaultFFNMultiplier, false, false, false, false,
		[]BlockSpec{{Type: "unknown"}})
	if err == nil {
		t.Error("expected error for unsupported block type")
	}
}

func TestCollectWeightShapesWithBigram_AddsExpectedWeights(t *testing.T) {
	metas, err := CollectWeightShapesWithBigram(64, 256, 32, DefaultFFNMultiplier, false, false, false, false, 257, 16,
		[]BlockSpec{{Type: "plain", Heads: 4}})
	if err != nil {
		t.Fatalf("CollectWeightShapesWithBigram: %v", err)
	}
	if len(metas) != 13 {
		t.Fatalf("expected 13 weights, got %d", len(metas))
	}
	if metas[3].Name != "bigram_table" || metas[3].Shape[0] != 257 || metas[3].Shape[1] != 16 {
		t.Fatalf("unexpected bigram table meta: %+v", metas[3])
	}
	if metas[4].Name != "bigram_proj" || metas[4].Shape[0] != 16 || metas[4].Shape[1] != 64 {
		t.Fatalf("unexpected bigram proj meta: %+v", metas[4])
	}
	if metas[5].Name != "bigram_scale" || len(metas[5].Shape) != 1 || metas[5].Shape[0] != 1 {
		t.Fatalf("unexpected bigram scale meta: %+v", metas[5])
	}
}

func TestCollectWeightShapesWithBigram_DisabledMatchesBase(t *testing.T) {
	base, err := CollectWeightShapes(64, 256, 32, DefaultFFNMultiplier, false, false, false, false,
		[]BlockSpec{{Type: "plain", Heads: 4}})
	if err != nil {
		t.Fatalf("CollectWeightShapes: %v", err)
	}
	withBigramDisabled, err := CollectWeightShapesWithBigram(64, 256, 32, DefaultFFNMultiplier, false, false, false, false, 0, 0,
		[]BlockSpec{{Type: "plain", Heads: 4}})
	if err != nil {
		t.Fatalf("CollectWeightShapesWithBigram: %v", err)
	}
	if len(base) != len(withBigramDisabled) {
		t.Fatalf("disabled bigram changed weight count: base=%d disabled=%d", len(base), len(withBigramDisabled))
	}
}

func TestCollectWeightShapes_TiedEmbeddingsOmitsHead(t *testing.T) {
	metas, err := CollectWeightShapes(64, 256, 32, DefaultFFNMultiplier, true, false, false, false,
		[]BlockSpec{{Type: "plain", Heads: 4}})
	if err != nil {
		t.Fatalf("CollectWeightShapes: %v", err)
	}
	if len(metas) != 9 {
		t.Fatalf("expected 9 metas with tied embeddings, got %d", len(metas))
	}
	if metas[0].Name != "embed" {
		t.Fatalf("first meta = %q, want embed", metas[0].Name)
	}
	if metas[1].Name != "final_norm" {
		t.Fatalf("second meta = %q, want final_norm", metas[1].Name)
	}
	for _, meta := range metas {
		if meta.Name == "head" {
			t.Fatal("unexpected head meta with tied embeddings")
		}
	}
}

func TestCollectWeightShapes_WithBlockScalesAndResidMix(t *testing.T) {
	metas, err := CollectWeightShapes(64, 256, 32, DefaultFFNMultiplier, false, true, true, false,
		[]BlockSpec{{Type: "plain", Heads: 4}, {Type: "swiglu"}})
	if err != nil {
		t.Fatalf("CollectWeightShapes: %v", err)
	}
	if metas[3].Name != "resid_mix" {
		t.Fatalf("expected first plain weight to be resid_mix, got %q", metas[3].Name)
	}
	foundAttnScale := false
	foundMLPScale := 0
	for _, meta := range metas {
		if meta.Name == "attn_scale" {
			foundAttnScale = true
		}
		if meta.Name == "mlp_scale" {
			foundMLPScale++
		}
	}
	if !foundAttnScale {
		t.Fatal("missing attn_scale meta")
	}
	if foundMLPScale != 2 {
		t.Fatalf("expected 2 mlp_scale metas, got %d", foundMLPScale)
	}
}

func TestCollectWeightShapes_UNetAddsSkipWeights(t *testing.T) {
	metas, err := CollectWeightShapes(64, 256, 32, DefaultFFNMultiplier, false, false, false, true,
		[]BlockSpec{{Type: "plain", Heads: 4}, {Type: "plain", Heads: 4}})
	if err != nil {
		t.Fatalf("CollectWeightShapes unet: %v", err)
	}
	if len(metas) != 18 {
		t.Fatalf("expected 18 metas, got %d", len(metas))
	}
	if metas[10].Name != "skip_weight_0" {
		t.Fatalf("expected skip weight at index 10, got %q", metas[10].Name)
	}
}

func TestCollectWeightShapes_CustomMLPMultChangesFFNShapes(t *testing.T) {
	defaultMetas, err := CollectWeightShapes(64, 256, 32, DefaultFFNMultiplier, false, false, false, false,
		[]BlockSpec{{Type: "plain", Heads: 4}, {Type: "swiglu"}})
	if err != nil {
		t.Fatalf("CollectWeightShapes default: %v", err)
	}
	customMetas, err := CollectWeightShapes(64, 256, 32, 4.0, false, false, false, false,
		[]BlockSpec{{Type: "plain", Heads: 4}, {Type: "swiglu"}})
	if err != nil {
		t.Fatalf("CollectWeightShapes custom: %v", err)
	}

	if got, want := defaultMetas[8].Shape, []int{64, 171}; len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("default plain ff1 shape = %v, want %v", got, want)
	}
	if got, want := customMetas[8].Shape, []int{64, 256}; len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("custom plain ff1 shape = %v, want %v", got, want)
	}
	if got, want := customMetas[11].Shape, []int{64, 256}; len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("custom swiglu gate shape = %v, want %v", got, want)
	}
}

func TestCollectWeightShapes_LeaderOrderingRegression(t *testing.T) {
	blocks := []BlockSpec{
		{Type: "token_blend"},
		{Type: "plain", Heads: 8, KVHeads: 4},
		{Type: "swiglu"},
		{Type: "plain", Heads: 8, KVHeads: 4},
		{Type: "swiglu"},
		{Type: "plain", Heads: 8, KVHeads: 4},
		{Type: "swiglu"},
		{Type: "plain", Heads: 8, KVHeads: 4},
		{Type: "swiglu"},
		{Type: "plain", Heads: 8, KVHeads: 4},
		{Type: "swiglu"},
		{Type: "plain", Heads: 8, KVHeads: 4},
		{Type: "swiglu"},
		{Type: "plain", Heads: 8, KVHeads: 4},
		{Type: "swiglu"},
		{Type: "plain", Heads: 8, KVHeads: 4},
		{Type: "swiglu"},
		{Type: "plain", Heads: 8, KVHeads: 4},
		{Type: "swiglu"},
	}

	metas, err := CollectWeightShapesWithBigram(512, 1024, 2048, 3.0, true, true, true, true, 4096, 128, blocks)
	if err != nil {
		t.Fatalf("CollectWeightShapesWithBigram: %v", err)
	}

	if len(metas) != 150 {
		t.Fatalf("expected 150 weights, got %d", len(metas))
	}

	cases := map[int]string{
		0:   "embed",
		1:   "final_norm",
		2:   "bigram_table",
		3:   "bigram_proj",
		4:   "bigram_scale",
		5:   "w_gate",
		6:   "resid_mix",
		15:  "mlp_scale",
		16:  "ffn_norm_scale",
		20:  "mlp_scale",
		66:  "skip_weight_0",
		74:  "skip_weight_8",
		75:  "resid_mix",
		84:  "mlp_scale",
		85:  "ffn_norm_scale",
		149: "mlp_scale",
	}
	for idx, want := range cases {
		if got := metas[idx].Name; got != want {
			t.Fatalf("meta[%d] = %q, want %q", idx, got, want)
		}
	}

	for _, idx := range []int{4, 15, 20, 66, 74, 84, 149} {
		if !metas[idx].InitOne {
			t.Fatalf("meta[%d] (%q) should initialize to 1.0", idx, metas[idx].Name)
		}
	}
}
