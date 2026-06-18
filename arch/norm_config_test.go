package arch

import (
	"strings"
	"testing"
)

func TestConfigurableLayerNormNoAffineSandwichBuildsWithoutNormWeights(t *testing.T) {
	cfg, err := ParseArchConfig([]byte(`{
		"name": "ln_no_affine_sandwich",
		"model_dim": 8,
		"vocab_size": 16,
		"seq_len": 4,
		"mlp_mult": 1.0,
		"norm_type": "layernorm",
		"norm_affine": false,
		"norm_eps": 1e-7,
		"norm_placement": "sandwich",
		"ffn_internal_norm": true,
		"blocks": [
			{"type": "plain", "heads": 2},
			{"type": "geglu"},
			{"type": "mlp", "activation": "gelu"}
		],
		"training": {"steps": 1, "batch_tokens": 4}
	}`), "ln_no_affine_sandwich")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	metas, err := CollectWeightShapesFromConfig(cfg)
	if err != nil {
		t.Fatalf("CollectWeightShapesFromConfig: %v", err)
	}
	for _, meta := range metas {
		if meta.Name == "final_norm" || meta.Name == "norm_scale" || meta.Name == "ffn_norm_scale" ||
			meta.Name == "norm_bias" || meta.Name == "ffn_norm_bias" || meta.Name == "post_ffn_norm_bias" {
			t.Fatalf("no-affine LayerNorm should not create norm weight %q in %+v", meta.Name, metas)
		}
	}
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}
	if prog.NumWeights != len(metas) {
		t.Fatalf("program weights=%d metas=%d", prog.NumWeights, len(metas))
	}
	layerNorms := 0
	rmsNorms := 0
	for _, op := range prog.Ops {
		switch op.Code {
		case OpLayerNorm:
			layerNorms++
			if len(op.Inputs) != 1 {
				t.Fatalf("no-affine LayerNorm op has inputs=%v, want one input", op.Inputs)
			}
			if len(op.FloatParams) != 1 || op.FloatParams[0] != 1e-7 {
				t.Fatalf("LayerNorm params=%v, want eps 1e-7", op.FloatParams)
			}
		case OpRMSNorm:
			rmsNorms++
		}
	}
	if layerNorms == 0 {
		t.Fatal("expected configurable LayerNorm ops")
	}
	if rmsNorms != 0 {
		t.Fatalf("expected no RMSNorm ops, found %d", rmsNorms)
	}
}

func TestConfigurableNormValidationRejectsUnsupportedBlocks(t *testing.T) {
	_, err := ParseArchConfig([]byte(`{
		"model_dim": 8,
		"vocab_size": 16,
		"seq_len": 4,
		"norm_type": "layernorm",
		"blocks": [{"type": "hgrn2", "heads": 2}],
		"training": {"steps": 1, "batch_tokens": 4}
	}`), "bad_norm_block")
	if err == nil {
		t.Fatal("ParseArchConfig succeeded")
	}
	if got := err.Error(); !containsAll(got, []string{"non-default norm", "hgrn2"}) {
		t.Fatalf("error=%v, want non-default norm hgrn2 rejection", err)
	}
}

func TestConfigurableNormValidationRejectsRMSNoAffine(t *testing.T) {
	falseValue := false
	cfg := &ArchConfig{
		Name:       "bad_rms_affine",
		ModelDim:   8,
		VocabSize:  16,
		SeqLen:     4,
		NormType:   "rmsnorm",
		NormAffine: &falseValue,
		Blocks:     []BlockSpec{{Type: "plain", Heads: 2}},
		Training:   TrainingSpec{Steps: 1, BatchTokens: 4},
	}
	if _, err := validateConfig(cfg, "bad_rms_affine"); err == nil {
		t.Fatal("validateConfig succeeded")
	}
}

func containsAll(s string, needles []string) bool {
	for _, needle := range needles {
		if !strings.Contains(s, needle) {
			return false
		}
	}
	return true
}
