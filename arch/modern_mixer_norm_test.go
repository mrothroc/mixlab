package arch

import (
	"reflect"
	"strings"
	"testing"
)

func modernMixerNormConfig(t *testing.T, normFields, block string) *ArchConfig {
	t.Helper()
	raw := `{
		"name":"modern_mixer_norm",
		"model_dim":8,
		"vocab_size":16,
		"seq_len":4,
		"positional_embedding":"none",
		` + normFields + `
		"blocks":[` + block + `],
		"training":{
			"objective":"classification",
			"classification":{"num_labels":2,"pooling":"mean"},
			"batch_tokens":8,
			"steps":2,
			"lr":0.001
		}
	}`
	cfg, err := ParseArchConfig([]byte(raw), t.Name())
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	return cfg
}

func TestModernRecurrentMixersAcceptConfigurableOuterPreNorm(t *testing.T) {
	blocks := []struct {
		name  string
		block string
	}{
		{"gated_deltanet", `{"type":"gated_deltanet","heads":2,"d_k":2,"d_v":4,"bidirectional":true}`},
		{"mamba3_canonical", `{"type":"mamba3-canonical","inner_dim":8,"state_size":4,"n_groups":2,"dt_rank":2,"use_conv":false,"bidirectional":true}`},
	}
	norms := []struct {
		name   string
		fields string
		typeOp int
	}{
		{"batchnorm", `"norm_type":"batchnorm","norm_placement":"pre",`, OpBatchNorm},
		{"layernorm", `"norm_type":"layernorm","norm_placement":"pre",`, OpLayerNorm},
		{"layernorm_no_affine", `"norm_type":"layernorm","norm_affine":false,"norm_placement":"pre",`, OpLayerNorm},
		{"custom_rms_eps", `"norm_type":"rmsnorm","norm_eps":0.0001,"norm_placement":"pre",`, OpRMSNorm},
	}
	for _, block := range blocks {
		for _, norm := range norms {
			t.Run(block.name+"/"+norm.name, func(t *testing.T) {
				cfg := modernMixerNormConfig(t, norm.fields, block.block)
				prog, err := BuildIRProgramFromConfig(cfg)
				if err != nil {
					t.Fatalf("BuildIRProgramFromConfig: %v", err)
				}
				if got := countOps(prog, norm.typeOp); got < 2 {
					t.Fatalf("outer block plus final norm op count=%d want at least 2", got)
				}
			})
		}
	}
}

func TestModernMixerBatchNormWeightLayoutsPreserveInternalNorms(t *testing.T) {
	cases := []struct {
		name            string
		block           string
		outerPrefix     string
		internalScales  []string
		mambaExternalOp bool
	}{
		{
			name:           "gated_deltanet",
			block:          `{"type":"gated_deltanet","heads":2,"d_k":2,"d_v":4,"bidirectional":true}`,
			outerPrefix:    "norm",
			internalScales: []string{"o_norm_scale"},
		},
		{
			name:            "mamba3_canonical",
			block:           `{"type":"mamba3-canonical","inner_dim":8,"state_size":4,"n_groups":2,"dt_rank":2,"use_conv":false,"bidirectional":true}`,
			outerPrefix:     "pre_norm",
			internalScales:  []string{"B_norm_scale", "C_norm_scale", "post_norm_scale"},
			mambaExternalOp: true,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := modernMixerNormConfig(t, `"norm_type":"batchnorm","norm_placement":"pre",`, tc.block)
			metas, err := CollectWeightShapesFromConfig(cfg)
			if err != nil {
				t.Fatal(err)
			}
			wantOuter := map[string]bool{
				tc.outerPrefix + "_scale":        false,
				tc.outerPrefix + "_bias":         false,
				tc.outerPrefix + "_running_mean": false,
				tc.outerPrefix + "_running_var":  false,
			}
			for _, meta := range metas {
				if _, ok := wantOuter[meta.Name]; ok {
					wantOuter[meta.Name] = true
					if strings.Contains(meta.Name, "running_") && !meta.IsBuffer {
						t.Fatalf("%s must be a persistent buffer", meta.Name)
					}
				}
			}
			for name, found := range wantOuter {
				if !found {
					t.Fatalf("missing outer BatchNorm weight %q in %+v", name, metas)
				}
			}
			for _, name := range tc.internalScales {
				found := false
				for _, meta := range metas {
					if meta.Name == name {
						found = true
						if meta.IsBuffer || !meta.IsNormScale {
							t.Fatalf("internal norm %s changed semantics: %+v", name, meta)
						}
					}
				}
				if !found {
					t.Fatalf("missing internal norm scale %q", name)
				}
			}

			prog, err := BuildIRProgramFromConfig(cfg)
			if err != nil {
				t.Fatal(err)
			}
			if tc.mambaExternalOp {
				found := false
				for _, op := range prog.Ops {
					if op.Code != OpMamba3CanonicalBlock {
						continue
					}
					found = true
					if len(op.IntParams) != 5 || op.IntParams[4] != 1 {
						t.Fatalf("external-pre-norm params=%v", op.IntParams)
					}
					if len(op.Inputs) < 2 || strings.HasPrefix(op.Inputs[1], "w") {
						t.Fatalf("Mamba external pre-norm input=%v", op.Inputs)
					}
				}
				if !found {
					t.Fatal("missing canonical Mamba op")
				}
			}
		})
	}
}

func TestModernMixerDefaultNormIRAndWeightsRemainLegacy(t *testing.T) {
	blocks := []string{
		`{"type":"gated_deltanet","heads":2,"d_k":2,"d_v":4}`,
		`{"type":"mamba3-canonical","inner_dim":8,"state_size":4,"n_groups":2,"dt_rank":2,"use_conv":false}`,
	}
	for _, block := range blocks {
		t.Run(block, func(t *testing.T) {
			omitted := modernMixerNormConfig(t, "", block)
			explicit := modernMixerNormConfig(t, `"norm_type":"rmsnorm","norm_eps":0.00001,"norm_affine":true,"norm_placement":"pre",`, block)
			omittedMetas, err := CollectWeightShapesFromConfig(omitted)
			if err != nil {
				t.Fatal(err)
			}
			explicitMetas, err := CollectWeightShapesFromConfig(explicit)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(omittedMetas, explicitMetas) {
				t.Fatalf("explicit default changed weight metadata\nomitted=%+v\nexplicit=%+v", omittedMetas, explicitMetas)
			}
			omittedProg, err := BuildIRProgramFromConfig(omitted)
			if err != nil {
				t.Fatal(err)
			}
			explicitProg, err := BuildIRProgramFromConfig(explicit)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(omittedProg, explicitProg) {
				t.Fatal("explicit default norm changed legacy IR")
			}
			for _, op := range omittedProg.Ops {
				if op.Code == OpMamba3CanonicalBlock && len(op.IntParams) != 4 {
					t.Fatalf("legacy Mamba params changed: %v", op.IntParams)
				}
			}
		})
	}
}

func TestModernMixerNormValidationNamesUnsupportedFieldAndBlock(t *testing.T) {
	for _, tc := range []struct {
		name, fields, want string
	}{
		{"post placement", `"norm_type":"layernorm","norm_placement":"post",`, `norm_placement="post"`},
		{"sandwich placement", `"norm_type":"layernorm","norm_placement":"sandwich",`, `norm_placement="sandwich"`},
		{"internal FFN norm", `"norm_type":"layernorm","ffn_internal_norm":true,`, "ffn_internal_norm"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw := `{
				"model_dim":8,"vocab_size":16,"seq_len":4,
				` + tc.fields + `
				"blocks":[{"type":"mamba3-canonical","inner_dim":8,"state_size":4,"n_groups":2,"dt_rank":2,"use_conv":false}],
				"training":{"objective":"classification","classification":{"num_labels":2},"batch_tokens":8}
			}`
			_, err := ParseArchConfig([]byte(raw), tc.name)
			if err == nil || !strings.Contains(err.Error(), tc.want) || !strings.Contains(err.Error(), "mamba3-canonical") {
				t.Fatalf("error=%v want field %q and block type", err, tc.want)
			}
		})
	}
}
