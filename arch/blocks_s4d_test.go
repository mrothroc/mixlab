package arch

import (
	"math"
	"strings"
	"testing"
)

func validS4DConfigJSON() []byte {
	return []byte(`{
		"name": "s4d_tiny",
		"model_dim": 8,
		"vocab_size": 32,
		"seq_len": 16,
		"blocks": [{"type":"s4d"},{"type":"swiglu"}],
		"training": {"objective":"causal","batch_tokens":16,"steps":2,"lr":0.001}
	}`)
}

func TestS4DConfigDefaultsWeightsAndIR(t *testing.T) {
	cfg, err := ParseArchConfig(validS4DConfigJSON(), "s4d-test")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	spec := cfg.Blocks[0]
	if got := effectiveS4DStateSize(spec); got != 64 {
		t.Fatalf("state_size=%d want 64", got)
	}
	if got := effectiveS4DInit(spec); got != S4DInitLin {
		t.Fatalf("init=%q want %q", got, S4DInitLin)
	}
	dtMin, dtMax := effectiveS4DDTRange(spec)
	if dtMin != 0.001 || dtMax != 0.1 {
		t.Fatalf("dt range=[%g,%g] want [0.001,0.1]", dtMin, dtMax)
	}

	metas, err := blockWeightShapes(spec, cfg.ModelDim, cfg.SeqLen, 1, cfg.VocabSize, cfg.EffectiveMLPMult(), false, false)
	if err != nil {
		t.Fatalf("blockWeightShapes: %v", err)
	}
	want := map[string][]int{
		"s4d_norm_scale": {8},
		"s4d_log_dt":     {8},
		"s4d_log_A_real": {8, 32},
		"s4d_A_imag":     {8, 32},
		"s4d_C_real":     {8, 32},
		"s4d_C_imag":     {8, 32},
		"s4d_D":          {8},
	}
	assertWeightShapes(t, metas, want)
	for _, meta := range metas {
		switch meta.Name {
		case "s4d_log_dt":
			if meta.InitMode != "s4d_log_dt" || meta.DtMin != 0.001 || meta.DtMax != 0.1 {
				t.Fatalf("log_dt metadata=%+v", meta)
			}
		case "s4d_log_A_real":
			if math.Abs(float64(meta.InitValue)-math.Log(0.5)) > 1e-7 {
				t.Fatalf("log_A_real init=%g want log(0.5)", meta.InitValue)
			}
		case "s4d_A_imag":
			if meta.InitMode != "s4d_A_imag_lin" {
				t.Fatalf("A_imag init=%q", meta.InitMode)
			}
		case "s4d_C_real", "s4d_C_imag":
			if meta.InitMode != "s4d_C_normal" {
				t.Fatalf("%s init=%q", meta.Name, meta.InitMode)
			}
		case "s4d_D":
			if meta.InitMode != "s4d_D_normal" {
				t.Fatalf("D init=%q", meta.InitMode)
			}
		}
	}

	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}
	if got := countOps(prog, OpS4D); got != 1 {
		t.Fatalf("S4D ops=%d want 1", got)
	}
	params, expanded, err := ParameterCountsFromConfig(cfg)
	if err != nil {
		t.Fatalf("ParameterCountsFromConfig: %v", err)
	}
	if params != 2080 || expanded != params {
		t.Fatalf("parameter counts=(%d,%d) want (2080,2080)", params, expanded)
	}
	if flops := EstimateFLOPs(cfg); flops.ForwardFLOPs <= 0 || flops.FLOPsPerToken <= 0 {
		t.Fatalf("invalid S4D FLOP estimate: %+v", flops)
	}
	for _, op := range prog.Ops {
		if op.Code != OpS4D {
			continue
		}
		wantParams := []int{1, 16, 8, 64, 0}
		if !intSlicesEqual(op.IntParams, wantParams) {
			t.Fatalf("S4D params=%v want %v", op.IntParams, wantParams)
		}
		return
	}
	t.Fatal("missing S4D op")
}

func TestS4DBlockWeightCount(t *testing.T) {
	for _, tc := range []struct {
		blockScales bool
		want        int
	}{
		{false, 7},
		{true, 8},
	} {
		got, err := BlockWeightCount(BlockSpec{Type: "s4d", StateSize: 16}, tc.blockScales, false)
		if err != nil {
			t.Fatal(err)
		}
		if got != tc.want {
			t.Fatalf("BlockWeightCount(block_scales=%v)=%d want %d", tc.blockScales, got, tc.want)
		}
	}
}

func TestS4DReferenceGLUOutputTransform(t *testing.T) {
	spec := BlockSpec{Type: "s4d", StateSize: 16, OutputTransform: "glu"}
	metas, err := s4dWeightShapesWithOptions(spec, 12, EmitOptions{
		Norm:          defaultNormSpec(),
		NormPlacement: NormPlacementPre,
		Dropout:       0.1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(metas) != 9 {
		t.Fatalf("weights=%d want 9", len(metas))
	}
	if got := metas[7]; got.Name != "s4d_out_proj" || !intSlicesEqual(got.Shape, []int{12, 24}) {
		t.Fatalf("output projection metadata=%+v", got)
	}
	if got := metas[8]; got.Name != "s4d_out_bias" || !intSlicesEqual(got.Shape, []int{24}) || !got.InitZero {
		t.Fatalf("output bias metadata=%+v", got)
	}

	cfg, err := ParseArchConfig([]byte(`{
		"model_dim":12,"vocab_size":32,"seq_len":8,"dropout":0.1,
		"blocks":[{"type":"s4d","state_size":16,"output_transform":"glu"}],
		"training":{"objective":"causal","batch_tokens":8}
	}`), "s4d-glu")
	if err != nil {
		t.Fatal(err)
	}
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if got := countOps(prog, OpSlice); got != 2 {
		t.Fatalf("Slice ops=%d want 2", got)
	}
	if got := countOps(prog, OpSigmoid); got != 1 {
		t.Fatalf("Sigmoid ops=%d want 1", got)
	}
	if got := countOps(prog, OpGELUExact); got != 1 {
		t.Fatalf("GELUExact ops=%d want 1", got)
	}
	if got := countOps(prog, OpGELU); got != 0 {
		t.Fatalf("approximate GELU ops=%d want 0 in reference transform", got)
	}
	if got := countOps(prog, OpDropout); got != 2 {
		t.Fatalf("Dropout ops=%d want 2 (S4D inner and residual dropout)", got)
	}
}

func TestS4DConfigValidation(t *testing.T) {
	cases := []struct {
		name  string
		block string
		want  string
	}{
		{"odd state", `{"type":"s4d","state_size":7}`, "positive even state_size"},
		{"negative state", `{"type":"s4d","state_size":-2}`, "positive even state_size"},
		{"unknown init", `{"type":"s4d","init":"random"}`, `v1 supports "s4d-lin"`},
		{"bad dt min", `{"type":"s4d","dt_min":-0.1}`, "0 < dt_min < dt_max"},
		{"bad dt order", `{"type":"s4d","dt_min":0.2,"dt_max":0.1}`, "0 < dt_min < dt_max"},
		{"bidirectional", `{"type":"s4d","bidirectional":true}`, "not supported in v1"},
		{"bad output transform", `{"type":"s4d","output_transform":"linear"}`, "output_transform"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			raw := []byte(`{
				"model_dim":8,"vocab_size":32,"seq_len":8,
				"blocks":[` + tc.block + `],
				"training":{"objective":"causal","batch_tokens":8}
			}`)
			_, err := ParseArchConfig(raw, tc.name)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error=%v want substring %q", err, tc.want)
			}
		})
	}
}

func TestS4DNormPlacementAndBlockScaleWeights(t *testing.T) {
	scale := float64(0)
	spec := BlockSpec{Type: "s4d", StateSize: 16, ResidualScaleInit: &scale}
	metas, err := s4dWeightShapesWithOptions(spec, 12, EmitOptions{
		Norm:          NormSpec{Type: NormTypeLayerNorm, Affine: true, Eps: 1e-5},
		NormPlacement: NormPlacementSandwich,
		BlockScales:   true,
	})
	if err != nil {
		t.Fatal(err)
	}
	wantNames := []string{
		"s4d_norm_scale", "s4d_norm_bias",
		"s4d_log_dt", "s4d_log_A_real", "s4d_A_imag", "s4d_C_real", "s4d_C_imag", "s4d_D",
		"s4d_post_norm_scale", "s4d_post_norm_bias", "s4d_scale",
	}
	if len(metas) != len(wantNames) {
		t.Fatalf("weights=%d want %d: %+v", len(metas), len(wantNames), metas)
	}
	for i, wantName := range wantNames {
		if metas[i].Name != wantName {
			t.Fatalf("weight[%d]=%q want %q", i, metas[i].Name, wantName)
		}
	}
	if got := metas[len(metas)-1]; !got.InitZero || got.InitValue != 0 {
		t.Fatalf("zero residual scale metadata=%+v", got)
	}
}

func TestS4DDisabledParityDoesNotChangePlainConfig(t *testing.T) {
	cfg, err := ParseArchConfig([]byte(`{
		"model_dim":8,"vocab_size":32,"seq_len":8,
		"blocks":[{"type":"plain","heads":2},{"type":"swiglu"}],
		"training":{"objective":"causal","batch_tokens":8}
	}`), "plain-parity")
	if err != nil {
		t.Fatal(err)
	}
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if got := countOps(prog, OpS4D); got != 0 {
		t.Fatalf("plain config emitted %d S4D ops", got)
	}
}
