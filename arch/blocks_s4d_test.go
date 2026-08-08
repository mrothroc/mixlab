package arch

import (
	"encoding/json"
	"math"
	"os"
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

func TestS4DTopLevelTiedDropoutOnlyAffectsS4DBlocks(t *testing.T) {
	cfg, err := ParseArchConfig([]byte(`{
		"model_dim":8,
		"vocab_size":32,
		"seq_len":8,
		"dropout":0.1,
		"tie_dropout":true,
		"blocks":[{"type":"s4d"},{"type":"swiglu"}],
		"training":{"objective":"causal","batch_tokens":8}
	}`), "s4d-mixed-tied-dropout")
	if err != nil {
		t.Fatal(err)
	}
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	tied, ordinary := 0, 0
	for _, op := range prog.Ops {
		if op.Code != OpDropout {
			continue
		}
		switch len(op.IntParams) {
		case 4:
			tied++
		case 1:
			ordinary++
		}
	}
	if tied == 0 || ordinary == 0 {
		t.Fatalf("dropout ops tied=%d ordinary=%d, want both in mixed stack", tied, ordinary)
	}
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

func TestS4DFrequencyTuningConfigWeightsAndIR(t *testing.T) {
	cfg, err := ParseArchConfig([]byte(`{
		"model_dim":8,"vocab_size":32,"seq_len":8,
		"blocks":[{"type":"s4d","state_size":16,"freq_scale":3,
			"sobolev_filter":{"beta_init":-0.5,"learning_rate":0.004}}],
		"training":{"objective":"causal","batch_tokens":8}
	}`), "s4d-frequency-tuning")
	if err != nil {
		t.Fatal(err)
	}
	block := cfg.Blocks[0]
	if got := effectiveS4DFreqScale(block); got != 3 {
		t.Fatalf("freq_scale=%g want 3", got)
	}
	if !block.S4DSobolevFilterEnabled() || effectiveS4DSobolevLearningRate(block) != 0.004 {
		t.Fatalf("sobolev filter=%+v", block.SobolevFilter)
	}
	metas, err := s4dWeightShapesWithOptions(block, cfg.ModelDim, EmitOptions{
		Norm: defaultNormSpec(), NormPlacement: NormPlacementPre,
	})
	if err != nil {
		t.Fatal(err)
	}
	byName := make(map[string]WeightMeta, len(metas))
	for _, meta := range metas {
		byName[meta.Name] = meta
	}
	if got := byName["s4d_A_imag"].InitScale; got != 3 {
		t.Fatalf("A_imag init scale=%g want 3", got)
	}
	beta := byName["s4d_sobolev_beta"]
	if !intSlicesEqual(beta.Shape, []int{8}) || beta.InitValue != -0.5 || beta.InitZero ||
		beta.OptimizerRole != "s4d_state" || math.Abs(float64(beta.OptimizerLR)-0.004) > 1e-7 || !beta.ForceNoDecay {
		t.Fatalf("sobolev beta metadata=%+v", beta)
	}
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, op := range prog.Ops {
		if op.Code != OpS4D {
			continue
		}
		if len(op.Inputs) != 8 || !intSlicesEqual(op.IntParams, []int{1, 8, 8, 16, 0, 1}) {
			t.Fatalf("frequency-tuned S4D inputs=%d params=%v", len(op.Inputs), op.IntParams)
		}
		return
	}
	t.Fatal("missing S4D op")
}

func TestS4DSobolevBooleanShorthandAndDisabledParity(t *testing.T) {
	for _, tc := range []struct {
		value       string
		enabled     bool
		wantWeights int
	}{
		{value: "true", enabled: true, wantWeights: 8},
		{value: "false", enabled: false, wantWeights: 7},
	} {
		raw := []byte(`{
			"model_dim":8,"vocab_size":32,"seq_len":8,
			"blocks":[{"type":"s4d","state_size":16,"sobolev_filter":` + tc.value + `}],
			"training":{"objective":"causal","batch_tokens":8}
		}`)
		cfg, err := ParseArchConfig(raw, "s4d-sobolev-"+tc.value)
		if err != nil {
			t.Fatal(err)
		}
		block := cfg.Blocks[0]
		if block.S4DSobolevFilterEnabled() != tc.enabled {
			t.Fatalf("sobolev_filter=%s enabled=%v want %v", tc.value, block.S4DSobolevFilterEnabled(), tc.enabled)
		}
		if tc.enabled && effectiveS4DSobolevLearningRate(block) != DefaultS4DSobolevLR {
			t.Fatalf("default Sobolev LR=%g want %g", effectiveS4DSobolevLearningRate(block), DefaultS4DSobolevLR)
		}
		got, err := BlockWeightCount(block, false, false)
		if err != nil {
			t.Fatal(err)
		}
		if got != tc.wantWeights {
			t.Fatalf("sobolev_filter=%s weights=%d want %d", tc.value, got, tc.wantWeights)
		}
		encoded, err := json.Marshal(block)
		if err != nil {
			t.Fatal(err)
		}
		var roundTrip BlockSpec
		if err := json.Unmarshal(encoded, &roundTrip); err != nil {
			t.Fatal(err)
		}
		if roundTrip.S4DSobolevFilterEnabled() != tc.enabled {
			t.Fatalf("round-trip sobolev_filter=%s enabled=%v want %v; JSON=%s", tc.value, roundTrip.S4DSobolevFilterEnabled(), tc.enabled, encoded)
		}
	}
}

func TestS4DFrequencyTuningValidation(t *testing.T) {
	for _, tc := range []struct {
		name      string
		blockJSON string
		want      string
	}{
		{name: "zero frequency scale", blockJSON: `"freq_scale":0`, want: "freq_scale"},
		{name: "negative frequency scale", blockJSON: `"freq_scale":-1`, want: "freq_scale"},
		{name: "bad beta", blockJSON: `"sobolev_filter":{"beta_init":1e999}`, want: "cannot unmarshal"},
		{name: "zero filter lr", blockJSON: `"sobolev_filter":{"learning_rate":0}`, want: "learning_rate"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw := []byte(`{
				"model_dim":8,"vocab_size":32,"seq_len":8,
				"blocks":[{"type":"s4d","state_size":16,` + tc.blockJSON + `}],
				"training":{"objective":"causal","batch_tokens":8}
			}`)
			_, err := ParseArchConfig(raw, tc.name)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error=%v want containing %q", err, tc.want)
			}
		})
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
		{"bad n_ssm", `{"type":"s4d","n_ssm":3}`, "n_ssm to divide model_dim"},
		{"bad discretization", `{"type":"s4d","discretization":"euler"}`, "discretization"},
		{"bad state lr", `{"type":"s4d","state_lr":-0.001}`, "state_lr"},
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

func TestS4DReferenceConfigWeightsIRAndTiedDropout(t *testing.T) {
	cfg, err := ParseArchConfig([]byte(`{
		"model_dim":12,"seq_len":8,"dropout":0.1,"tie_dropout":true,
		"norm_type":"layernorm","norm_placement":"post_residual","final_norm":false,
		"input_adapter":{"kind":"linear_frames","feature_dim":1,"bias":true,"norm":"none"},
		"blocks":[{
			"type":"s4d","state_size":16,"n_ssm":2,"bidirectional":true,
			"discretization":"bilinear","trainable_b":true,"state_lr":0.001,
			"output_transform":"glu"
		}],
		"training":{
			"objective":"classification",
			"classification":{"num_labels":3,"pooling":"mean","classifier_dropout":0},
			"optimizer":"adamw","lr":0.01,"weight_decay":0.05,
			"weight_decay_policy":"all","batch_tokens":8
		}
	}`), "s4d-reference")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.EffectiveFinalNorm() {
		t.Fatal("final_norm=true, want false")
	}
	if cfg.Training.EffectiveWeightDecayPolicy() != WeightDecayPolicyAll {
		t.Fatalf("weight decay policy=%q", cfg.Training.EffectiveWeightDecayPolicy())
	}

	metas, err := CollectWeightShapesFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	byName := make(map[string]WeightMeta, len(metas))
	for _, meta := range metas {
		byName[meta.Name] = meta
		if strings.HasPrefix(meta.Name, "final_norm") {
			t.Fatalf("final norm weight unexpectedly present: %+v", meta)
		}
	}
	wantShapes := map[string][]int{
		"s4d_log_dt":                   {12},
		"s4d_log_A_real":               {2, 8},
		"s4d_A_imag":                   {2, 8},
		"s4d_B_real":                   {2, 8},
		"s4d_B_imag":                   {2, 8},
		"s4d_C_real":                   {12, 8},
		"s4d_C_imag":                   {12, 8},
		"s4d_C_backward_real":          {12, 8},
		"s4d_C_backward_imag":          {12, 8},
		"s4d_D":                        {12},
		"s4d_out_proj":                 {12, 24},
		"s4d_out_bias":                 {24},
		"s4d_post_residual_norm_scale": {12},
		"s4d_post_residual_norm_bias":  {12},
	}
	for name, shape := range wantShapes {
		meta, ok := byName[name]
		if !ok || !intSlicesEqual(meta.Shape, shape) {
			t.Fatalf("weight %q=%+v want shape %v", name, meta, shape)
		}
	}
	for _, name := range []string{"s4d_log_A_real", "s4d_A_imag", "s4d_B_real", "s4d_B_imag"} {
		meta := byName[name]
		if meta.OptimizerRole != "s4d_state" || meta.OptimizerLR != 0.001 || !meta.ForceNoDecay {
			t.Fatalf("%s optimizer metadata=%+v", name, meta)
		}
	}
	if meta := byName["s4d_B_real"]; meta.InitMode != "s4d_B_one" || meta.InitOne {
		t.Fatalf("B real initialization metadata=%+v", meta)
	}
	if meta := byName["s4d_B_imag"]; !meta.InitZero {
		t.Fatalf("B imaginary initialization metadata=%+v", meta)
	}
	if meta := byName["s4d_log_dt"]; meta.OptimizerRole != "s4d_main" || !meta.ForceNoDecay {
		t.Fatalf("dt optimizer metadata=%+v", meta)
	}
	for _, name := range []string{"s4d_C_real", "s4d_C_imag", "s4d_D"} {
		meta := byName[name]
		if meta.OptimizerRole != "s4d_main" || meta.ForceNoDecay {
			t.Fatalf("%s optimizer metadata=%+v", name, meta)
		}
	}

	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	var s4d *Op
	tiedDropouts := 0
	for i := range prog.Ops {
		op := &prog.Ops[i]
		if op.Code == OpS4D {
			s4d = op
		}
		if op.Code == OpDropout && len(op.IntParams) == 4 {
			tiedDropouts++
			if !intSlicesEqual(op.IntParams[1:], []int{1, 8, 12}) {
				t.Fatalf("tied dropout params=%v", op.IntParams)
			}
		}
	}
	if s4d == nil {
		t.Fatal("missing S4D op")
	}
	if len(s4d.Inputs) != 11 || !intSlicesEqual(s4d.IntParams, []int{1, 8, 12, 16, 0, 2, 7}) {
		t.Fatalf("advanced S4D inputs=%d params=%v", len(s4d.Inputs), s4d.IntParams)
	}
	if tiedDropouts != 2 {
		t.Fatalf("tied dropout ops=%d want 2", tiedDropouts)
	}
}

func TestS4DLegacyIRAndWeightLayoutRemainExact(t *testing.T) {
	cfg, err := ParseArchConfig(validS4DConfigJSON(), "legacy-parity")
	if err != nil {
		t.Fatal(err)
	}
	metas, err := CollectWeightShapesFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{"s4d_B_real", "s4d_C_backward_real"} {
		for _, meta := range metas {
			if meta.Name == forbidden {
				t.Fatalf("legacy layout contains %q", forbidden)
			}
		}
	}
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, op := range prog.Ops {
		if op.Code == OpS4D {
			if len(op.Inputs) != 7 || !intSlicesEqual(op.IntParams, []int{1, 16, 8, 64, 0}) {
				t.Fatalf("legacy S4D inputs=%d params=%v", len(op.Inputs), op.IntParams)
			}
			return
		}
	}
	t.Fatal("missing legacy S4D op")
}

func TestS4DLRAImageReferenceExampleParsesAndCounts(t *testing.T) {
	raw, err := os.ReadFile("../examples/continuous_s4d_lra_image_reference.json")
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := ParseArchConfig(raw, "continuous_s4d_lra_image_reference.json")
	if err != nil {
		t.Fatal(err)
	}
	if len(cfg.Blocks) != 6 || cfg.ModelDim != 512 || cfg.SeqLen != 1024 {
		t.Fatalf("unexpected reference dimensions: blocks=%d D=%d T=%d", len(cfg.Blocks), cfg.ModelDim, cfg.SeqLen)
	}
	if cfg.Training.BatchTokens/cfg.SeqLen != 50 || cfg.Training.Steps != 180000 ||
		cfg.Training.LRScheduleSteps != 200000 || cfg.Training.WarmupSteps != 1000 ||
		cfg.Training.WeightInit != "pytorch_linear" || cfg.Training.Seed != 2222 {
		t.Fatalf("unexpected training recipe: %+v", cfg.Training)
	}
	metas, err := CollectWeightShapesFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	wantFanIn := map[string]int{
		"input_adapter_proj":   1,
		"input_adapter_bias":   1,
		"s4d_out_proj":         512,
		"s4d_out_bias":         512,
		"head_classifier_proj": 512,
		"head_classifier_bias": 512,
	}
	seen := map[string]int{}
	for _, meta := range metas {
		if want, ok := wantFanIn[meta.Name]; ok {
			if meta.PyTorchLinearFanIn != want {
				t.Fatalf("%s PyTorchLinearFanIn=%d want %d", meta.Name, meta.PyTorchLinearFanIn, want)
			}
			seen[meta.Name]++
		}
	}
	for name := range wantFanIn {
		if seen[name] == 0 {
			t.Fatalf("missing affine metadata for %s", name)
		}
	}
	params, expanded, err := ParameterCountsFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if params <= 0 || expanded != params {
		t.Fatalf("parameter counts=(%d,%d)", params, expanded)
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
