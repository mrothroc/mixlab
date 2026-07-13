package arch

import (
	"strings"
	"testing"
)

func validTTTMLPConfigJSON(objective string) []byte {
	return []byte(`{
		"model_dim": 16,
		"vocab_size": 64,
		"seq_len": 8,
		"blocks": [{"type":"ttt_mlp","heads":2}],
		"training": {"objective":"` + objective + `","batch_tokens":8,"steps":2,"lr":0.001}
	}`)
}

func TestTTTMLPConfigDefaultsAndWeights(t *testing.T) {
	cfg, err := ParseArchConfig(validTTTMLPConfigJSON(ObjectiveCausal), "ttt-test")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	spec := cfg.Blocks[0]
	if got := effectiveTTTMLPChunkSize(spec); got != 16 {
		t.Fatalf("chunk_size=%d want 16", got)
	}
	if got := effectiveTTTMLPInnerHiddenMult(spec); got != 4 {
		t.Fatalf("inner_hidden_mult=%g want 4", got)
	}
	if got := effectiveTTTMLPInnerLRBase(spec); got != 0.1 {
		t.Fatalf("inner_lr_base=%g want 0.1", got)
	}
	if got := effectiveTTTMLPInnerLRInit(spec); got != 0.01 {
		t.Fatalf("inner_lr_init=%g want 0.01", got)
	}
	metas, err := blockWeightShapes(spec, cfg.ModelDim, cfg.SeqLen, 1, cfg.VocabSize, cfg.EffectiveMLPMult(), false, false)
	if err != nil {
		t.Fatalf("blockWeightShapes: %v", err)
	}
	if len(metas) != 20 {
		t.Fatalf("weight count=%d want 20", len(metas))
	}
	stateCount, found, err := TTTMLPRecurrentStateCountFromConfig(cfg)
	if err != nil {
		t.Fatalf("TTTMLPRecurrentStateCountFromConfig: %v", err)
	}
	if !found || stateCount != 1104 {
		t.Fatalf("recurrent state count=%d found=%v want 1104", stateCount, found)
	}
	params, expanded, err := ParameterCountsFromConfig(cfg)
	if err != nil {
		t.Fatalf("ParameterCountsFromConfig: %v", err)
	}
	if params != 4482 || expanded != params {
		t.Fatalf("parameter counts=(%d,%d) want (4482,4482)", params, expanded)
	}
	flops := EstimateFLOPs(cfg)
	if flops.ForwardFLOPs <= 0 || flops.TrainingFLOPs != 0 || flops.FLOPsPerToken != 0 {
		t.Fatalf("invalid TTT-MLP FLOP estimate: %+v", flops)
	}
	if flops.TrainingFLOPsReliable {
		t.Fatalf("TTT training FLOPs must be marked unreliable: %+v", flops)
	}
	want := map[string][]int{
		"w_qk":              {16, 16},
		"q_conv":            {16, 4},
		"inner_lr_w":        {16, 2},
		"inner_token_coeff": {16},
		"inner_w1":          {16, 32},
		"inner_b1":          {2, 32},
		"inner_w2":          {64, 8},
		"inner_norm_scale":  {2, 8},
		"post_norm_scale":   {16},
		"w_out":             {16, 16},
	}
	for _, meta := range metas {
		if shape, ok := want[meta.Name]; ok {
			if len(shape) != len(meta.Shape) {
				t.Fatalf("%s shape=%v want %v", meta.Name, meta.Shape, shape)
			}
			for i := range shape {
				if shape[i] != meta.Shape[i] {
					t.Fatalf("%s shape=%v want %v", meta.Name, meta.Shape, shape)
				}
			}
		}
	}
	for _, meta := range metas {
		switch meta.Name {
		case "w_qk", "w_v", "w_out_gate", "w_out", "inner_lr_w", "inner_w1", "inner_w2":
			if meta.InitMode != "ttt_normal_0_02" {
				t.Fatalf("%s init=%q want ttt_normal_0_02", meta.Name, meta.InitMode)
			}
		case "q_conv", "q_conv_bias", "k_conv", "k_conv_bias":
			if meta.InitMode != "ttt_conv_uniform_4" {
				t.Fatalf("%s init=%q want ttt_conv_uniform_4", meta.Name, meta.InitMode)
			}
		}
	}
}

func TestTTTMLPIRAndWarmupInput(t *testing.T) {
	cfg, err := ParseArchConfig(validTTTMLPConfigJSON(ObjectiveCausal), "ttt-test")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	prog, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{Objective: ObjectiveCausal})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig: %v", err)
	}
	if n := countOps(prog, OpTTTMLPScan); n != 1 {
		t.Fatalf("TTTMLPScan ops=%d want 1", n)
	}
	var reversedConvs int
	for _, op := range prog.Ops {
		if op.Code == OpDepthwiseConv1D && len(op.IntParams) == 5 && op.IntParams[4] == 1 {
			reversedConvs++
		}
	}
	if reversedConvs != 2 {
		t.Fatalf("PyTorch-layout causal depthwise convs=%d want 2", reversedConvs)
	}
	foundInput := false
	for _, input := range prog.Inputs {
		if input.Name == "ttt_inner_lr_scale" {
			foundInput = len(input.Shape) == 1 && input.Shape[0] == len(cfg.Blocks)
		}
	}
	if !foundInput {
		t.Fatal("missing stable ttt_inner_lr_scale input")
	}
	for _, name := range []string{"block_0_ttt_inner_loss_before", "block_0_ttt_inner_loss_after", "block_0_ttt_inner_update_norm", "block_0_ttt_state_drift", "block_0_ttt_inner_lr_mean"} {
		found := false
		for _, output := range prog.Outputs {
			found = found || output.Name == name
		}
		if !found {
			t.Fatalf("missing diagnostic output %q", name)
		}
	}
	start := TTTMLPInnerLRScalesForStep(cfg.Blocks, 0)[0]
	end := TTTMLPInnerLRScalesForStep(cfg.Blocks, 4999)[0]
	if start != 0.1 {
		t.Fatalf("step-0 scale=%g want exactly inner_lr_init/inner_lr_base=0.1", start)
	}
	if end != 1 {
		t.Fatalf("warmup-end scale=%g want 1", end)
	}
}

func TestTTTMLPStatefulInferenceIRUsesCheckpointLayout(t *testing.T) {
	cfg, err := ParseArchConfig([]byte(`{
		"model_dim":16,"vocab_size":64,"seq_len":8,
		"blocks":[{"type":"ttt_mlp","heads":2,"chunk_size":4},{"type":"swiglu"}],
		"training":{"objective":"causal","batch_tokens":8,"steps":1,"lr":0.001}
	}`), "ttt-stateful")
	if err != nil {
		t.Fatal(err)
	}
	prog, layouts, err := BuildTTTMLPStatefulInferenceIRProgram(cfg, 2, []int{1})
	if err != nil {
		t.Fatalf("BuildTTTMLPStatefulInferenceIRProgram: %v", err)
	}
	wantWeights, err := CountIRWeightsFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if prog.NumWeights != wantWeights {
		t.Fatalf("NumWeights=%d want checkpoint layout %d", prog.NumWeights, wantWeights)
	}
	if len(layouts) != 1 || layouts[0].BlockIndex != 0 || layouts[0].ChunkSize != 4 || layouts[0].StateSize != 1104 {
		t.Fatalf("unexpected layouts: %+v", layouts)
	}
	if got := countOps(prog, OpTTTMLPStatefulScan); got != 1 {
		t.Fatalf("stateful scan ops=%d want 1", got)
	}
	if got := countOps(prog, OpTTTMLPScan); got != 0 {
		t.Fatalf("stateless scan ops=%d want 0", got)
	}
	for _, name := range []string{layouts[0].StateInput, layouts[0].GradientInput, layouts[0].ConvInput} {
		found := false
		for _, input := range prog.Inputs {
			found = found || input.Name == name
		}
		if !found {
			t.Fatalf("missing state input %q", name)
		}
	}
}

func TestTTTMLPStatefulInferenceIRRejectsBoundaryCrossingAndOtherMixers(t *testing.T) {
	cfg, err := ParseArchConfig(validTTTMLPConfigJSON(ObjectiveCausal), "ttt-stateful")
	if err != nil {
		t.Fatal(err)
	}
	cfg.Blocks[0].ChunkSize = 4
	if _, _, err := BuildTTTMLPStatefulInferenceIRProgram(cfg, 2, []int{3}); err == nil || !strings.Contains(err.Error(), "crosses") {
		t.Fatalf("boundary crossing error=%v", err)
	}
	cfg.Blocks = append(cfg.Blocks, BlockSpec{Type: "plain", Heads: 2})
	if _, _, err := BuildTTTMLPStatefulInferenceIRProgram(cfg, 1, []int{0}); err == nil || !strings.Contains(err.Error(), "no state cache") {
		t.Fatalf("unsupported mixer error=%v", err)
	}
}

func TestTTTMLPRejectsExplicitZeroPositiveFields(t *testing.T) {
	for _, field := range []string{"chunk_size", "inner_hidden_mult", "inner_lr_base", "inner_lr_init"} {
		t.Run(field, func(t *testing.T) {
			raw := []byte(`{
				"model_dim":16,"vocab_size":32,"seq_len":4,
				"blocks":[{"type":"ttt_mlp","heads":2,"` + field + `":0}],
				"training":{"objective":"causal","batch_tokens":8}
			}`)
			if _, err := ParseArchConfig(raw, "ttt-explicit-zero"); err == nil || !strings.Contains(err.Error(), field) {
				t.Fatalf("error=%v want explicit %s=0 rejection", err, field)
			}
		})
	}
}

func TestTTTMLPExplicitZeroDisablesInnerLRWarmup(t *testing.T) {
	raw := []byte(`{
		"model_dim":16,"vocab_size":32,"seq_len":4,
		"blocks":[{"type":"ttt_mlp","heads":2,"inner_lr_warmup_steps":0}],
		"training":{"objective":"causal","batch_tokens":8}
	}`)
	cfg, err := ParseArchConfig(raw, "ttt-zero-warmup")
	if err != nil {
		t.Fatal(err)
	}
	if got := effectiveTTTMLPInnerLRWarmupSteps(cfg.Blocks[0]); got != 0 {
		t.Fatalf("warmup=%d want explicit zero", got)
	}
	if got := TTTMLPInnerLRScalesForStep(cfg.Blocks, 0)[0]; got != 1 {
		t.Fatalf("step-zero inner LR scale=%g want 1 with warmup disabled", got)
	}
}

func TestTTTMLPRejectsUnsupportedV1Compositions(t *testing.T) {
	for _, objective := range []string{ObjectiveMLM, ObjectiveMNTP, ObjectiveHybrid, ObjectiveBlockDiffusion, ObjectiveMultihead} {
		t.Run(objective, func(t *testing.T) {
			_, err := ParseArchConfig(validTTTMLPConfigJSON(objective), "ttt-test")
			if err == nil || !strings.Contains(err.Error(), "causal only") {
				t.Fatalf("error=%v want causal-only rejection", err)
			}
		})
	}
	badHead := []byte(`{"model_dim":15,"vocab_size":64,"seq_len":8,"blocks":[{"type":"ttt_mlp","heads":3}],"training":{"objective":"causal","batch_tokens":8}}`)
	if _, err := ParseArchConfig(badHead, "ttt-test"); err == nil || !strings.Contains(err.Error(), "even head_dim") {
		t.Fatalf("error=%v want even head_dim rejection", err)
	}
}
