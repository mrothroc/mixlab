package arch

import (
	"strings"
	"testing"
)

func TestParseArchConfig_MoEDefaultsAndValidation(t *testing.T) {
	valid := `{
		"name":"moe_valid",
		"model_dim":32,
		"vocab_size":128,
		"seq_len":8,
		"blocks":[
			{"type":"plain","heads":4},
			{"type":"moe","num_experts":4}
		],
		"training":{"batch_tokens":8}
	}`
	cfg, err := ParseArchConfig([]byte(valid), "moe_valid")
	if err != nil {
		t.Fatalf("ParseArchConfig(valid): %v", err)
	}
	moe := cfg.Blocks[1]
	if got := effectiveMoETopK(moe); got != 2 {
		t.Fatalf("default top_k=%d, want 2", got)
	}
	if got := effectiveMoELoadBalanceLossWeight(moe); got != 0.01 {
		t.Fatalf("default load_balance_loss_weight=%g, want 0.01", got)
	}
	if got := effectiveMoEExpertBlock(moe).Type; got != "swiglu" {
		t.Fatalf("default expert type=%q, want swiglu", got)
	}

	cases := []struct {
		name string
		body string
		want string
	}{
		{"bad_experts", `{"type":"moe","num_experts":0}`, "num_experts > 0"},
		{"bad_topk", `{"type":"moe","num_experts":2,"top_k":3}`, "invalid top_k"},
		{"bad_router", `{"type":"moe","num_experts":2,"router":"hash"}`, "invalid router"},
		{"bad_aux", `{"type":"moe","num_experts":2,"load_balance_loss_weight":-0.1}`, "load_balance_loss_weight"},
		{"bad_expert", `{"type":"moe","num_experts":2,"expert_block":{"type":"plain","heads":2}}`, "expert_block.type"},
		{"bad_mlp_activation", `{"type":"moe","num_experts":2,"expert_block":{"type":"mlp","activation":"bogus"}}`, "invalid activation"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			raw := `{
				"name":"` + tc.name + `",
				"model_dim":32,
				"vocab_size":128,
				"seq_len":8,
				"blocks":[` + tc.body + `],
				"training":{"batch_tokens":8}
			}`
			_, err := ParseArchConfig([]byte(raw), tc.name)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error=%v, want substring %q", err, tc.want)
			}
		})
	}
}

func TestMoEWeightShapesAndActiveParams(t *testing.T) {
	spec := BlockSpec{
		Type:       "moe",
		NumExperts: 4,
		TopK:       2,
		ExpertBlock: &BlockSpec{
			Type: "geglu",
		},
	}
	metas, err := blockWeightShapes(spec, 32, 8, 1, 128, 2.0, true, false)
	if err != nil {
		t.Fatalf("blockWeightShapes: %v", err)
	}
	if got, want := len(metas), 2+4*3+1; got != want {
		t.Fatalf("moe weight count=%d, want %d", got, want)
	}
	if metas[0].Name != "moe_norm_scale" || !metas[0].IsNormScale {
		t.Fatalf("first meta=%+v, want moe_norm_scale norm", metas[0])
	}
	if got, want := metas[1].Shape, []int{32, 4}; len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("router shape=%v, want %v", got, want)
	}
	if metas[len(metas)-1].Name != "moe_scale" || !metas[len(metas)-1].InitOne {
		t.Fatalf("last meta=%+v, want moe_scale init-one", metas[len(metas)-1])
	}

	cfg := &ArchConfig{
		Name:        "moe_params",
		ModelDim:    32,
		VocabSize:   128,
		SeqLen:      8,
		MLPMult:     2.0,
		BlockScales: true,
		Blocks:      []BlockSpec{{Type: "plain", Heads: 4}, spec},
		Training:    TrainingSpec{BatchTokens: 8},
	}
	active, hasMoE, err := ActiveParameterCountFromConfig(cfg)
	if err != nil {
		t.Fatalf("ActiveParameterCountFromConfig: %v", err)
	}
	if !hasMoE {
		t.Fatal("hasMoE=false, want true")
	}
	total, _, err := ParameterCountsFromConfig(cfg)
	if err != nil {
		t.Fatalf("ParameterCountsFromConfig: %v", err)
	}
	if active >= total {
		t.Fatalf("active params=%d, want less than total=%d", active, total)
	}
}

func TestBuildTrainingIRProgram_MoEAddsAuxLossAndOutputs(t *testing.T) {
	cfg := &ArchConfig{
		Name:      "moe_ir",
		ModelDim:  32,
		VocabSize: 64,
		SeqLen:    4,
		Blocks: []BlockSpec{
			{Type: "plain", Heads: 4},
			{Type: "moe", NumExperts: 3, TopK: 1, LoadBalanceLossWeight: 0},
		},
		Training: TrainingSpec{BatchTokens: 4},
	}
	cfg.Blocks[1].loadBalanceLossWeightSet = true
	prog, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig: %v", err)
	}
	var moeOps int
	var lossAddsAux bool
	for _, op := range prog.Ops {
		if op.Code == OpMoEFeedForward {
			moeOps++
			if len(op.Outputs) != 3 {
				t.Fatalf("moe op outputs=%v, want 3 outputs", op.Outputs)
			}
		}
		if op.Code == OpAdd && len(op.Inputs) == 2 && op.Inputs[0] == "task_loss" && op.Inputs[1] == "moe_aux_loss" && len(op.Outputs) == 1 && op.Outputs[0] == "loss" {
			lossAddsAux = true
		}
	}
	if moeOps != 1 {
		t.Fatalf("moe op count=%d, want 1", moeOps)
	}
	if !lossAddsAux {
		t.Fatal("training loss does not add moe_aux_loss")
	}
	for _, name := range []string{"eval_loss", "moe_aux_loss", "moe_router_entropy"} {
		if !programDeclaresOutputArch(prog, name) {
			t.Fatalf("missing output %q", name)
		}
	}
}

func TestParallelResidualAcceptsMoEFollowerAndOmitsNorm(t *testing.T) {
	on := true
	blocks := []BlockSpec{
		{Type: "plain", Heads: 4, ParallelResidual: &on},
		{Type: "moe", NumExperts: 2, TopK: 1},
	}
	metas, err := CollectWeightShapesWithBigramRecurrenceAndParallel(32, 128, 8, DefaultFFNMultiplier, false, false, false, false, true, 0, 0, blocks, nil)
	if err != nil {
		t.Fatalf("CollectWeightShapesWithBigramRecurrenceAndParallel: %v", err)
	}
	for _, meta := range metas {
		if meta.Name == "moe_norm_scale" {
			t.Fatal("parallel-residual moe follower should omit moe_norm_scale")
		}
	}
	cfg := &ArchConfig{
		Name:             "parallel_moe",
		ModelDim:         32,
		VocabSize:        128,
		SeqLen:           8,
		ParallelResidual: true,
		Blocks:           blocks,
		Training:         TrainingSpec{BatchTokens: 8},
	}
	prog, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig: %v", err)
	}
	found := false
	for _, op := range prog.Ops {
		if op.Code == OpMoEFeedForward {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("parallel-residual moe follower did not emit OpMoEFeedForward")
	}
}
