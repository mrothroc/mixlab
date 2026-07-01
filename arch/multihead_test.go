package arch

import (
	"strings"
	"testing"
)

func TestMultiheadConfigDefaultsAndValidation(t *testing.T) {
	cfg := parseMultiheadConfig(t, `"training": {
		"objective": "multihead",
		"steps": 1,
		"lr": 0.001,
		"batch_tokens": 8,
		"mlm_mask_token_id": 31,
		"heads": [
			{"name": "scorer", "objective": "mntp", "loss_weight": 0.7},
			{"name": "denoiser", "objective": "block_diffusion", "loss_weight": 0.3,
			 "diffusion": {"block_size": 2, "timestep_conditioning": "adaln", "timestep_conditioning_dim": 5}}
		]
	}`)
	if cfg.Training.ExportHead != "scorer" {
		t.Fatalf("export_head=%q, want scorer", cfg.Training.ExportHead)
	}
	if cfg.Training.DiffusionHead != "denoiser" {
		t.Fatalf("diffusion_head=%q, want denoiser", cfg.Training.DiffusionHead)
	}
	if cfg.Training.Heads[0].OutputHead != MultiheadOutputBERTMLM {
		t.Fatalf("scorer output_head=%q, want bert_mlm", cfg.Training.Heads[0].OutputHead)
	}
	if !cfg.Training.Heads[0].TieEmbeddings || !cfg.Training.Heads[0].FinalNorm {
		t.Fatalf("scorer tie/final norm defaults = %v/%v, want true/true", cfg.Training.Heads[0].TieEmbeddings, cfg.Training.Heads[0].FinalNorm)
	}
	if cfg.Training.Heads[1].OutputHead != MultiheadOutputLinear {
		t.Fatalf("denoiser output_head=%q, want linear", cfg.Training.Heads[1].OutputHead)
	}
	if cfg.Training.Heads[1].Diffusion == nil || cfg.Training.Heads[1].Diffusion.TimestepConditioning != DiffusionTimestepConditioningAdaLN {
		t.Fatalf("denoiser diffusion spec not normalized: %+v", cfg.Training.Heads[1].Diffusion)
	}
}

func TestMultiheadValidationErrors(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string
	}{
		{
			name: "duplicate names",
			body: `"training": {"objective": "multihead", "steps": 1, "lr": 0.001, "batch_tokens": 8, "mlm_mask_token_id": 31,
				"heads": [{"name": "h", "objective": "mntp"}, {"name": "h", "objective": "block_diffusion"}]}`,
			want: "duplicate",
		},
		{
			name: "nested hybrid",
			body: `"training": {"objective": "multihead", "steps": 1, "lr": 0.001, "batch_tokens": 8, "mlm_mask_token_id": 31,
				"heads": [{"name": "s", "objective": "hybrid"}, {"name": "d", "objective": "block_diffusion"}]}`,
			want: "objective",
		},
		{
			name: "top level diffusion",
			body: `"training": {"objective": "multihead", "steps": 1, "lr": 0.001, "batch_tokens": 8, "mlm_mask_token_id": 31,
				"diffusion": {"block_size": 2},
				"heads": [{"name": "s", "objective": "mntp"}, {"name": "d", "objective": "block_diffusion"}]}`,
			want: "training.diffusion",
		},
		{
			name: "bad export head",
			body: `"training": {"objective": "multihead", "steps": 1, "lr": 0.001, "batch_tokens": 8, "mlm_mask_token_id": 31,
				"export_head": "d",
				"heads": [{"name": "s", "objective": "mntp"}, {"name": "d", "objective": "block_diffusion"}]}`,
			want: "cannot select a block_diffusion head",
		},
		{
			name: "window size",
			body: `"blocks": [{"type": "plain", "heads": 2, "window_size": 2}],
				"training": {"objective": "multihead", "steps": 1, "lr": 0.001, "batch_tokens": 8, "mlm_mask_token_id": 31,
				"heads": [{"name": "s", "objective": "mntp"}, {"name": "d", "objective": "block_diffusion"}]}`,
			want: "window_size",
		},
		{
			name: "bert head must be tied",
			body: `"training": {"objective": "multihead", "steps": 1, "lr": 0.001, "batch_tokens": 8, "mlm_mask_token_id": 31,
				"heads": [{"name": "s", "objective": "mntp", "output_head": "bert_mlm", "tie_embeddings": false}, {"name": "d", "objective": "block_diffusion"}]}`,
			want: "tie_embeddings",
		},
		{
			name: "replacement probabilities",
			body: `"training": {"objective": "multihead", "steps": 1, "lr": 0.001, "batch_tokens": 8, "mlm_mask_token_id": 31,
				"mlm_mask_token_prob": 0.5, "mlm_random_token_prob": 0.1, "mlm_kept_unchanged_prob": 0.1,
				"heads": [{"name": "s", "objective": "mntp"}, {"name": "d", "objective": "block_diffusion"}]}`,
			want: "must sum to 1.0",
		},
		{
			name: "rtd requires training rtd",
			body: `"training": {"objective": "multihead", "steps": 1, "lr": 0.001, "batch_tokens": 8, "mlm_mask_token_id": 31,
				"heads": [{"name": "s", "objective": "mntp"}, {"name": "detector", "objective": "rtd"}]}`,
			want: "requires training.rtd",
		},
		{
			name: "rtd dedicated generator bad heads",
			body: `"training": {"objective": "multihead", "steps": 1, "lr": 0.001, "batch_tokens": 8, "mlm_mask_token_id": 31,
				"rtd": {"generator": {"type": "dedicated", "model_dim": 10, "layers": 1, "heads": 4}},
				"heads": [{"name": "s", "objective": "mntp"}, {"name": "detector", "objective": "rtd"}]}`,
			want: "must be divisible",
		},
		{
			name: "rtd generator head must be masked",
			body: `"training": {"objective": "multihead", "steps": 1, "lr": 0.001, "batch_tokens": 8, "mlm_mask_token_id": 31,
				"rtd": {"generator": "tied", "generator_head": "c"},
				"heads": [{"name": "c", "objective": "causal"}, {"name": "detector", "objective": "rtd"}]}`,
			want: "must select an mlm or mntp head",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ParseArchConfig([]byte(multiheadConfigJSON(tt.body)), tt.name)
			if err == nil {
				t.Fatal("expected validation error")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error=%q, want substring %q", err, tt.want)
			}
		})
	}
}

func TestMultiheadRTDDedicatedGeneratorWeightShapesAndIR(t *testing.T) {
	cfg := parseMultiheadConfig(t, `"training": {
		"objective": "multihead",
		"steps": 1,
		"lr": 0.001,
		"batch_tokens": 8,
		"mlm_mask_token_id": 31,
		"export_head": "scorer",
		"rtd": {"generator": {"type": "dedicated", "model_dim": 8, "layers": 1, "heads": 2, "generator_loss_weight": 0.25}},
		"heads": [
			{"name": "scorer", "objective": "mntp", "loss_weight": 0.8},
			{"name": "detector", "objective": "rtd", "loss_weight": 1.0}
		]
	}`)
	if !cfg.Training.RTDDedicatedGeneratorEnabled() {
		t.Fatal("dedicated RTD generator should be enabled")
	}
	if cfg.Training.RTD.DedicatedGenerator.MLPMult != 4.0 {
		t.Fatalf("dedicated mlp_mult=%g, want default 4", cfg.Training.RTD.DedicatedGenerator.MLPMult)
	}
	shapes, err := CollectWeightShapesFromConfig(cfg)
	if err != nil {
		t.Fatalf("CollectWeightShapesFromConfig: %v", err)
	}
	for _, want := range []string{
		"rtd_generator_embed",
		"rtd_generator_l0_norm_scale",
		"rtd_generator_l0_wq",
		"rtd_generator_l0_ff_gate",
		"rtd_generator_l0_ff2",
		"rtd_generator_final_norm",
		"rtd_generator_mlm_dense",
		"rtd_generator_mlm_dense_bias",
		"rtd_generator_mlm_output_bias",
	} {
		if countWeightNames(shapes, want) != 1 {
			t.Fatalf("weight %q count=%d, want 1 in %+v", want, countWeightNames(shapes, want), shapes)
		}
	}
	prog, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig: %v", err)
	}
	if prog.NumWeights != len(shapes) {
		t.Fatalf("prog weights=%d shapes=%d", prog.NumWeights, len(shapes))
	}
	for _, want := range []string{"rtd_generator_tokens", "rtd_generator_targets", "rtd_generator_loss_mask"} {
		if !programDeclaresInputArch(prog, want) {
			t.Fatalf("missing input %q: %+v", want, prog.Inputs)
		}
	}
	if countOps(prog, OpMaskedCrossEntropy) != 2 {
		t.Fatalf("OpMaskedCrossEntropy count=%d, want scorer+generator", countOps(prog, OpMaskedCrossEntropy))
	}
	if countOps(prog, OpMaskedBCEWithLogits) != 1 {
		t.Fatalf("OpMaskedBCEWithLogits count=%d, want 1", countOps(prog, OpMaskedBCEWithLogits))
	}
	for _, want := range []string{"rtd_generator_loss", RTDGeneratorLogitsName} {
		if !programDeclaresOutputArch(prog, want) {
			t.Fatalf("missing output %q: %+v", want, prog.Outputs)
		}
	}
	finalNormReadsGeneratorStream := false
	for _, op := range prog.Ops {
		if op.Code == OpRMSNorm && len(op.Outputs) == 1 && op.Outputs[0] == "rtd_generator_final_norm" {
			if len(op.Inputs) == 0 || op.Inputs[0] != "rtd_generator_x" {
				t.Fatalf("rtd generator final norm inputs=%v, want first input rtd_generator_x", op.Inputs)
			}
			finalNormReadsGeneratorStream = true
		}
	}
	if !finalNormReadsGeneratorStream {
		t.Fatal("missing rtd_generator_final_norm RMSNorm op")
	}
}

func TestMultiheadRTDConfigWeightShapesAndIR(t *testing.T) {
	cfg := parseMultiheadConfig(t, `"training": {
		"objective": "multihead",
		"steps": 1,
		"lr": 0.001,
		"batch_tokens": 8,
		"mlm_mask_token_id": 31,
		"export_head": "scorer",
		"rtd": {"generator": "tied", "generator_head": "scorer", "mask_prob": 0.2, "sample_temperature": 1.3, "discriminator_loss_weight": 50},
		"heads": [
			{"name": "scorer", "objective": "mntp", "loss_weight": 0.8},
			{"name": "detector", "objective": "rtd", "loss_weight": 1.0}
		]
	}`)
	if cfg.Training.DiffusionHead != "" {
		t.Fatalf("diffusion_head=%q, want empty", cfg.Training.DiffusionHead)
	}
	if cfg.Training.Heads[1].OutputHead != MultiheadOutputBinary {
		t.Fatalf("detector output_head=%q, want binary", cfg.Training.Heads[1].OutputHead)
	}
	shapes, err := CollectWeightShapesFromConfig(cfg)
	if err != nil {
		t.Fatalf("CollectWeightShapesFromConfig: %v", err)
	}
	for _, want := range []string{"head_detector_binary_proj", "head_detector_binary_bias"} {
		if countWeightNames(shapes, want) != 1 {
			t.Fatalf("weight %q count=%d, want 1 in %+v", want, countWeightNames(shapes, want), shapes)
		}
	}
	prog, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig: %v", err)
	}
	if prog.NumWeights != len(shapes) {
		t.Fatalf("prog weights=%d shapes=%d", prog.NumWeights, len(shapes))
	}
	if countOps(prog, OpMaskedCrossEntropy) != 1 {
		t.Fatalf("OpMaskedCrossEntropy count=%d, want 1", countOps(prog, OpMaskedCrossEntropy))
	}
	if countOps(prog, OpMaskedBCEWithLogits) != 1 {
		t.Fatalf("OpMaskedBCEWithLogits count=%d, want 1", countOps(prog, OpMaskedBCEWithLogits))
	}
	if countOps(prog, OpMaskedBinaryAccuracy) != 1 {
		t.Fatalf("OpMaskedBinaryAccuracy count=%d, want 1", countOps(prog, OpMaskedBinaryAccuracy))
	}
	for _, want := range []string{"head_scorer_logits", "head_detector_logits", "head_detector_accuracy"} {
		if !programDeclaresOutputArch(prog, want) {
			t.Fatalf("missing output %q: %+v", want, prog.Outputs)
		}
	}
}

func TestMultiheadWeightShapesAndIR(t *testing.T) {
	cfg := parseMultiheadConfig(t, `"training": {
		"objective": "multihead",
		"steps": 1,
		"lr": 0.001,
		"batch_tokens": 8,
		"mlm_mask_token_id": 31,
		"heads": [
			{"name": "scorer", "objective": "mntp", "loss_weight": 0.7, "layer_aggregation": "dwa"},
			{"name": "denoiser", "objective": "block_diffusion", "loss_weight": 0.3,
			 "diffusion": {"block_size": 2, "timestep_conditioning": "adaln", "timestep_conditioning_dim": 5}}
		]
	}`)
	shapes, err := CollectWeightShapesFromConfig(cfg)
	if err != nil {
		t.Fatalf("CollectWeightShapesFromConfig: %v", err)
	}
	wantNames := []string{
		"embed",
		"norm_scale",
		"wq",
		"wk",
		"wv",
		"wo",
		"ff1",
		"ff2",
		"adaln_0_w1",
		"adaln_0_w2",
		"head_scorer_dwa_alpha",
		"head_scorer_final_norm_scale",
		"head_scorer_mlm_dense",
		"head_scorer_mlm_dense_bias",
		"head_scorer_mlm_output_bias",
		"head_denoiser_final_norm_scale",
		"head_denoiser_proj",
	}
	if len(shapes) != len(wantNames) {
		t.Fatalf("shape count=%d want %d: %+v", len(shapes), len(wantNames), shapes)
	}
	for i, want := range wantNames {
		if shapes[i].Name != want {
			t.Fatalf("shape[%d].Name=%q want %q", i, shapes[i].Name, want)
		}
	}
	prog, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig: %v", err)
	}
	if prog.NumWeights != len(shapes) {
		t.Fatalf("prog weights=%d shapes=%d", prog.NumWeights, len(shapes))
	}
	if !programDeclaresInputArch(prog, "diffusion_timestep") {
		t.Fatal("multihead AdaLN program should declare diffusion_timestep")
	}
	if countOps(prog, OpBlockDiffusionMask) != 1 {
		t.Fatalf("OpBlockDiffusionMask count=%d, want 1", countOps(prog, OpBlockDiffusionMask))
	}
	if countOps(prog, OpMaskedCrossEntropy) != 2 {
		t.Fatalf("OpMaskedCrossEntropy count=%d, want 2", countOps(prog, OpMaskedCrossEntropy))
	}
	if !programDeclaresOutputArch(prog, "head_scorer_loss") || !programDeclaresOutputArch(prog, "head_denoiser_loss") {
		t.Fatalf("head loss outputs missing: %+v", prog.Outputs)
	}
	if !programDeclaresOutputArch(prog, "head_denoiser_logits") {
		t.Fatalf("denoiser logits output missing: %+v", prog.Outputs)
	}
	if _, err := BuildEvalIRProgramFromConfig(cfg); err == nil {
		t.Fatal("BuildEvalIRProgramFromConfig should reject multihead")
	}
}

func TestMultiheadGPTBERTStyleTrunkWeightAccounting(t *testing.T) {
	cfg := parseMultiheadConfig(t, `
		"model_dim": 32,
		"vocab_size": 64,
		"seq_len": 8,
		"mlp_mult": 2.0,
		"norm_type": "layernorm",
		"norm_affine": false,
		"ffn_internal_norm": true,
		"blocks": [
			{"type": "plain", "heads": 4, "relative_attention": "deberta_p2c_c2p", "relative_attention_window": 4,
			 "relative_attention_parameterization": "shared_qk_reuse", "attn_bias": true, "attn_value_gate": true,
			 "attn_post_norm": "before_outproj", "ffn_activation": "geglu", "ffn_pre_norm": true},
			{"type": "plain", "heads": 4, "relative_attention": "deberta_p2c_c2p", "relative_attention_window": 4,
			 "relative_attention_parameterization": "shared_qk_reuse", "attn_bias": true, "attn_value_gate": true,
			 "attn_post_norm": "before_outproj", "ffn_activation": "geglu", "ffn_pre_norm": true},
			{"type": "plain", "heads": 4, "relative_attention": "deberta_p2c_c2p", "relative_attention_window": 4,
			 "relative_attention_parameterization": "shared_qk_reuse", "attn_bias": true, "attn_value_gate": true,
			 "attn_post_norm": "before_outproj", "ffn_activation": "geglu", "ffn_pre_norm": true}
		],
		"training": {
			"objective": "multihead",
			"steps": 1,
			"lr": 0.001,
			"batch_tokens": 16,
			"mlm_mask_token_id": 63,
			"heads": [
				{"name": "scorer", "objective": "mntp", "loss_weight": 0.7, "layer_aggregation": "dwa"},
				{"name": "denoiser", "objective": "block_diffusion", "loss_weight": 0.3,
				 "diffusion": {"block_size": 4, "timestep_conditioning": "adaln", "timestep_conditioning_dim": 6}}
			]
		}`)
	shapes, err := CollectWeightShapesFromConfig(cfg)
	if err != nil {
		t.Fatalf("CollectWeightShapesFromConfig: %v", err)
	}
	prog, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig: %v", err)
	}
	if prog.NumWeights != len(shapes) {
		t.Fatalf("prog weights=%d shapes=%d", prog.NumWeights, len(shapes))
	}
	if countWeightNames(shapes, SharedRelativeEmbeddingsWeightName) != 1 {
		t.Fatalf("shared relative embedding count=%d, want 1", countWeightNames(shapes, SharedRelativeEmbeddingsWeightName))
	}
	if countWeightNames(shapes, "relative_embeddings") != 0 || countWeightNames(shapes, "w_pos_key") != 0 || countWeightNames(shapes, "w_pos_query") != 0 {
		t.Fatalf("shared_qk_reuse should not emit per-block relative projection weights: %+v", shapes)
	}
}

func parseMultiheadConfig(t *testing.T, body string) *ArchConfig {
	t.Helper()
	cfg, err := ParseArchConfig([]byte(multiheadConfigJSON(body)), "multihead_test")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	return cfg
}

func multiheadConfigJSON(body string) string {
	if !strings.Contains(body, `"blocks"`) {
		body = `"blocks": [{"type": "plain", "heads": 2}], ` + body
	}
	return `{
		"name": "multihead_test",
		"model_dim": 16,
		"vocab_size": 32,
		"seq_len": 4,
		` + body + `
	}`
}

func countWeightNames(shapes []WeightMeta, name string) int {
	count := 0
	for _, shape := range shapes {
		if shape.Name == name {
			count++
		}
	}
	return count
}
