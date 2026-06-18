package arch

import (
	"math"
	"strings"
	"testing"
)

func TestTrainingObjectiveDefaultsCausal(t *testing.T) {
	cfg := parseObjectiveConfig(t, `"training": {"steps": 1, "lr": 0.001, "batch_tokens": 8}`)
	if cfg.Training.Objective != ObjectiveCausal {
		t.Fatalf("objective = %q, want %q", cfg.Training.Objective, ObjectiveCausal)
	}
	if cfg.Training.MLMMaskProb != 0.15 {
		t.Fatalf("mlm_mask_prob = %g, want 0.15", cfg.Training.MLMMaskProb)
	}
	if cfg.Training.MLMMaskTokenProb != 0.8 || cfg.Training.MLMRandomTokenProb != 0.1 || cfg.Training.MLMKeptUnchangedProb != 0.1 {
		t.Fatalf("replacement probs = %g/%g/%g, want 0.8/0.1/0.1",
			cfg.Training.MLMMaskTokenProb, cfg.Training.MLMRandomTokenProb, cfg.Training.MLMKeptUnchangedProb)
	}
	if cfg.Training.HybridCLMFraction != 0.5 {
		t.Fatalf("hybrid_clm_fraction = %g, want 0.5", cfg.Training.HybridCLMFraction)
	}
	if cfg.Training.HybridSecondaryObjective != ObjectiveMNTP {
		t.Fatalf("hybrid_secondary_objective = %q, want %q", cfg.Training.HybridSecondaryObjective, ObjectiveMNTP)
	}
}

func TestTrainingObjectiveValidation(t *testing.T) {
	tests := []struct {
		name     string
		training string
		wantErr  string
	}{
		{
			name:     "mlm requires mask token",
			training: `"training": {"steps": 1, "lr": 0.001, "batch_tokens": 8, "objective": "mlm"}`,
			wantErr:  "mlm_mask_token_id is required",
		},
		{
			name: "replacement probabilities sum to one",
			training: `"training": {
				"steps": 1, "lr": 0.001, "batch_tokens": 8,
				"objective": "mlm", "mlm_mask_token_id": 7,
				"mlm_mask_token_prob": 0.7,
				"mlm_random_token_prob": 0.1,
				"mlm_kept_unchanged_prob": 0.1
			}`,
			wantErr: "must sum to 1.0",
		},
		{
			name: "hybrid secondary must be masked",
			training: `"training": {
				"steps": 1, "lr": 0.001, "batch_tokens": 8,
				"objective": "hybrid", "mlm_mask_token_id": 7,
				"hybrid_secondary_objective": "causal"
			}`,
			wantErr: "hybrid_secondary_objective",
		},
		{
			name: "window size requires causal resolved mask",
			training: `"blocks": [{"type": "plain", "heads": 2, "window_size": 4}],
				"training": {"steps": 1, "lr": 0.001, "batch_tokens": 8, "objective": "mlm", "mlm_mask_token_id": 7}`,
			wantErr: "window_size",
		},
		{
			name: "hybrid window size rejects masked secondary",
			training: `"blocks": [{"type": "plain", "heads": 2, "window_size": 4}],
				"training": {
					"steps": 1, "lr": 0.001, "batch_tokens": 8,
					"objective": "hybrid", "mlm_mask_token_id": 7,
					"hybrid_clm_fraction": 0.5
				}`,
			wantErr: "masked secondary",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			raw := objectiveConfigJSON(tt.training)
			_, err := ParseArchConfig([]byte(raw), tt.name)
			if err == nil {
				t.Fatal("expected validation error")
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("error = %q, want substring %q", err, tt.wantErr)
			}
		})
	}
}

func TestTrainingObjectiveValidation_HybridWindowSizeAllowedWhenCausalOnly(t *testing.T) {
	_ = parseObjectiveConfig(t, `"blocks": [{"type": "plain", "heads": 2, "window_size": 4}],
		"training": {
			"steps": 1, "lr": 0.001, "batch_tokens": 8,
			"objective": "hybrid", "mlm_mask_token_id": 7,
			"hybrid_clm_fraction": 1.0
		}`)
}

func TestTrainingRecipeKnobValidationAndDefaults(t *testing.T) {
	cfg := parseObjectiveConfig(t, `"training": {
		"steps": 10,
		"lr": 0.001,
		"batch_tokens": 8,
		"z_loss": 0.0001,
		"seq_len_schedule": [[0,2],[5,4]],
		"mlm_mask_prob_schedule": [[0,0.3],[5,0.15]],
		"early_stop": {"metric": "val", "patience": 3, "min_delta": 0.01, "min_steps": 10}
	}`)
	if cfg.Training.ZLoss != 0.0001 {
		t.Fatalf("z_loss=%g", cfg.Training.ZLoss)
	}
	if got := cfg.Training.EffectiveSeqLenForStep(cfg.SeqLen, 0); got != 2 {
		t.Fatalf("seq len step0=%d, want 2", got)
	}
	if got := cfg.Training.EffectiveSeqLenForStep(cfg.SeqLen, 5); got != 4 {
		t.Fatalf("seq len step5=%d, want 4", got)
	}
	if got := cfg.Training.EffectiveMLMMaskProbForStep(0); got != 0.3 {
		t.Fatalf("mask prob step0=%g, want 0.3", got)
	}
	if got := cfg.Training.EffectiveMLMMaskProbForStep(5); got != 0.15 {
		t.Fatalf("mask prob step5=%g, want 0.15", got)
	}
	if cfg.Training.MLMMaskProbScheduleMode != "step" {
		t.Fatalf("mlm_mask_prob_schedule_mode=%q, want step", cfg.Training.MLMMaskProbScheduleMode)
	}
	if cfg.Training.EarlyStop == nil || cfg.Training.EarlyStop.Patience != 3 {
		t.Fatalf("early_stop=%+v, want patience 3", cfg.Training.EarlyStop)
	}
}

func TestEffectiveMLMMaskProbScheduleLinear(t *testing.T) {
	cfg := parseObjectiveConfig(t, `"training": {
		"batch_tokens": 8,
		"mlm_mask_prob": 0.9,
		"mlm_mask_prob_schedule": [[0,0.3],[10,0.15]],
		"mlm_mask_prob_schedule_mode": "linear"
	}`)
	if got := cfg.Training.MLMMaskProbScheduleMode; got != "linear" {
		t.Fatalf("mlm_mask_prob_schedule_mode=%q, want linear", got)
	}
	tests := []struct {
		step int
		want float64
	}{
		{step: 0, want: 0.3},
		{step: 5, want: 0.225},
		{step: 10, want: 0.15},
		{step: 20, want: 0.15},
	}
	for _, tt := range tests {
		if got := cfg.Training.EffectiveMLMMaskProbForStep(tt.step); math.Abs(got-tt.want) > 1e-12 {
			t.Fatalf("EffectiveMLMMaskProbForStep(%d)=%g, want %g", tt.step, got, tt.want)
		}
	}
}

func TestTrainingRecipeKnobValidationErrors(t *testing.T) {
	tests := []struct {
		name    string
		body    string
		wantErr string
	}{
		{name: "bad z loss", body: `"training": {"batch_tokens": 8, "z_loss": -1}`, wantErr: "z_loss"},
		{name: "seq schedule must start at zero", body: `"training": {"batch_tokens": 8, "seq_len_schedule": [[1,2]]}`, wantErr: "first step must be 0"},
		{name: "seq schedule value bounded", body: `"training": {"batch_tokens": 8, "seq_len_schedule": [[0,8]]}`, wantErr: "must be in [1, seq_len=4]"},
		{name: "seq schedule divides batch", body: `"training": {"batch_tokens": 8, "seq_len_schedule": [[0,3]]}`, wantErr: "must be divisible"},
		{name: "mask schedule integer step", body: `"training": {"batch_tokens": 8, "mlm_mask_prob_schedule": [[0.5,0.2]]}`, wantErr: "step must be an integer"},
		{name: "mask schedule probability", body: `"training": {"batch_tokens": 8, "mlm_mask_prob_schedule": [[0,1.5]]}`, wantErr: "must be in [0,1]"},
		{name: "mask schedule mode", body: `"training": {"batch_tokens": 8, "mlm_mask_prob_schedule": [[0,0.2]], "mlm_mask_prob_schedule_mode": "spline"}`, wantErr: "mlm_mask_prob_schedule_mode"},
		{name: "early stop metric", body: `"training": {"batch_tokens": 8, "early_stop": {"metric": "train", "patience": 2}}`, wantErr: "early_stop.metric"},
		{name: "early stop patience", body: `"training": {"batch_tokens": 8, "early_stop": {"patience": -1}}`, wantErr: "early_stop.patience"},
		{name: "early stop val gt", body: `"training": {"batch_tokens": 8, "early_stop": {"val_gt": -1}}`, wantErr: "early_stop.val_gt"},
		{name: "seq schedule rejects distillation", body: `"training": {"batch_tokens": 8, "seq_len_schedule": [[0,2]], "distillation": {"teacher_checkpoints":["a"], "teacher_configs":["b"], "loss_weight_ce":1, "loss_weight_kl":0, "ensemble_strategy":"mean_logits"}}`, wantErr: "seq_len_schedule"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ParseArchConfig([]byte(objectiveConfigJSON(tt.body)), "recipe_error")
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("ParseArchConfig error=%v, want containing %q", err, tt.wantErr)
			}
		})
	}
}

func TestBuildIRProgram_ZLossAddsTrainingOnlyLoss(t *testing.T) {
	cfg := parseObjectiveConfig(t, `"training": {"steps": 1, "lr": 0.001, "batch_tokens": 8, "z_loss": 0.0001}`)
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}
	if n := countOps(prog, OpZLoss); n != 1 {
		t.Fatalf("OpZLoss count=%d, want 1", n)
	}
	if !programDeclaresOutputArch(prog, "eval_loss") {
		t.Fatal("z_loss program should expose eval_loss for CE-only validation")
	}
	evalProg, err := BuildEvalIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildEvalIRProgramFromConfig: %v", err)
	}
	if n := countOps(evalProg, OpZLoss); n != 0 {
		t.Fatalf("eval OpZLoss count=%d, want 0", n)
	}
}

func TestBuildIRProgram_MLMUsesMaskedLossAndBidirectionalPlainAttention(t *testing.T) {
	cfg := parseObjectiveConfig(t, `"training": {
		"steps": 1,
		"lr": 0.001,
		"batch_tokens": 8,
		"objective": "mlm",
		"mlm_mask_token_id": 7
	}`)
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}
	if !programDeclaresInputArch(prog, "loss_mask") {
		t.Fatal("masked objective program missing loss_mask input")
	}
	if !programDeclaresOutputArch(prog, "eval_loss") {
		t.Fatal("masked objective program missing eval_loss output")
	}
	if n := countOps(prog, OpMaskedCrossEntropy); n != 1 {
		t.Fatalf("OpMaskedCrossEntropy count = %d, want 1", n)
	}
	if n := countOps(prog, OpMaskedCEPerToken); n != 1 {
		t.Fatalf("OpMaskedCEPerToken count = %d, want 1", n)
	}
	if n := countOps(prog, OpCausalMask); n != 0 {
		t.Fatalf("MLM default plain attention emitted %d causal masks, want 0", n)
	}
}

func TestBuildTrainingIRProgram_HybridForcesConcreteAttentionMasks(t *testing.T) {
	for _, mode := range []string{"", AttentionMaskBidirectional, AttentionMaskNone} {
		t.Run("mode_"+mode, func(t *testing.T) {
			block := `{"type": "plain", "heads": 2}`
			if mode != "" {
				block = `{"type": "plain", "heads": 2, "attention_mask": "` + mode + `"}`
			}
			cfg := parseObjectiveConfig(t, `"blocks": [`+block+`],
				"training": {
					"steps": 1,
					"lr": 0.001,
					"batch_tokens": 8,
					"objective": "hybrid",
					"mlm_mask_token_id": 7,
					"hybrid_secondary_objective": "mlm"
				}`)
			causalProg, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{
				RecurrenceActive: true,
				Objective:        ObjectiveCausal,
			})
			if err != nil {
				t.Fatalf("BuildTrainingIRProgramFromConfig(causal): %v", err)
			}
			if n := countOps(causalProg, OpCausalMask); n != 1 {
				t.Fatalf("hybrid causal program emitted %d causal masks, want 1", n)
			}
			if programDeclaresInputArch(causalProg, "loss_mask") {
				t.Fatal("hybrid causal program declares loss_mask")
			}

			maskedProg, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{
				RecurrenceActive: true,
				Objective:        ObjectiveMLM,
			})
			if err != nil {
				t.Fatalf("BuildTrainingIRProgramFromConfig(mlm): %v", err)
			}
			if n := countOps(maskedProg, OpCausalMask); n != 0 {
				t.Fatalf("hybrid masked program emitted %d causal masks, want 0", n)
			}
			if !programDeclaresInputArch(maskedProg, "loss_mask") {
				t.Fatal("hybrid masked program missing loss_mask")
			}
		})
	}
}

func TestBuildEvalIRProgram_HybridUsesCausalAttention(t *testing.T) {
	cfg := parseObjectiveConfig(t, `"blocks": [{"type": "plain", "heads": 2, "attention_mask": "bidirectional"}],
		"training": {
			"steps": 1,
			"lr": 0.001,
			"batch_tokens": 8,
			"objective": "hybrid",
			"mlm_mask_token_id": 7
		}`)
	prog, err := BuildEvalIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildEvalIRProgramFromConfig: %v", err)
	}
	if n := countOps(prog, OpCausalMask); n != 1 {
		t.Fatalf("hybrid eval program emitted %d causal masks, want 1", n)
	}
	if programDeclaresInputArch(prog, "loss_mask") {
		t.Fatal("hybrid eval program declares loss_mask")
	}
}

func TestBuildIRProgram_PlainAttentionMaskBidirectionalAndNoneOmitCausalMask(t *testing.T) {
	for _, mode := range []string{AttentionMaskBidirectional, AttentionMaskNone} {
		t.Run(mode, func(t *testing.T) {
			cfg := parseObjectiveConfig(t, `"blocks": [{"type": "plain", "heads": 2, "attention_mask": "`+mode+`"}],
				"training": {"steps": 1, "lr": 0.001, "batch_tokens": 8}`)
			prog, err := BuildIRProgramFromConfig(cfg)
			if err != nil {
				t.Fatalf("BuildIRProgramFromConfig: %v", err)
			}
			if n := countOps(prog, OpCausalMask); n != 0 {
				t.Fatalf("attention_mask=%s emitted %d causal masks, want 0", mode, n)
			}
		})
	}
}

func parseObjectiveConfig(t *testing.T, body string) *ArchConfig {
	t.Helper()
	cfg, err := ParseArchConfig([]byte(objectiveConfigJSON(body)), "objective_test")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	return cfg
}

func objectiveConfigJSON(body string) string {
	if !strings.Contains(body, `"blocks"`) {
		body = `"blocks": [{"type": "plain", "heads": 2}], ` + body
	}
	return `{
		"name": "objective_test",
		"model_dim": 16,
		"vocab_size": 32,
		"seq_len": 4,
		` + body + `
	}`
}

func programDeclaresInputArch(prog *Program, name string) bool {
	for _, in := range prog.Inputs {
		if in.Name == name {
			return true
		}
	}
	return false
}

func programDeclaresOutputArch(prog *Program, name string) bool {
	for _, out := range prog.Outputs {
		if out.Name == name {
			return true
		}
	}
	return false
}
