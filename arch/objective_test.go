package arch

import (
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
