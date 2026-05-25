package arch

import (
	"strings"
	"testing"
)

func TestTrainingDistillationDefaultsDisabled(t *testing.T) {
	cfg := parseObjectiveConfig(t, `"training": {"steps": 1, "lr": 0.001, "batch_tokens": 8}`)
	if cfg.Training.Distillation != nil {
		t.Fatalf("distillation default = %+v, want nil", cfg.Training.Distillation)
	}
}

func TestTrainingDistillationValidation(t *testing.T) {
	valid := parseObjectiveConfig(t, `"training": {
		"steps": 1,
		"lr": 0.001,
		"batch_tokens": 8,
		"distillation": {
			"teacher_checkpoints": ["teacher_1.safetensors", "teacher_2.safetensors"],
			"teacher_configs": ["teacher_1.json", "teacher_2.json"],
			"loss_weight_kl": 0.5,
			"loss_weight_ce": 0.5,
			"ensemble_strategy": "mean_logprobs"
		}
	}`)
	if valid.Training.Distillation == nil {
		t.Fatal("distillation block missing")
	}
	if valid.Training.Distillation.EnsembleStrategy != DistillationMeanLogProbs {
		t.Fatalf("ensemble_strategy = %q, want %q", valid.Training.Distillation.EnsembleStrategy, DistillationMeanLogProbs)
	}

	tests := []struct {
		name    string
		body    string
		wantErr string
	}{
		{
			name: "mismatched teacher arrays",
			body: `"training": {"steps": 1, "lr": 0.001, "batch_tokens": 8,
				"distillation": {
					"teacher_checkpoints": ["teacher_1.safetensors"],
					"teacher_configs": ["teacher_1.json", "teacher_2.json"],
					"loss_weight_kl": 0.5,
					"loss_weight_ce": 0.5
				}}`,
			wantErr: "must match",
		},
		{
			name: "zero weights",
			body: `"training": {"steps": 1, "lr": 0.001, "batch_tokens": 8,
				"distillation": {
					"teacher_checkpoints": ["teacher_1.safetensors"],
					"teacher_configs": ["teacher_1.json"],
					"loss_weight_kl": 0.0,
					"loss_weight_ce": 0.0
				}}`,
			wantErr: "sum to > 0",
		},
		{
			name: "bad strategy",
			body: `"training": {"steps": 1, "lr": 0.001, "batch_tokens": 8,
				"distillation": {
					"teacher_checkpoints": ["teacher_1.safetensors"],
					"teacher_configs": ["teacher_1.json"],
					"loss_weight_kl": 0.5,
					"loss_weight_ce": 0.5,
					"ensemble_strategy": "median"
				}}`,
			wantErr: "ensemble_strategy",
		},
		{
			name: "non causal objective",
			body: `"training": {"steps": 1, "lr": 0.001, "batch_tokens": 8,
				"objective": "mlm", "mlm_mask_token_id": 7,
				"distillation": {
					"teacher_checkpoints": ["teacher_1.safetensors"],
					"teacher_configs": ["teacher_1.json"],
					"loss_weight_kl": 0.5,
					"loss_weight_ce": 0.5
				}}`,
			wantErr: "only supported",
		},
		{
			name: "mtp rejected",
			body: `"mtp": {"n": 2},
				"training": {"steps": 1, "lr": 0.001, "batch_tokens": 8,
				"distillation": {
					"teacher_checkpoints": ["teacher_1.safetensors"],
					"teacher_configs": ["teacher_1.json"],
					"loss_weight_kl": 0.5,
					"loss_weight_ce": 0.5
				}}`,
			wantErr: "cannot be combined",
		},
		{
			name: "first byte mask rejected",
			body: `"training": {"steps": 1, "lr": 0.001, "batch_tokens": 8,
				"first_byte_mask": true,
				"distillation": {
					"teacher_checkpoints": ["teacher_1.safetensors"],
					"teacher_configs": ["teacher_1.json"],
					"loss_weight_kl": 0.5,
					"loss_weight_ce": 0.5
				}}`,
			wantErr: "first_byte_mask",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ParseArchConfig([]byte(objectiveConfigJSON(tt.body)), tt.name)
			if err == nil {
				t.Fatal("expected validation error")
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("error = %q, want substring %q", err, tt.wantErr)
			}
		})
	}
}

func TestBuildIRProgram_DistillationAddsTeacherLossInput(t *testing.T) {
	cfg := parseObjectiveConfig(t, `"training": {
		"steps": 1,
		"lr": 0.001,
		"batch_tokens": 8,
		"distillation": {
			"teacher_checkpoints": ["teacher_1.safetensors"],
			"teacher_configs": ["teacher_1.json"],
			"loss_weight_kl": 0.5,
			"loss_weight_ce": 0.5
		}
	}`)
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}
	if !programDeclaresInputArch(prog, "teacher_probs") {
		t.Fatal("distillation program missing teacher_probs input")
	}
	if !programDeclaresOutputArch(prog, "eval_loss") {
		t.Fatal("distillation program missing eval_loss output")
	}
	if n := countOps(prog, OpDistillationKL); n != 1 {
		t.Fatalf("OpDistillationKL count = %d, want 1", n)
	}
	if n := countOps(prog, OpCrossEntropy); n != 1 {
		t.Fatalf("OpCrossEntropy count = %d, want 1", n)
	}
	if n := countOps(prog, OpCrossEntropyPerToken); n != 1 {
		t.Fatalf("OpCrossEntropyPerToken count = %d, want 1", n)
	}

	evalProg, err := BuildEvalIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildEvalIRProgramFromConfig: %v", err)
	}
	if programDeclaresInputArch(evalProg, "teacher_probs") {
		t.Fatal("eval program unexpectedly declares teacher_probs")
	}
	if n := countOps(evalProg, OpDistillationKL); n != 0 {
		t.Fatalf("eval OpDistillationKL count = %d, want 0", n)
	}
}
