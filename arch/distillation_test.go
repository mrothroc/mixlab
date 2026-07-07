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
	masked := parseObjectiveConfig(t, `"training": {
		"steps": 1,
		"lr": 0.001,
		"batch_tokens": 8,
		"objective": "mntp",
		"mlm_mask_token_id": 7,
		"distillation": {
			"teacher_checkpoints": ["teacher_1.safetensors"],
			"teacher_configs": ["teacher_1.json"],
			"loss_weight_kl": 0.5,
			"loss_weight_ce": 0.5,
			"temperature": 2.0
		}
	}`)
	if masked.Training.Distillation == nil || masked.Training.Distillation.Temperature != 2.0 {
		t.Fatalf("masked distillation defaults = %+v", masked.Training.Distillation)
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
			name: "bad temperature",
			body: `"training": {"steps": 1, "lr": 0.001, "batch_tokens": 8,
				"objective": "mlm", "mlm_mask_token_id": 7,
				"distillation": {
					"teacher_checkpoints": ["teacher_1.safetensors"],
					"teacher_configs": ["teacher_1.json"],
					"loss_weight_kl": 0.5,
					"loss_weight_ce": 0.5,
					"temperature": -1
				}}`,
			wantErr: "temperature",
		},
		{
			name: "kl disabled requires ce parity",
			body: `"training": {"steps": 1, "lr": 0.001, "batch_tokens": 8,
				"objective": "mlm", "mlm_mask_token_id": 7,
				"distillation": {
					"loss_weight_kl": 0.0,
					"loss_weight_ce": 0.5
				}}`,
			wantErr: "loss_weight_ce=1",
		},
		{
			name: "hybrid block diffusion rejected",
			body: `"training": {"steps": 1, "lr": 0.001, "batch_tokens": 8,
				"objective": "hybrid", "hybrid_secondary_objective": "block_diffusion", "mlm_mask_token_id": 7,
				"distillation": {
					"teacher_checkpoints": ["teacher_1.safetensors"],
					"teacher_configs": ["teacher_1.json"],
					"loss_weight_kl": 0.5,
					"loss_weight_ce": 0.5
				}}`,
			wantErr: "training.distillation",
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
	if n := countOps(prog, OpMaskedDistillationKL); n != 0 {
		t.Fatalf("OpMaskedDistillationKL count = %d, want 0", n)
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

func TestBuildIRProgram_MaskedDistillationUsesMaskedKL(t *testing.T) {
	cfg := parseObjectiveConfig(t, `"training": {
		"steps": 1,
		"lr": 0.001,
		"batch_tokens": 8,
		"objective": "mlm",
		"mlm_mask_token_id": 7,
		"distillation": {
			"teacher_checkpoints": ["teacher_1.safetensors"],
			"teacher_configs": ["teacher_1.json"],
			"loss_weight_kl": 0.5,
			"loss_weight_ce": 0.5,
			"temperature": 2.0
		}
	}`)
	prog, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{Objective: ObjectiveMLM})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig: %v", err)
	}
	if !programDeclaresInputArch(prog, "teacher_probs") || !programDeclaresInputArch(prog, "loss_mask") {
		t.Fatal("masked distillation program missing teacher_probs/loss_mask inputs")
	}
	if n := countOps(prog, OpMaskedCrossEntropy); n != 1 {
		t.Fatalf("OpMaskedCrossEntropy count = %d, want 1", n)
	}
	if n := countOps(prog, OpMaskedDistillationKL); n != 1 {
		t.Fatalf("OpMaskedDistillationKL count = %d, want 1", n)
	}
	if n := countOps(prog, OpDistillationKL); n != 0 {
		t.Fatalf("OpDistillationKL count = %d, want 0", n)
	}
	if !programDeclaresOutputArch(prog, "eval_loss") || !programDeclaresOutputArch(prog, "per_token_nll") {
		t.Fatal("masked distillation program missing eval_loss/per_token_nll")
	}
}

func TestBuildIRProgram_DistillationKLZeroIsInactive(t *testing.T) {
	cfg := parseObjectiveConfig(t, `"training": {
		"steps": 1,
		"lr": 0.001,
		"batch_tokens": 8,
		"objective": "mlm",
		"mlm_mask_token_id": 7,
		"distillation": {
			"loss_weight_kl": 0.0,
			"loss_weight_ce": 1.0
		}
	}`)
	prog, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{Objective: ObjectiveMLM})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig: %v", err)
	}
	if programDeclaresInputArch(prog, "teacher_probs") {
		t.Fatal("kl=0 distillation unexpectedly declares teacher_probs")
	}
	if n := countOps(prog, OpMaskedDistillationKL) + countOps(prog, OpDistillationKL); n != 0 {
		t.Fatalf("kl=0 distillation op count = %d, want 0", n)
	}
	if n := countOps(prog, OpMaskedCrossEntropy); n != 1 {
		t.Fatalf("OpMaskedCrossEntropy count = %d, want 1", n)
	}
}

func TestBuildIRProgram_HybridDistillationConcreteObjectives(t *testing.T) {
	cfg := parseObjectiveConfig(t, `"training": {
		"steps": 1,
		"lr": 0.001,
		"batch_tokens": 8,
		"objective": "hybrid",
		"hybrid_mix_granularity": "example",
		"hybrid_secondary_objective": "mntp",
		"hybrid_clm_fraction": 0.5,
		"mlm_mask_token_id": 7,
		"distillation": {
			"teacher_checkpoints": ["teacher_1.safetensors"],
			"teacher_configs": ["teacher_1.json"],
			"loss_weight_kl": 0.5,
			"loss_weight_ce": 0.5
		}
	}`)
	exampleProg, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{Objective: ObjectiveHybridExample})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig hybrid_example: %v", err)
	}
	if !programDeclaresInputArch(exampleProg, "attention_causal_mask") || !programDeclaresInputArch(exampleProg, "teacher_probs") || !programDeclaresInputArch(exampleProg, "distill_loss_mask") {
		t.Fatal("hybrid-example distillation program missing attention_causal_mask/teacher_probs/distill_loss_mask")
	}
	maskedKLOps := 0
	for _, op := range exampleProg.Ops {
		if op.Code != OpMaskedDistillationKL {
			continue
		}
		maskedKLOps++
		if len(op.Inputs) != 3 || op.Inputs[2] != "distill_loss_mask" {
			t.Fatalf("hybrid-example KL inputs=%v, want distill_loss_mask as mask", op.Inputs)
		}
	}
	if maskedKLOps != 1 {
		t.Fatalf("hybrid-example masked KL count=%d want 1", maskedKLOps)
	}

	causalProg, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{Objective: ObjectiveCausal})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig causal: %v", err)
	}
	if programDeclaresInputArch(causalProg, "teacher_probs") {
		t.Fatal("hybrid causal concrete program unexpectedly declares teacher_probs")
	}
}

func TestBuildDistillationTeacherIRProgramUsesMaskedHead(t *testing.T) {
	cfg := parseObjectiveConfig(t, `"tie_embeddings": true,
		"mlm_head": "bert",
		"training": {"steps": 1, "batch_tokens": 8, "objective": "mlm", "mlm_mask_token_id": 7}`)
	prog, err := BuildDistillationTeacherIRProgramFromConfig(cfg, ObjectiveMLM)
	if err != nil {
		t.Fatalf("BuildDistillationTeacherIRProgramFromConfig: %v", err)
	}
	if !hasOpOutput(prog, OpMatMul, "mlm_head_dense_mm") {
		t.Fatal("masked distillation teacher did not emit BERT MLM head")
	}
	if !programDeclaresInputArch(prog, "loss_mask") {
		t.Fatal("masked distillation teacher program missing loss_mask input")
	}
}
