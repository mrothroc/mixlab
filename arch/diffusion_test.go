package arch

import (
	"strings"
	"testing"
)

func TestBlockDiffusionConfigMinimalDefaults(t *testing.T) {
	cfg := parseObjectiveConfig(t, `"training": {
		"steps": 1,
		"lr": 0.001,
		"batch_tokens": 8,
		"objective": "block_diffusion",
		"mlm_mask_token_id": 7
	}`)
	if cfg.Training.Objective != ObjectiveBlockDiffusion {
		t.Fatalf("objective=%q, want %q", cfg.Training.Objective, ObjectiveBlockDiffusion)
	}
	if cfg.Training.Diffusion == nil {
		t.Fatal("training.diffusion default is nil")
	}
	d := cfg.Training.Diffusion
	if d.BlockSize != cfg.SeqLen {
		t.Fatalf("diffusion.block_size=%d, want seq_len=%d", d.BlockSize, cfg.SeqLen)
	}
	if d.StepsPerBlock != d.BlockSize {
		t.Fatalf("diffusion.steps_per_block=%d, want block_size=%d", d.StepsPerBlock, d.BlockSize)
	}
	if d.MinMaskFraction != 0.05 || d.MaxMaskFraction != 1.0 {
		t.Fatalf("diffusion mask fraction range=%g/%g, want 0.05/1.0", d.MinMaskFraction, d.MaxMaskFraction)
	}
	if d.ConfidenceThreshold != 0.8 {
		t.Fatalf("diffusion.confidence_threshold=%g, want 0.8", d.ConfidenceThreshold)
	}
	if d.CommitFloor != 1 {
		t.Fatalf("diffusion.commit_floor=%d, want 1", d.CommitFloor)
	}
}

func TestBlockDiffusionConfigExplicitValid(t *testing.T) {
	_ = parseObjectiveConfig(t, `"blocks": [
			{"type": "plain", "heads": 2},
			{"type": "swiglu"},
			{"type": "geglu"},
			{"type": "mlp", "activation": "gelu"},
			{"type": "moe", "num_experts": 2}
		],
		"training": {
			"steps": 1,
			"lr": 0.001,
			"batch_tokens": 8,
			"objective": "block_diffusion",
			"mlm_mask_token_id": 7,
			"diffusion": {
				"block_size": 2,
				"steps_per_block": 3,
				"min_mask_fraction": 0,
				"max_mask_fraction": 0.5,
				"confidence_threshold": 0,
				"commit_floor": 2
			}
		}`)
}

func TestHybridBlockDiffusionConfigDefaults(t *testing.T) {
	cfg := parseObjectiveConfig(t, `"training": {
		"steps": 1,
		"lr": 0.001,
		"batch_tokens": 8,
		"objective": "hybrid",
		"hybrid_secondary_objective": "block_diffusion",
		"hybrid_clm_fraction": 0.75,
		"mlm_mask_token_id": 7
	}`)
	if cfg.Training.EffectiveObjective() != ObjectiveHybrid {
		t.Fatalf("objective=%q, want %q", cfg.Training.EffectiveObjective(), ObjectiveHybrid)
	}
	if cfg.Training.EffectiveHybridSecondaryObjective() != ObjectiveBlockDiffusion {
		t.Fatalf("secondary=%q, want %q", cfg.Training.EffectiveHybridSecondaryObjective(), ObjectiveBlockDiffusion)
	}
	if !cfg.Training.UsesBlockDiffusionObjective() {
		t.Fatal("hybrid block_diffusion config should use block diffusion")
	}
	if cfg.Training.Diffusion == nil {
		t.Fatal("training.diffusion default is nil")
	}
	if cfg.Training.Diffusion.BlockSize != cfg.SeqLen {
		t.Fatalf("diffusion.block_size=%d, want seq_len=%d", cfg.Training.Diffusion.BlockSize, cfg.SeqLen)
	}
}

func TestBlockDiffusionObjectiveHelpers(t *testing.T) {
	spec := TrainingSpec{Objective: ObjectiveBlockDiffusion}
	if got := spec.EffectiveObjective(); got != ObjectiveBlockDiffusion {
		t.Fatalf("EffectiveObjective=%q, want %q", got, ObjectiveBlockDiffusion)
	}
	if got := spec.DefaultConcreteObjective(); got != ObjectiveBlockDiffusion {
		t.Fatalf("DefaultConcreteObjective=%q, want %q", got, ObjectiveBlockDiffusion)
	}
	if !spec.NeedsMaskedLoss() {
		t.Fatal("block_diffusion should need masked loss")
	}
	if (TrainingSpec{Objective: ObjectiveCausal}).NeedsMaskedLoss() {
		t.Fatal("causal should not need masked loss")
	}
}

func TestBlockDiffusionConfigValidation(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string
	}{
		{
			name: "missing mask token",
			body: `"training": {
				"steps": 1, "lr": 0.001, "batch_tokens": 8,
				"objective": "block_diffusion",
				"diffusion": {"block_size": 2}
			}`,
			want: "mlm_mask_token_id is required",
		},
		{
			name: "block size zero",
			body: `"training": {
				"steps": 1, "lr": 0.001, "batch_tokens": 8,
				"objective": "block_diffusion", "mlm_mask_token_id": 7,
				"diffusion": {"block_size": 0}
			}`,
			want: "block_size",
		},
		{
			name: "block size too large",
			body: `"training": {
				"steps": 1, "lr": 0.001, "batch_tokens": 8,
				"objective": "block_diffusion", "mlm_mask_token_id": 7,
				"diffusion": {"block_size": 8}
			}`,
			want: "must be <= seq_len",
		},
		{
			name: "block size must divide seq len",
			body: `"training": {
				"steps": 1, "lr": 0.001, "batch_tokens": 8,
				"objective": "block_diffusion", "mlm_mask_token_id": 7,
				"diffusion": {"block_size": 3}
			}`,
			want: "must divide seq_len",
		},
		{
			name: "steps per block zero",
			body: `"training": {
				"steps": 1, "lr": 0.001, "batch_tokens": 8,
				"objective": "block_diffusion", "mlm_mask_token_id": 7,
				"diffusion": {"block_size": 2, "steps_per_block": 0}
			}`,
			want: "steps_per_block",
		},
		{
			name: "stray diffusion config",
			body: `"training": {
				"steps": 1, "lr": 0.001, "batch_tokens": 8,
				"objective": "mlm", "mlm_mask_token_id": 7,
				"diffusion": {"block_size": 2}
			}`,
			want: "training.diffusion",
		},
		{
			name: "hybrid diffusion requires block secondary",
			body: `"training": {
				"steps": 1, "lr": 0.001, "batch_tokens": 8,
				"objective": "hybrid", "mlm_mask_token_id": 7,
				"hybrid_secondary_objective": "mntp",
				"diffusion": {"block_size": 2}
			}`,
			want: "training.diffusion",
		},
		{
			name: "hybrid block diffusion rejects example granularity",
			body: `"training": {
				"steps": 1, "lr": 0.001, "batch_tokens": 8,
				"objective": "hybrid", "mlm_mask_token_id": 7,
				"hybrid_secondary_objective": "block_diffusion",
				"hybrid_mix_granularity": "example",
				"diffusion": {"block_size": 2}
			}`,
			want: "hybrid_mix_granularity",
		},
		{
			name: "top level mtp",
			body: `"mtp": {"n": 2},
				"training": {
					"steps": 1, "lr": 0.001, "batch_tokens": 8,
					"objective": "block_diffusion", "mlm_mask_token_id": 7,
					"diffusion": {"block_size": 2}
				}`,
			want: "top-level mtp",
		},
		{
			name: "first byte mask",
			body: `"training": {
				"steps": 1, "lr": 0.001, "batch_tokens": 8,
				"objective": "block_diffusion", "mlm_mask_token_id": 7,
				"first_byte_mask": true,
				"diffusion": {"block_size": 2}
			}`,
			want: "first_byte_mask",
		},
		{
			name: "distillation",
			body: `"training": {
				"steps": 1, "lr": 0.001, "batch_tokens": 8,
				"objective": "block_diffusion", "mlm_mask_token_id": 7,
				"diffusion": {"block_size": 2},
				"distillation": {
					"teacher_checkpoints": ["teacher.safetensors"],
					"teacher_configs": ["teacher.json"],
					"loss_weight_ce": 1,
					"loss_weight_kl": 0
				}
			}`,
			want: "training.distillation",
		},
		{
			name: "segment attention mask",
			body: `"training": {
				"steps": 1, "lr": 0.001, "batch_tokens": 8,
				"objective": "block_diffusion", "mlm_mask_token_id": 7,
				"diffusion": {"block_size": 2},
				"attention_segment_mask": "boundary_token",
				"attention_segment_boundary_token_id": 1
			}`,
			want: "attention_segment_mask",
		},
		{
			name: "window size",
			body: `"blocks": [{"type": "plain", "heads": 2, "window_size": 2}],
				"training": {
					"steps": 1, "lr": 0.001, "batch_tokens": 8,
					"objective": "block_diffusion", "mlm_mask_token_id": 7,
					"diffusion": {"block_size": 2}
				}`,
			want: "window_size",
		},
		{
			name: "unsupported block",
			body: `"blocks": [{"type": "mamba"}],
				"training": {
					"steps": 1, "lr": 0.001, "batch_tokens": 8,
					"objective": "block_diffusion", "mlm_mask_token_id": 7,
					"diffusion": {"block_size": 2}
				}`,
			want: "cannot be combined",
		},
		{
			name: "no plain block",
			body: `"blocks": [{"type": "swiglu"}],
				"training": {
					"steps": 1, "lr": 0.001, "batch_tokens": 8,
					"objective": "block_diffusion", "mlm_mask_token_id": 7,
					"diffusion": {"block_size": 2}
				}`,
			want: "requires at least one type=plain",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ParseArchConfig([]byte(objectiveConfigJSON(tt.body)), tt.name)
			if err == nil {
				t.Fatal("ParseArchConfig succeeded, want error")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error = %q, want substring %q", err, tt.want)
			}
		})
	}
}

func TestDiffusionSpecValidation(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string
	}{
		{
			name: "mask fraction range",
			body: `"training": {
				"steps": 1, "lr": 0.001, "batch_tokens": 8,
				"objective": "block_diffusion", "mlm_mask_token_id": 7,
				"diffusion": {"block_size": 2, "min_mask_fraction": 0.75, "max_mask_fraction": 0.5}
			}`,
			want: "min_mask_fraction",
		},
		{
			name: "zero max mask fraction",
			body: `"training": {
				"steps": 1, "lr": 0.001, "batch_tokens": 8,
				"objective": "block_diffusion", "mlm_mask_token_id": 7,
				"diffusion": {"block_size": 2, "max_mask_fraction": 0}
			}`,
			want: "max_mask_fraction",
		},
		{
			name: "bad confidence threshold",
			body: `"training": {
				"steps": 1, "lr": 0.001, "batch_tokens": 8,
				"objective": "block_diffusion", "mlm_mask_token_id": 7,
				"diffusion": {"block_size": 2, "confidence_threshold": 1.5}
			}`,
			want: "confidence_threshold",
		},
		{
			name: "bad commit floor",
			body: `"training": {
				"steps": 1, "lr": 0.001, "batch_tokens": 8,
				"objective": "block_diffusion", "mlm_mask_token_id": 7,
				"diffusion": {"block_size": 2, "commit_floor": 3}
			}`,
			want: "commit_floor",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ParseArchConfig([]byte(objectiveConfigJSON(tt.body)), tt.name)
			if err == nil {
				t.Fatal("ParseArchConfig succeeded, want error")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error = %q, want substring %q", err, tt.want)
			}
		})
	}
}
