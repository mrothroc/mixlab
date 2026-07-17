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
	if got := cfg.Training.EffectiveMLMMaskUnit(); got != MLMMaskUnitToken {
		t.Fatalf("mlm_mask_unit = %q, want %q", got, MLMMaskUnitToken)
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

func TestTrainingAttentionSegmentMaskValidation(t *testing.T) {
	cfg := parseObjectiveConfig(t, `"training": {"steps": 1, "lr": 0.001, "batch_tokens": 8}`)
	if cfg.Training.AttentionSegmentMaskEnabled() {
		t.Fatal("segment mask should be disabled by default")
	}
	_ = parseObjectiveConfig(t, `"training": {
		"steps": 1, "lr": 0.001, "batch_tokens": 8,
		"attention_segment_mask": "boundary_token",
		"attention_segment_boundary_token_id": 1
	}`)
	_ = parseObjectiveConfig(t, `"training": {
		"steps": 1, "lr": 0.001, "batch_tokens": 8,
		"attention_segment_mask": "boundary_token",
		"attention_segment_boundary_token_id": 0
	}`)

	tests := []struct {
		name    string
		body    string
		wantErr string
	}{
		{
			name: "missing boundary token",
			body: `"training": {
				"steps": 1, "lr": 0.001, "batch_tokens": 8,
				"attention_segment_mask": "boundary_token"
			}`,
			wantErr: "attention_segment_boundary_token_id is required",
		},
		{
			name: "bad mode",
			body: `"training": {
				"steps": 1, "lr": 0.001, "batch_tokens": 8,
				"attention_segment_mask": "manifest",
				"attention_segment_boundary_token_id": 1
			}`,
			wantErr: "attention_segment_mask",
		},
		{
			name: "out of range boundary token",
			body: `"training": {
				"steps": 1, "lr": 0.001, "batch_tokens": 8,
				"attention_segment_mask": "boundary_token",
				"attention_segment_boundary_token_id": 32
			}`,
			wantErr: "attention_segment_boundary_token_id",
		},
		{
			name: "no plain blocks",
			body: `"blocks": [{"type": "swiglu"}],
				"training": {
					"steps": 1, "lr": 0.001, "batch_tokens": 8,
					"attention_segment_mask": "boundary_token",
					"attention_segment_boundary_token_id": 1
				}`,
			wantErr: "requires at least one type=plain",
		},
		{
			name: "unsupported token mixer",
			body: `"blocks": [{"type": "plain", "heads": 2}, {"type": "retnet", "heads": 2}],
				"training": {
					"steps": 1, "lr": 0.001, "batch_tokens": 8,
					"attention_segment_mask": "boundary_token",
					"attention_segment_boundary_token_id": 1
				}`,
			wantErr: "cannot be combined",
		},
		{
			name: "distillation unsupported",
			body: `"training": {
				"steps": 1, "lr": 0.001, "batch_tokens": 8,
				"attention_segment_mask": "boundary_token",
				"attention_segment_boundary_token_id": 1,
				"distillation": {
					"teacher_checkpoints": ["teacher.safetensors"],
					"teacher_configs": ["teacher.json"],
					"loss_weight_ce": 0.5,
					"loss_weight_kl": 0.5,
					"ensemble_strategy": "mean_logits"
				}
			}`,
			wantErr: "training.distillation",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ParseArchConfig([]byte(objectiveConfigJSON(tt.body)), tt.name)
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("ParseArchConfig error=%v, want containing %q", err, tt.wantErr)
			}
		})
	}
}

func TestTrainingExampleFramingValidation(t *testing.T) {
	cfg := parseObjectiveConfig(t, `"seq_len": 6,
		"training": {
			"steps": 1, "lr": 0.001, "batch_tokens": 12,
			"example_framing": {"content_len": 4, "bos_id": 0, "eos_id": 2}
		}`)
	if !cfg.Training.ExampleFramingEnabled() {
		t.Fatal("example framing should be enabled")
	}
	if cfg.Training.ExampleFraming.BosID != 0 {
		t.Fatalf("bos_id=%d, want explicit 0", cfg.Training.ExampleFraming.BosID)
	}

	tests := []struct {
		name    string
		body    string
		wantErr string
	}{
		{
			name: "missing content len",
			body: `"training": {"steps": 1, "lr": 0.001, "batch_tokens": 4,
				"example_framing": {"bos_id": 1, "eos_id": 2}}`,
			wantErr: "content_len is required",
		},
		{
			name: "missing bos",
			body: `"seq_len": 6, "training": {"steps": 1, "lr": 0.001, "batch_tokens": 12,
				"example_framing": {"content_len": 4, "eos_id": 2}}`,
			wantErr: "bos_id is required",
		},
		{
			name: "missing eos",
			body: `"seq_len": 6, "training": {"steps": 1, "lr": 0.001, "batch_tokens": 12,
				"example_framing": {"content_len": 4, "bos_id": 1}}`,
			wantErr: "eos_id is required",
		},
		{
			name: "bad seq len",
			body: `"seq_len": 5, "training": {"steps": 1, "lr": 0.001, "batch_tokens": 10,
				"example_framing": {"content_len": 4, "bos_id": 1, "eos_id": 2}}`,
			wantErr: "requires seq_len=6",
		},
		{
			name: "bad batch tokens",
			body: `"seq_len": 6, "training": {"steps": 1, "lr": 0.001, "batch_tokens": 10,
				"example_framing": {"content_len": 4, "bos_id": 1, "eos_id": 2}}`,
			wantErr: "batch_tokens",
		},
		{
			name: "bad bos id",
			body: `"seq_len": 6, "training": {"steps": 1, "lr": 0.001, "batch_tokens": 12,
				"example_framing": {"content_len": 4, "bos_id": 32, "eos_id": 2}}`,
			wantErr: "bos_id=32",
		},
		{
			name: "non causal objective",
			body: `"seq_len": 6, "training": {"steps": 1, "lr": 0.001, "batch_tokens": 12,
				"objective": "mlm", "mlm_mask_token_id": 7,
				"example_framing": {"content_len": 4, "bos_id": 1, "eos_id": 2}}`,
			wantErr: "only supports training.objective=\"causal\"",
		},
		{
			name: "mtp",
			body: `"seq_len": 6, "mtp": {"n": 2}, "training": {"steps": 1, "lr": 0.001, "batch_tokens": 12,
				"example_framing": {"content_len": 4, "bos_id": 1, "eos_id": 2}}`,
			wantErr: "top-level mtp",
		},
		{
			name: "first byte mask",
			body: `"seq_len": 6, "training": {"steps": 1, "lr": 0.001, "batch_tokens": 12,
				"first_byte_mask": true,
				"example_framing": {"content_len": 4, "bos_id": 1, "eos_id": 2}}`,
			wantErr: "first_byte_mask",
		},
		{
			name: "distillation",
			body: `"seq_len": 6, "training": {"steps": 1, "lr": 0.001, "batch_tokens": 12,
				"distillation": {"teacher_checkpoints": ["a.safetensors"], "teacher_configs": ["a.json"]},
				"example_framing": {"content_len": 4, "bos_id": 1, "eos_id": 2}}`,
			wantErr: "distillation",
		},
		{
			name: "attention segment mask",
			body: `"seq_len": 6, "training": {"steps": 1, "lr": 0.001, "batch_tokens": 12,
				"attention_segment_mask": "boundary_token", "attention_segment_boundary_token_id": 1,
				"example_framing": {"content_len": 4, "bos_id": 1, "eos_id": 2}}`,
			wantErr: "attention_segment_mask",
		},
		{
			name: "shuffle conflict",
			body: `"seq_len": 6, "training": {"steps": 1, "lr": 0.001, "batch_tokens": 12,
				"shuffle_chunk_tokens": 8,
				"example_framing": {"content_len": 4, "bos_id": 1, "eos_id": 2}}`,
			wantErr: "shuffle_chunk_tokens",
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

func TestTrainingRecipeKnobValidationAndDefaults(t *testing.T) {
	cfg := parseObjectiveConfig(t, `"training": {
		"steps": 10,
		"lr": 0.001,
		"batch_tokens": 8,
		"objective": "hybrid",
		"mlm_mask_token_id": 7,
		"z_loss": 0.0001,
		"seq_len_schedule": [[0,2],[5,4]],
		"mlm_mask_prob_schedule": [[0,0.3],[5,0.15]],
		"hybrid_clm_fraction_schedule": [[0,0.75],[5,0.25]],
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
	if got := cfg.Training.EffectiveHybridCLMFractionForStep(0); got != 0.75 {
		t.Fatalf("hybrid clm fraction step0=%g, want 0.75", got)
	}
	if got := cfg.Training.EffectiveHybridCLMFractionForStep(5); got != 0.25 {
		t.Fatalf("hybrid clm fraction step5=%g, want 0.25", got)
	}
	if cfg.Training.HybridCLMFractionScheduleMode != "step" {
		t.Fatalf("hybrid_clm_fraction_schedule_mode=%q, want step", cfg.Training.HybridCLMFractionScheduleMode)
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

func TestEffectiveHybridCLMFractionScheduleLinear(t *testing.T) {
	cfg := parseObjectiveConfig(t, `"training": {
		"batch_tokens": 8,
		"objective": "hybrid",
		"mlm_mask_token_id": 7,
		"hybrid_clm_fraction": 0.9,
		"hybrid_clm_fraction_schedule": [[0,1.0],[10,0.25]],
		"hybrid_clm_fraction_schedule_mode": "linear"
	}`)
	if got := cfg.Training.HybridCLMFractionScheduleMode; got != "linear" {
		t.Fatalf("hybrid_clm_fraction_schedule_mode=%q, want linear", got)
	}
	tests := []struct {
		step int
		want float64
	}{
		{step: 0, want: 1.0},
		{step: 5, want: 0.625},
		{step: 10, want: 0.25},
		{step: 20, want: 0.25},
	}
	for _, tt := range tests {
		if got := cfg.Training.EffectiveHybridCLMFractionForStep(tt.step); math.Abs(got-tt.want) > 1e-12 {
			t.Fatalf("EffectiveHybridCLMFractionForStep(%d)=%g, want %g", tt.step, got, tt.want)
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
		{name: "hybrid fraction schedule requires hybrid", body: `"training": {"batch_tokens": 8, "hybrid_clm_fraction_schedule": [[0,0.5]]}`, wantErr: "hybrid_clm_fraction_schedule"},
		{name: "hybrid fraction schedule integer step", body: `"training": {"batch_tokens": 8, "objective":"hybrid", "mlm_mask_token_id":7, "hybrid_clm_fraction_schedule": [[0.5,0.2]]}`, wantErr: "step must be an integer"},
		{name: "hybrid fraction schedule probability", body: `"training": {"batch_tokens": 8, "objective":"hybrid", "mlm_mask_token_id":7, "hybrid_clm_fraction_schedule": [[0,1.5]]}`, wantErr: "must be in [0,1]"},
		{name: "hybrid fraction schedule mode", body: `"training": {"batch_tokens": 8, "objective":"hybrid", "mlm_mask_token_id":7, "hybrid_clm_fraction_schedule": [[0,0.2]], "hybrid_clm_fraction_schedule_mode": "spline"}`, wantErr: "hybrid_clm_fraction_schedule_mode"},
		{name: "early stop metric", body: `"training": {"batch_tokens": 8, "early_stop": {"metric": "train", "patience": 2}}`, wantErr: "early_stop.metric"},
		{name: "early stop patience", body: `"training": {"batch_tokens": 8, "early_stop": {"patience": -1}}`, wantErr: "early_stop.patience"},
		{name: "early stop val gt", body: `"training": {"batch_tokens": 8, "early_stop": {"val_gt": -1}}`, wantErr: "early_stop.val_gt"},
		{name: "seq schedule rejects distillation", body: `"training": {"batch_tokens": 8, "seq_len_schedule": [[0,2]], "distillation": {"teacher_checkpoints":["a"], "teacher_configs":["b"], "loss_weight_ce":0.5, "loss_weight_kl":0.5, "ensemble_strategy":"mean_logits"}}`, wantErr: "seq_len_schedule"},
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

func TestBuildIRProgram_MaskedObjectiveUsesMaskedZLoss(t *testing.T) {
	cfg := parseObjectiveConfig(t, `"training": {
		"steps": 1,
		"lr": 0.001,
		"batch_tokens": 8,
		"objective": "mntp",
		"mlm_mask_token_id": 7,
		"z_loss": 0.0001
	}`)
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}
	if n := countOps(prog, OpMaskedZLoss); n != 1 {
		t.Fatalf("OpMaskedZLoss count=%d, want 1", n)
	}
	if n := countOps(prog, OpZLoss); n != 0 {
		t.Fatalf("OpZLoss count=%d, want 0 for masked objective", n)
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

func TestBuildIRProgram_ExampleFramingCausalUsesMaskedLoss(t *testing.T) {
	cfg := parseObjectiveConfig(t, `"seq_len": 6,
		"training": {
			"steps": 1,
			"lr": 0.001,
			"batch_tokens": 12,
			"example_framing": {"content_len": 4, "bos_id": 1, "eos_id": 2}
		}`)
	prog, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig: %v", err)
	}
	if !programDeclaresInputArch(prog, "loss_mask") {
		t.Fatal("framed causal program missing loss_mask input")
	}
	if n := countOps(prog, OpMaskedCrossEntropy); n != 1 {
		t.Fatalf("OpMaskedCrossEntropy count = %d, want 1", n)
	}
	if n := countOps(prog, OpMaskedCEPerToken); n != 1 {
		t.Fatalf("OpMaskedCEPerToken count = %d, want 1", n)
	}

	defaultCfg := parseObjectiveConfig(t, `"training": {"steps": 1, "lr": 0.001, "batch_tokens": 8}`)
	defaultProg, err := BuildTrainingIRProgramFromConfig(defaultCfg, TrainingProgramState{})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig default: %v", err)
	}
	if programDeclaresInputArch(defaultProg, "loss_mask") {
		t.Fatal("default causal program declares loss_mask")
	}
}

func TestBlockDiffusionIRInputsMaskedLossAndCausalEval(t *testing.T) {
	cfg := parseObjectiveConfig(t, `"training": {
		"steps": 1,
		"lr": 0.001,
		"batch_tokens": 8,
		"objective": "block_diffusion",
		"mlm_mask_token_id": 7,
		"diffusion": {"block_size": 2}
	}`)
	trainProg, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{
		RecurrenceActive: true,
		Objective:        ObjectiveBlockDiffusion,
	})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig(block_diffusion): %v", err)
	}
	for _, name := range []string{"loss_mask", "diffusion_block_start", "diffusion_block_end"} {
		if !programDeclaresInputArch(trainProg, name) {
			t.Fatalf("block_diffusion training program missing input %q", name)
		}
	}
	if n := countOps(trainProg, OpMaskedCrossEntropy); n != 1 {
		t.Fatalf("OpMaskedCrossEntropy count=%d, want 1", n)
	}
	if n := countOps(trainProg, OpMaskedCEPerToken); n != 1 {
		t.Fatalf("OpMaskedCEPerToken count=%d, want 1", n)
	}
	if n := countOps(trainProg, OpBlockDiffusionMask); n != 1 {
		t.Fatalf("OpBlockDiffusionMask count=%d, want 1", n)
	}
	if n := countOps(trainProg, OpCausalMask); n != 0 {
		t.Fatalf("block_diffusion training program emitted %d causal masks, want 0", n)
	}

	evalProg, err := BuildEvalIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildEvalIRProgramFromConfig(block_diffusion): %v", err)
	}
	for _, name := range []string{"loss_mask", "diffusion_block_start", "diffusion_block_end"} {
		if programDeclaresInputArch(evalProg, name) {
			t.Fatalf("block_diffusion eval program declares training-only input %q", name)
		}
	}
	if n := countOps(evalProg, OpBlockDiffusionMask); n != 0 {
		t.Fatalf("eval OpBlockDiffusionMask count=%d, want 0", n)
	}
	if n := countOps(evalProg, OpCausalMask); n != 1 {
		t.Fatalf("eval OpCausalMask count=%d, want 1", n)
	}
	if n := countOps(evalProg, OpCrossEntropy); n != 1 {
		t.Fatalf("eval OpCrossEntropy count=%d, want 1", n)
	}
	if n := countOps(evalProg, OpMaskedCrossEntropy); n != 0 {
		t.Fatalf("eval OpMaskedCrossEntropy count=%d, want 0", n)
	}
}

func TestBlockDiffusionTrainingEvalWeightCountParity(t *testing.T) {
	cfg := parseObjectiveConfig(t, `"blocks": [
			{"type": "plain", "heads": 2},
			{"type": "swiglu"}
		],
		"tie_embeddings": false,
		"training": {
			"steps": 1,
			"lr": 0.001,
			"batch_tokens": 8,
			"objective": "block_diffusion",
			"mlm_mask_token_id": 7,
			"diffusion": {"block_size": 2}
		}`)
	trainProg, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{
		RecurrenceActive: true,
		Objective:        ObjectiveBlockDiffusion,
	})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig(block_diffusion): %v", err)
	}
	evalProg, err := BuildEvalIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildEvalIRProgramFromConfig(block_diffusion): %v", err)
	}
	counted, err := CountIRWeightsFromConfig(cfg)
	if err != nil {
		t.Fatalf("CountIRWeightsFromConfig: %v", err)
	}
	if trainProg.NumWeights != evalProg.NumWeights {
		t.Fatalf("weight count mismatch train=%d eval=%d", trainProg.NumWeights, evalProg.NumWeights)
	}
	if trainProg.NumWeights != counted {
		t.Fatalf("training NumWeights=%d, counted=%d", trainProg.NumWeights, counted)
	}
}

func TestHybridBlockDiffusionConcreteGraphParity(t *testing.T) {
	cfg := parseObjectiveConfig(t, `"blocks": [
			{"type": "plain", "heads": 2},
			{"type": "swiglu"}
		],
		"tie_embeddings": false,
		"training": {
			"steps": 1,
			"lr": 0.001,
			"batch_tokens": 8,
			"objective": "hybrid",
			"hybrid_secondary_objective": "block_diffusion",
			"hybrid_clm_fraction": 0.5,
			"mlm_mask_token_id": 7,
			"diffusion": {"block_size": 2}
		}`)
	causalProg, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{
		RecurrenceActive: true,
		Objective:        ObjectiveCausal,
	})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig(causal): %v", err)
	}
	diffusionProg, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{
		RecurrenceActive: true,
		Objective:        ObjectiveBlockDiffusion,
	})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig(block_diffusion): %v", err)
	}
	evalProg, err := BuildEvalIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildEvalIRProgramFromConfig(hybrid block diffusion): %v", err)
	}
	if causalProg.NumWeights != diffusionProg.NumWeights || causalProg.NumWeights != evalProg.NumWeights {
		t.Fatalf("weight count mismatch causal=%d diffusion=%d eval=%d", causalProg.NumWeights, diffusionProg.NumWeights, evalProg.NumWeights)
	}
	for _, name := range []string{"loss_mask", "diffusion_block_start", "diffusion_block_end"} {
		if !programDeclaresInputArch(diffusionProg, name) {
			t.Fatalf("hybrid diffusion concrete program missing input %q", name)
		}
		if programDeclaresInputArch(causalProg, name) {
			t.Fatalf("hybrid causal concrete program unexpectedly declares input %q", name)
		}
	}
	if n := countOps(diffusionProg, OpBlockDiffusionMask); n != 1 {
		t.Fatalf("hybrid diffusion OpBlockDiffusionMask count=%d, want 1", n)
	}
	if n := countOps(causalProg, OpBlockDiffusionMask); n != 0 {
		t.Fatalf("hybrid causal OpBlockDiffusionMask count=%d, want 0", n)
	}
}

func TestBuildIRProgram_SegmentAttentionMaskCausalTrainingOnly(t *testing.T) {
	cfg := parseObjectiveConfig(t, `"training": {
		"steps": 1,
		"lr": 0.001,
		"batch_tokens": 8,
		"attention_segment_mask": "boundary_token",
		"attention_segment_boundary_token_id": 1
	}`)
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}
	if !programDeclaresInputArch(prog, "segment_ids") {
		t.Fatal("segment-mask training program should declare segment_ids")
	}
	if n := countOps(prog, OpSegmentAttentionMask); n != 1 {
		t.Fatalf("OpSegmentAttentionMask count=%d want 1", n)
	}
	if n := countOps(prog, OpCausalMask); n != 0 {
		t.Fatalf("OpCausalMask count=%d want 0 when segment mask owns causal masking", n)
	}

	evalProg, err := BuildEvalIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildEvalIRProgramFromConfig: %v", err)
	}
	if programDeclaresInputArch(evalProg, "segment_ids") {
		t.Fatal("eval program should not declare segment_ids")
	}
	if n := countOps(evalProg, OpSegmentAttentionMask); n != 0 {
		t.Fatalf("eval OpSegmentAttentionMask count=%d want 0", n)
	}
	if n := countOps(evalProg, OpCausalMask); n != 1 {
		t.Fatalf("eval OpCausalMask count=%d want 1", n)
	}
}

func TestBuildIRProgram_SegmentAttentionMaskMaskedAndHybridExample(t *testing.T) {
	cfg := parseObjectiveConfig(t, `"training": {
		"steps": 1,
		"lr": 0.001,
		"batch_tokens": 8,
		"objective": "mlm",
		"mlm_mask_token_id": 7,
		"attention_segment_mask": "boundary_token",
		"attention_segment_boundary_token_id": 1
	}`)
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig(mlm): %v", err)
	}
	if !programDeclaresInputArch(prog, "segment_ids") {
		t.Fatal("MLM segment-mask program should declare segment_ids")
	}
	if n := countOps(prog, OpSegmentAttentionMask); n != 1 {
		t.Fatalf("MLM OpSegmentAttentionMask count=%d want 1", n)
	}
	if n := countOps(prog, OpCausalMask) + countOps(prog, OpSelectiveCausalMask); n != 0 {
		t.Fatalf("MLM causal mask op count=%d want 0", n)
	}

	hybrid := parseObjectiveConfig(t, `"training": {
		"steps": 1,
		"lr": 0.001,
		"batch_tokens": 8,
		"objective": "hybrid",
		"hybrid_mix_granularity": "example",
		"hybrid_secondary_objective": "mlm",
		"mlm_mask_token_id": 7,
		"attention_segment_mask": "boundary_token",
		"attention_segment_boundary_token_id": 1
	}`)
	hybridProg, err := BuildTrainingIRProgramFromConfig(hybrid, TrainingProgramState{Objective: ObjectiveHybridExample})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig(hybrid_example): %v", err)
	}
	if !programDeclaresInputArch(hybridProg, "attention_causal_mask") || !programDeclaresInputArch(hybridProg, "segment_ids") {
		t.Fatal("hybrid example segment-mask program should declare both attention_causal_mask and segment_ids")
	}
	if n := countOps(hybridProg, OpSegmentAttentionMask); n != 1 {
		t.Fatalf("hybrid OpSegmentAttentionMask count=%d want 1", n)
	}
	if n := countOps(hybridProg, OpSelectiveCausalMask); n != 0 {
		t.Fatalf("hybrid OpSelectiveCausalMask count=%d want 0", n)
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

func TestBuildTrainingIRProgram_HybridExampleUsesSelectiveAttentionMask(t *testing.T) {
	cfg := parseObjectiveConfig(t, `"blocks": [{"type": "plain", "heads": 2, "attention_mask": "bidirectional"}],
		"training": {
			"steps": 1,
			"lr": 0.001,
			"batch_tokens": 8,
			"objective": "hybrid",
			"hybrid_mix_granularity": "example",
			"hybrid_clm_fraction": 0.0625,
			"mlm_mask_token_id": 7,
			"hybrid_secondary_objective": "mlm"
		}`)
	prog, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{
		RecurrenceActive: true,
		Objective:        ObjectiveHybridExample,
	})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig(hybrid_example): %v", err)
	}
	if !programDeclaresInputArch(prog, "loss_mask") {
		t.Fatal("hybrid example program missing loss_mask")
	}
	if !programDeclaresInputArch(prog, "attention_causal_mask") {
		t.Fatal("hybrid example program missing attention_causal_mask")
	}
	if n := countOps(prog, OpSelectiveCausalMask); n != 1 {
		t.Fatalf("OpSelectiveCausalMask count = %d, want 1", n)
	}
	if n := countOps(prog, OpCausalMask); n != 0 {
		t.Fatalf("hybrid example program emitted %d plain causal masks, want 0", n)
	}
	if n := countOps(prog, OpMaskedCrossEntropy); n != 1 {
		t.Fatalf("OpMaskedCrossEntropy count = %d, want 1", n)
	}
}

func TestParseArchConfig_HybridMixGranularityValidation(t *testing.T) {
	for _, tc := range []struct {
		name string
		body string
		want string
	}{
		{
			name: "invalid mode",
			body: `"blocks": [{"type": "plain", "heads": 2}],
				"training": {
					"steps": 1,
					"lr": 0.001,
					"batch_tokens": 8,
					"objective": "hybrid",
					"hybrid_mix_granularity": "token",
					"mlm_mask_token_id": 7
				}`,
			want: "hybrid_mix_granularity",
		},
		{
			name: "example non hybrid",
			body: `"blocks": [{"type": "plain", "heads": 2}],
				"training": {
					"steps": 1,
					"lr": 0.001,
					"batch_tokens": 8,
					"objective": "mlm",
					"hybrid_mix_granularity": "example",
					"mlm_mask_token_id": 7
				}`,
			want: "training.objective",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ParseArchConfig([]byte(objectiveConfigJSON(tc.body)), tc.name)
			if err == nil {
				t.Fatal("ParseArchConfig succeeded, want error")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want substring %q", err, tc.want)
			}
		})
	}

	cfg := parseObjectiveConfig(t, `"blocks": [{"type": "plain", "heads": 2}],
		"training": {
			"steps": 1,
			"lr": 0.001,
			"batch_tokens": 8,
			"objective": "hybrid",
			"hybrid_mix_granularity": "example",
			"mlm_mask_token_id": 7
		}`)
	if got := cfg.Training.HybridMixGranularity; got != HybridMixGranularityExample {
		t.Fatalf("hybrid_mix_granularity = %q, want %q", got, HybridMixGranularityExample)
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
