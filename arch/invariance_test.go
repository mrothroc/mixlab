package arch

import (
	"reflect"
	"strings"
	"testing"
)

func TestInvarianceConfigDefaultsAndZeroWeightParity(t *testing.T) {
	base := parseObjectiveConfig(t, `"training": {
		"steps": 1, "lr": 0.001, "batch_tokens": 8,
		"objective": "mlm", "mlm_mask_token_id": 7
	}`)
	zero := parseObjectiveConfig(t, `"training": {
		"steps": 1, "lr": 0.001, "batch_tokens": 8,
		"objective": "mlm", "mlm_mask_token_id": 7,
		"invariance": {"weight": 0}
	}`)
	if zero.Training.InvarianceActive() {
		t.Fatal("weight:0 must disable invariance")
	}
	if zero.Training.Invariance.Source != InvarianceSourceFile || zero.Training.Invariance.Loss != InvarianceLossSymKL || zero.Training.Invariance.Target != InvarianceTargetMasked || zero.Training.Invariance.BatchFraction != 0.25 {
		t.Fatalf("unexpected invariance defaults: %+v", zero.Training.Invariance)
	}
	baseProg, err := BuildTrainingIRProgramFromConfig(base, TrainingProgramState{Objective: ObjectiveMLM})
	if err != nil {
		t.Fatalf("build base: %v", err)
	}
	zeroProg, err := BuildTrainingIRProgramFromConfig(zero, TrainingProgramState{Objective: ObjectiveMLM})
	if err != nil {
		t.Fatalf("build zero-weight: %v", err)
	}
	if !reflect.DeepEqual(baseProg, zeroProg) {
		t.Fatal("weight:0 invariance graph differs from baseline")
	}
}

func TestInvarianceIRAndValidation(t *testing.T) {
	cfg := parseObjectiveConfig(t, `"training": {
		"steps": 1, "lr": 0.001, "batch_tokens": 8,
		"objective": "mlm", "mlm_mask_token_id": 7,
		"invariance": {"path": "pairs.bin", "weight": 0.1, "batch_fraction": 0.5}
	}`)
	prog, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{Objective: ObjectiveMLM})
	if err != nil {
		t.Fatalf("build invariance program: %v", err)
	}
	if !programDeclaresInputArch(prog, "invariance_loss_mask") || !programDeclaresOutputArch(prog, "invariance_loss") {
		t.Fatal("active invariance program missing input or output")
	}
	if got := countOps(prog, OpMaskedSymmetricKL); got != 1 {
		t.Fatalf("masked symmetric KL ops=%d, want 1", got)
	}

	for _, tc := range []struct {
		name string
		body string
		want string
	}{
		{"causal", `"training": {"batch_tokens": 8, "objective": "causal", "mlm_mask_token_id": 7, "invariance": {"path": "pairs.bin"}}`, "only supports"},
		{"missing path", `"training": {"batch_tokens": 8, "objective": "mlm", "mlm_mask_token_id": 7, "invariance": {}}`, "path is required"},
		{"bad loss", `"training": {"batch_tokens": 8, "objective": "mlm", "mlm_mask_token_id": 7, "invariance": {"path": "pairs.bin", "loss": "kl"}}`, "loss"},
		{"bad skip", `"training": {"batch_tokens": 8, "objective": "mlm", "mlm_mask_token_id": 7, "invariance": {"path": "pairs.bin", "skip_token_ids": [32]}}`, "skip_token_ids"},
		{"odd rows", `"training": {"batch_tokens": 12, "objective": "mlm", "mlm_mask_token_id": 7, "invariance": {"path": "pairs.bin"}}`, "even number"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ParseArchConfig([]byte(objectiveConfigJSON(tc.body)), tc.name)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("ParseArchConfig error=%v, want containing %q", err, tc.want)
			}
		})
	}
}

func TestMultiheadInvarianceUsesExportHead(t *testing.T) {
	cfg := parseMultiheadConfig(t, `"training": {
		"steps": 1, "lr": 0.001, "batch_tokens": 8,
		"objective": "multihead", "mlm_mask_token_id": 7,
		"export_head": "scorer",
		"invariance": {"path": "pairs.bin", "weight": 0.1},
		"heads": [
			{"name": "scorer", "objective": "mlm"},
			{"name": "denoiser", "objective": "block_diffusion", "diffusion": {"block_size": 2}}
		]
	}`)
	prog, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{Objective: ObjectiveMultihead})
	if err != nil {
		t.Fatalf("build multihead invariance program: %v", err)
	}
	if got := countOps(prog, OpMaskedSymmetricKL); got != 1 {
		t.Fatalf("masked symmetric KL ops=%d, want 1", got)
	}
	if !programDeclaresOutputArch(prog, "head_scorer_invariance_loss") {
		t.Fatal("multihead program missing scorer invariance loss output")
	}
	inactive, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{Objective: ObjectiveMultihead, InvarianceInactive: true})
	if err != nil {
		t.Fatalf("build inactive multihead program: %v", err)
	}
	if programDeclaresInputArch(inactive, "invariance_loss_mask") || countOps(inactive, OpMaskedSymmetricKL) != 0 {
		t.Fatal("inactive multihead program retained invariance graph")
	}
}
