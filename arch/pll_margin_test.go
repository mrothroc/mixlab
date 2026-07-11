package arch

import (
	"reflect"
	"strings"
	"testing"
)

func TestPLLMarginDefaultsAndZeroWeightParity(t *testing.T) {
	base := parseObjectiveConfig(t, `"training": {
		"steps": 1, "lr": 0.001, "batch_tokens": 8,
		"objective": "mlm", "mlm_mask_token_id": 7
	}`)
	zero := parseObjectiveConfig(t, `"training": {
		"steps": 1, "lr": 0.001, "batch_tokens": 8,
		"objective": "mlm", "mlm_mask_token_id": 7,
		"pll_margin": {"weight": 0}
	}`)
	if zero.Training.PLLMarginActive() {
		t.Fatal("weight:0 must disable PLL margin")
	}
	got := zero.Training.PLLMargin
	if got.Source != PLLMarginSourceFile || got.Margin != 1 || got.AnchorWeight != 0.5 || got.BatchFraction != 0.25 || got.Target != PLLMarginTargetAnnotatedSpan {
		t.Fatalf("unexpected PLL margin defaults: %+v", got)
	}
	enabled := parseObjectiveConfig(t, `"training": {
		"steps": 1, "lr": 0.001, "batch_tokens": 8,
		"objective": "mlm", "mlm_mask_token_id": 7,
		"pll_margin": {"path": "pairs.bin"}
	}`)
	if got := enabled.Training.PLLMargin.Weight; got != 0.1 {
		t.Fatalf("default PLL margin weight=%g, want conservative default 0.1", got)
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
		t.Fatal("weight:0 PLL margin graph differs from baseline")
	}
}

func TestPLLMarginIRAndValidation(t *testing.T) {
	cfg := parseObjectiveConfig(t, `"training": {
		"steps": 1, "lr": 0.001, "batch_tokens": 8,
		"objective": "mlm", "mlm_mask_token_id": 7,
		"pll_margin": {"path": "pairs.bin", "margin": 1, "weight": 0.5, "anchor_weight": 0.25, "batch_fraction": 0.5, "target": "distractor_span"}
	}`)
	if cfg.Training.PLLMargin.Target != PLLMarginTargetAnnotatedSpan {
		t.Fatalf("target alias resolved to %q", cfg.Training.PLLMargin.Target)
	}
	prog, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{Objective: ObjectiveMLM})
	if err != nil {
		t.Fatalf("build PLL margin program: %v", err)
	}
	if !programDeclaresInputArch(prog, "pll_margin_loss_mask") || !programDeclaresOutputArch(prog, "pll_margin_loss") {
		t.Fatal("active PLL margin program missing input or output")
	}
	if got := countOps(prog, OpMaskedMarginPLL); got != 1 {
		t.Fatalf("masked margin PLL ops=%d, want 1", got)
	}

	for _, tc := range []struct {
		name string
		body string
		want string
	}{
		{"causal", `"training": {"batch_tokens": 8, "objective": "causal", "mlm_mask_token_id": 7, "pll_margin": {"path": "pairs.bin"}}`, "only supports"},
		{"missing path", `"training": {"batch_tokens": 8, "objective": "mlm", "mlm_mask_token_id": 7, "pll_margin": {}}`, "path is required"},
		{"bad margin", `"training": {"batch_tokens": 8, "objective": "mlm", "mlm_mask_token_id": 7, "pll_margin": {"path": "pairs.bin", "margin": -1}}`, "margin"},
		{"bad anchor", `"training": {"batch_tokens": 8, "objective": "mlm", "mlm_mask_token_id": 7, "pll_margin": {"path": "pairs.bin", "anchor_weight": -1}}`, "anchor_weight"},
		{"odd rows", `"training": {"batch_tokens": 12, "objective": "mlm", "mlm_mask_token_id": 7, "pll_margin": {"path": "pairs.bin"}}`, "even number"},
		{"invariance conflict", `"training": {"batch_tokens": 8, "objective": "mlm", "mlm_mask_token_id": 7, "invariance": {"path": "i.bin"}, "pll_margin": {"path": "pairs.bin"}}`, "cannot be combined"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ParseArchConfig([]byte(objectiveConfigJSON(tc.body)), tc.name)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("ParseArchConfig error=%v, want containing %q", err, tc.want)
			}
		})
	}
}

func TestMultiheadPLLMarginUsesExportHead(t *testing.T) {
	cfg := parseMultiheadConfig(t, `"training": {
		"steps": 1, "lr": 0.001, "batch_tokens": 8,
		"objective": "multihead", "mlm_mask_token_id": 7,
		"export_head": "scorer",
		"pll_margin": {"path": "pairs.bin", "weight": 0.1},
		"heads": [
			{"name": "scorer", "objective": "mntp"},
			{"name": "denoiser", "objective": "block_diffusion", "diffusion": {"block_size": 2}}
		]
	}`)
	prog, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{Objective: ObjectiveMultihead})
	if err != nil {
		t.Fatalf("build multihead PLL margin program: %v", err)
	}
	if got := countOps(prog, OpMaskedMarginPLL); got != 1 {
		t.Fatalf("masked margin PLL ops=%d, want 1", got)
	}
	inactive, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{Objective: ObjectiveMultihead, PLLMarginInactive: true})
	if err != nil {
		t.Fatalf("build inactive multihead program: %v", err)
	}
	if programDeclaresInputArch(inactive, "pll_margin_loss_mask") || countOps(inactive, OpMaskedMarginPLL) != 0 {
		t.Fatal("inactive multihead program retained PLL margin graph")
	}
}
