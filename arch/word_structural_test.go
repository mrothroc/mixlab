package arch

import (
	"strings"
	"testing"
)

func TestWordStructuralConfigValidation(t *testing.T) {
	cfg := parseObjectiveConfig(t, `"training": {
		"steps": 1,
		"lr": 0.001,
		"batch_tokens": 8,
		"objective": "mlm",
		"mlm_mask_token_id": 7,
		"word_structural_objective": {}
	}`)
	if !cfg.Training.WordStructuralActive() {
		t.Fatal("word_structural_objective object should default to enabled")
	}
	if cfg.Training.WordStructuralObjective.Fraction != 0.05 {
		t.Fatalf("fraction=%g, want 0.05", cfg.Training.WordStructuralObjective.Fraction)
	}
	if cfg.Training.WordStructuralObjective.Span != 3 {
		t.Fatalf("span=%d, want 3", cfg.Training.WordStructuralObjective.Span)
	}
	if cfg.Training.WordStructuralObjective.LossWeight != 1 {
		t.Fatalf("loss_weight=%g, want 1", cfg.Training.WordStructuralObjective.LossWeight)
	}
	if got := cfg.Training.WordStructuralObjective.SkipTokenIDs; len(got) != 1 || got[0] != 7 {
		t.Fatalf("skip_token_ids=%v, want [7]", got)
	}

	disabled := parseObjectiveConfig(t, `"training": {
		"steps": 1,
		"lr": 0.001,
		"batch_tokens": 8,
		"objective": "causal",
		"word_structural_objective": {"enabled": false}
	}`)
	if disabled.Training.WordStructuralActive() {
		t.Fatal("enabled:false should disable word_structural_objective")
	}
}

func TestWordStructuralConfigValidationErrors(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string
	}{
		{name: "causal", body: `"training": {"batch_tokens": 8, "objective": "causal", "word_structural_objective": {}}`, want: "requires training.objective"},
		{name: "bad fraction", body: `"training": {"batch_tokens": 8, "objective": "mlm", "mlm_mask_token_id": 7, "word_structural_objective": {"fraction": 0}}`, want: "fraction"},
		{name: "bad span", body: `"training": {"batch_tokens": 8, "objective": "mlm", "mlm_mask_token_id": 7, "word_structural_objective": {"span": 1}}`, want: "span"},
		{name: "bad weight", body: `"training": {"batch_tokens": 8, "objective": "mlm", "mlm_mask_token_id": 7, "word_structural_objective": {"loss_weight": -1}}`, want: "loss_weight"},
		{name: "bad skip", body: `"training": {"batch_tokens": 8, "objective": "mlm", "mlm_mask_token_id": 7, "word_structural_objective": {"skip_token_ids": [33]}}`, want: "skip_token_ids"},
		{name: "hybrid no masked steps", body: `"training": {"batch_tokens": 8, "objective": "hybrid", "mlm_mask_token_id": 7, "hybrid_clm_fraction": 1, "word_structural_objective": {}}`, want: "masked secondary steps"},
		{name: "hybrid diffusion", body: `"training": {"batch_tokens": 8, "objective": "hybrid", "hybrid_secondary_objective": "block_diffusion", "mlm_mask_token_id": 7, "word_structural_objective": {}}`, want: "hybrid_secondary_objective"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ParseArchConfig([]byte(objectiveConfigJSON(tc.body)), tc.name)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("ParseArchConfig error=%v, want containing %q", err, tc.want)
			}
		})
	}
}

func TestWordStructuralIRAddsSeparateMaskedLoss(t *testing.T) {
	cfg := parseObjectiveConfig(t, `"training": {
		"steps": 1,
		"lr": 0.001,
		"batch_tokens": 8,
		"objective": "mlm",
		"mlm_mask_token_id": 7,
		"word_structural_objective": {"loss_weight": 0}
	}`)
	prog, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{Objective: ObjectiveMLM})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig: %v", err)
	}
	for _, name := range []string{"word_struct_targets", "word_struct_loss_mask"} {
		if !programDeclaresInputArch(prog, name) {
			t.Fatalf("program missing input %q", name)
		}
	}
	if !programDeclaresOutputArch(prog, "word_struct_loss") {
		t.Fatal("program missing word_struct_loss output")
	}
	if n := countOps(prog, OpMaskedCrossEntropy); n != 2 {
		t.Fatalf("OpMaskedCrossEntropy count=%d, want primary + word-structural", n)
	}

	disabled := parseObjectiveConfig(t, `"training": {
		"steps": 1,
		"lr": 0.001,
		"batch_tokens": 8,
		"objective": "mlm",
		"mlm_mask_token_id": 7,
		"word_structural_objective": {"enabled": false}
	}`)
	disabledProg, err := BuildTrainingIRProgramFromConfig(disabled, TrainingProgramState{Objective: ObjectiveMLM})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig disabled: %v", err)
	}
	if programDeclaresInputArch(disabledProg, "word_struct_targets") {
		t.Fatal("disabled program declared word_struct_targets")
	}
	if n := countOps(disabledProg, OpMaskedCrossEntropy); n != 1 {
		t.Fatalf("disabled OpMaskedCrossEntropy count=%d, want primary only", n)
	}
}

func TestWordStructuralAddsNoWeights(t *testing.T) {
	base := parseObjectiveConfig(t, `"training": {
		"steps": 1,
		"lr": 0.001,
		"batch_tokens": 8,
		"objective": "mlm",
		"mlm_mask_token_id": 7
	}`)
	active := parseObjectiveConfig(t, `"training": {
		"steps": 1,
		"lr": 0.001,
		"batch_tokens": 8,
		"objective": "mlm",
		"mlm_mask_token_id": 7,
		"word_structural_objective": {}
	}`)
	baseWeights, err := CountIRWeightsFromConfig(base)
	if err != nil {
		t.Fatalf("CountIRWeightsFromConfig base: %v", err)
	}
	activeWeights, err := CountIRWeightsFromConfig(active)
	if err != nil {
		t.Fatalf("CountIRWeightsFromConfig active: %v", err)
	}
	if activeWeights != baseWeights {
		t.Fatalf("active weights=%d, want base count %d", activeWeights, baseWeights)
	}
}

func TestWordStructuralMultiheadSelectionValidationAndIR(t *testing.T) {
	cfg, err := ParseArchConfig([]byte(multiheadConfigJSON(`"training": {
		"steps": 1,
		"lr": 0.001,
		"batch_tokens": 8,
		"objective": "multihead",
		"mlm_mask_token_id": 7,
		"export_head": "scorer",
		"word_structural_objective": {"heads": ["scorer"]},
		"heads": [
			{"name": "scorer", "objective": "mntp", "output_head": "bert_mlm"},
			{"name": "aux", "objective": "mlm", "output_head": "bert_mlm"}
		]
	}`)), "word_struct_multihead")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	prog, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig: %v", err)
	}
	if !programDeclaresInputArch(prog, "word_struct_targets") {
		t.Fatal("multihead program missing word_struct_targets")
	}
	if !programDeclaresOutputArch(prog, "head_scorer_word_struct_loss") {
		t.Fatal("multihead program missing selected head word-structural output")
	}
	if programDeclaresOutputArch(prog, "head_aux_word_struct_loss") {
		t.Fatal("multihead program emitted unselected head word-structural output")
	}

	_, err = ParseArchConfig([]byte(multiheadConfigJSON(`"training": {
		"batch_tokens": 8,
		"objective": "multihead",
		"mlm_mask_token_id": 7,
		"word_structural_objective": {"heads": ["detector"]},
		"heads": [
			{"name": "scorer", "objective": "mntp", "output_head": "bert_mlm"},
			{"name": "detector", "objective": "rtd", "output_head": "binary"}
		],
		"rtd": {"generator": "tied", "generator_head": "scorer"}
	}`)), "word_struct_bad_multihead")
	if err == nil || !strings.Contains(err.Error(), "must select an mlm or mntp head") {
		t.Fatalf("ParseArchConfig error=%v, want selected head validation", err)
	}
}
