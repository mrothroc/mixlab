package arch

import (
	"strings"
	"testing"
)

func TestTrainingData2VecDefaultsAndLossWeightZero(t *testing.T) {
	cfg := parseObjectiveConfig(t, `"blocks": [
		{"type": "swiglu"}, {"type": "swiglu"}, {"type": "swiglu"}, {"type": "swiglu"},
		{"type": "swiglu"}, {"type": "swiglu"}, {"type": "swiglu"}, {"type": "swiglu"}
	], "training": {
		"steps": 1, "lr": 0.001, "batch_tokens": 8,
		"objective": "mntp", "mlm_mask_token_id": 7,
		"data2vec": {}
	}`)
	d := cfg.Training.Data2Vec
	if d == nil {
		t.Fatal("data2vec defaults missing")
	}
	if d.LossWeight != 1 || d.EMATau != 0.999 || d.EMATauStart != 0.999 || d.EMATauEnd != 0.999 {
		t.Fatalf("bad data2vec defaults: %+v", d)
	}
	if d.TopKLayers != 8 || d.SmoothL1Beta != 1 || d.TargetNorm != Data2VecTargetNormLayer || d.TargetNormEps != 1e-5 {
		t.Fatalf("bad data2vec defaults: %+v", d)
	}
	if d.MaskSource != Data2VecMaskSourceObject {
		t.Fatalf("mask_source=%q", d.MaskSource)
	}

	disabled := parseObjectiveConfig(t, `"training": {
		"steps": 1, "lr": 0.001, "batch_tokens": 8,
		"objective": "causal",
		"data2vec": {"loss_weight": 0}
	}`)
	if disabled.Training.Data2VecActive() {
		t.Fatal("loss_weight=0 should disable data2vec")
	}
	prog, err := BuildTrainingIRProgramFromConfig(disabled, TrainingProgramState{})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig(disabled): %v", err)
	}
	if programDeclaresInputArch(prog, "data2vec_targets") {
		t.Fatal("disabled data2vec graph declares data2vec inputs")
	}
}

func TestTrainingData2VecValidationRejectsInvalidCombinations(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string
	}{
		{
			name: "causal objective",
			body: `"training": {"steps": 1, "lr": 0.001, "batch_tokens": 8,
				"objective": "causal", "data2vec": {"top_k_layers": 1}}`,
			want: "not a masked objective",
		},
		{
			name: "top k too large",
			body: `"training": {"steps": 1, "lr": 0.001, "batch_tokens": 8,
				"objective": "mntp", "mlm_mask_token_id": 7,
				"data2vec": {"top_k_layers": 2}}`,
			want: "top_k_layers",
		},
		{
			name: "distillation",
			body: `"training": {"steps": 1, "lr": 0.001, "batch_tokens": 8,
				"objective": "mntp", "mlm_mask_token_id": 7,
				"data2vec": {"top_k_layers": 1},
				"distillation": {"teacher_checkpoints": ["a"], "teacher_configs": ["b"], "loss_weight_ce": 0.5, "loss_weight_kl": 0.5}}`,
			want: "training.distillation",
		},
		{
			name: "target norm",
			body: `"training": {"steps": 1, "lr": 0.001, "batch_tokens": 8,
				"objective": "mntp", "mlm_mask_token_id": 7,
				"data2vec": {"top_k_layers": 1, "target_norm": "bad"}}`,
			want: "target_norm",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ParseArchConfig([]byte(objectiveConfigJSON(tt.body)), tt.name)
			if err == nil {
				t.Fatal("ParseArchConfig succeeded, want error")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error %q does not contain %q", err, tt.want)
			}
		})
	}
}

func TestBuildTrainingIRProgram_Data2VecGraphAndTeacherCapture(t *testing.T) {
	cfg := parseObjectiveConfig(t, `"blocks": [
		{"type": "plain", "heads": 2},
		{"type": "swiglu"},
		{"type": "swiglu"}
	], "training": {
		"steps": 1, "lr": 0.001, "batch_tokens": 8,
		"objective": "mntp", "mlm_mask_token_id": 7,
		"data2vec": {"top_k_layers": 2, "loss_weight": 0.25, "predictor_hidden_dim": 12}
	}`)
	metas, err := CollectWeightShapesFromConfig(cfg)
	if err != nil {
		t.Fatalf("CollectWeightShapesFromConfig: %v", err)
	}
	if metas[len(metas)-2].Name != "data2vec_pred_1" || metas[len(metas)-1].Name != "data2vec_pred_2" {
		t.Fatalf("data2vec predictor weights not appended: %s, %s", metas[len(metas)-2].Name, metas[len(metas)-1].Name)
	}
	prog, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{Objective: ObjectiveMNTP})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig: %v", err)
	}
	if !programDeclaresInputArch(prog, "data2vec_targets") || !programDeclaresInputArch(prog, "data2vec_loss_mask") {
		t.Fatal("data2vec training inputs missing")
	}
	if !programDeclaresOutputArch(prog, "data2vec_loss") {
		t.Fatal("data2vec_loss output missing")
	}
	if countOps(prog, OpMaskedSmoothL1) != 1 {
		t.Fatalf("MaskedSmoothL1 count = %d, want 1", countOps(prog, OpMaskedSmoothL1))
	}

	teacherProg, err := BuildData2VecTeacherIRProgramFromConfig(cfg, ObjectiveMNTP)
	if err != nil {
		t.Fatalf("BuildData2VecTeacherIRProgramFromConfig: %v", err)
	}
	if programDeclaresInputArch(teacherProg, "data2vec_targets") {
		t.Fatal("teacher program unexpectedly declares data2vec training inputs")
	}
	if !programDeclaresOutputArch(teacherProg, "data2vec_layer_01_hidden_flat") ||
		!programDeclaresOutputArch(teacherProg, "data2vec_layer_02_hidden_flat") {
		t.Fatalf("teacher hidden outputs missing: %+v", teacherProg.Outputs)
	}
}
