package arch

import (
	"fmt"
	"reflect"
	"strings"
	"testing"
)

func TestParseArchConfig_MTP(t *testing.T) {
	cfg := `{
		"name": "test_mtp",
		"model_dim": 128,
		"vocab_size": 1024,
		"seq_len": 16,
		"tie_embeddings": true,
		"mtp": {
			"n": 4,
			"loss_weights": [1.0, 0.5, 0.25, 0.125],
			"untie_embed_at_frac": 0.667
		},
		"blocks": [{"type": "plain", "heads": 4}],
		"training": {"steps": 100, "lr": 0.001}
	}`
	got, err := ParseArchConfig([]byte(cfg), "test_mtp")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	if got.MTP == nil {
		t.Fatal("MTP = nil")
	}
	if got.MTP.EffectiveN() != 4 {
		t.Fatalf("mtp.n = %d, want 4", got.MTP.EffectiveN())
	}
	if weights := got.MTP.EffectiveLossWeights(); len(weights) != 4 || weights[1] != 0.5 {
		t.Fatalf("loss weights = %v", weights)
	}
	if got.MTP.EffectiveUntieEmbedAtFrac() != 0.667 {
		t.Fatalf("untie_embed_at_frac = %g, want 0.667", got.MTP.EffectiveUntieEmbedAtFrac())
	}
	if !got.MTPUntieEnabled() {
		t.Fatal("MTPUntieEnabled = false, want true")
	}
}

func TestParseArchConfig_MTPValidation(t *testing.T) {
	base := `{
		"name": "bad_mtp",
		"model_dim": 64,
		"vocab_size": 128,
		"seq_len": 8,
		"tie_embeddings": true,
		"blocks": [{"type": "plain", "heads": 4}],
		"training": {"steps": 100, "lr": 0.001},
		"mtp": %s
	}`
	tests := []struct {
		name string
		mtp  string
		want string
	}{
		{name: "zero_n", mtp: `{"n": 0}`, want: "mtp.n"},
		{name: "too_many_heads", mtp: `{"n": 9}`, want: "seq_len"},
		{name: "bad_weights_len", mtp: `{"n": 3, "loss_weights": [1, 0.5]}`, want: "loss_weights"},
		{name: "negative_weight", mtp: `{"n": 2, "loss_weights": [1, -0.5]}`, want: "loss_weights[1]"},
		{name: "bad_untie_frac", mtp: `{"untie_embed_at_frac": 1.25}`, want: "untie_embed_at_frac"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ParseArchConfig([]byte(fmt.Sprintf(base, tt.mtp)), tt.name)
			if err == nil {
				t.Fatal("ParseArchConfig succeeded, want error")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error %q does not contain %q", err, tt.want)
			}
		})
	}
}

func testMTPConfig(n int) *ArchConfig {
	return &ArchConfig{
		Name:          "mtp_test",
		ModelDim:      32,
		VocabSize:     64,
		SeqLen:        8,
		MLPMult:       2,
		TieEmbeddings: false,
		Blocks:        []BlockSpec{{Type: "plain", Heads: 4}},
		MTP:           &MTPSpec{N: n},
		Training:      TrainingSpec{Steps: 10, LR: 0.001, BatchTokens: 16},
	}
}

func TestBuildIRProgram_MTPN1MatchesNoMTP(t *testing.T) {
	base := testMTPConfig(1)
	base.MTP = nil
	withMTP := testMTPConfig(1)

	baseProg, err := BuildIRProgramFromConfig(base)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig(base): %v", err)
	}
	mtpProg, err := BuildIRProgramFromConfig(withMTP)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig(mtp n=1): %v", err)
	}
	if baseProg.NumWeights != mtpProg.NumWeights {
		t.Fatalf("NumWeights mismatch: base=%d mtp=%d", baseProg.NumWeights, mtpProg.NumWeights)
	}
	if !reflect.DeepEqual(baseProg.Ops, mtpProg.Ops) {
		t.Fatal("mtp.n=1 changed emitted IR ops")
	}
}

func TestBuildIRProgram_MTPSharedHeadShiftedLosses(t *testing.T) {
	cfg := testMTPConfig(4)
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}

	var ceCount int
	var evalLossOutput bool
	var horizon3Logits, horizon3Targets bool
	for _, decl := range prog.Outputs {
		if decl.Name == "eval_loss" {
			evalLossOutput = true
		}
	}
	for _, op := range prog.Ops {
		if op.Code == OpCrossEntropy {
			ceCount++
		}
		if op.Code == OpSlice && len(op.Outputs) == 1 && op.Outputs[0] == "mtp_logits_3_slice" {
			horizon3Logits = reflect.DeepEqual(op.IntParams, []int{0, 5, 1, 1})
		}
		if op.Code == OpSlice && len(op.Outputs) == 1 && op.Outputs[0] == "mtp_targets_3_slice" {
			horizon3Targets = reflect.DeepEqual(op.IntParams, []int{3, 8, 1, 1})
		}
	}
	if ceCount != 4 {
		t.Fatalf("cross-entropy op count = %d, want 4", ceCount)
	}
	if !evalLossOutput {
		t.Fatal("missing eval_loss output for MTP program")
	}
	if !horizon3Logits {
		t.Fatal("missing horizon-3 logits slice over valid prefix")
	}
	if !horizon3Targets {
		t.Fatal("missing horizon-3 target slice over shifted suffix")
	}
}

func TestBuildEvalIRProgram_MTPExcludesAuxiliaryLosses(t *testing.T) {
	cfg := testMTPConfig(4)
	prog, err := BuildEvalIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildEvalIRProgramFromConfig: %v", err)
	}
	var ceCount int
	for _, op := range prog.Ops {
		if op.Code == OpCrossEntropy {
			ceCount++
		}
		if len(op.Outputs) == 1 && op.Outputs[0] == "eval_loss" {
			t.Fatal("eval program unexpectedly emits eval_loss")
		}
	}
	if ceCount != 1 {
		t.Fatalf("eval cross-entropy op count = %d, want 1", ceCount)
	}
}

func TestBuildIRProgram_MTPUntieReservesHeadWeight(t *testing.T) {
	cfg := testMTPConfig(2)
	cfg.TieEmbeddings = true
	cfg.MTP.UntieEmbedAtFrac = 0.5

	tied, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{RecurrenceActive: true, HeadUntied: false})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig(tied): %v", err)
	}
	untied, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{RecurrenceActive: true, HeadUntied: true})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig(untied): %v", err)
	}
	if tied.NumWeights != untied.NumWeights {
		t.Fatalf("weight counts differ across untie schedule: tied=%d untied=%d", tied.NumWeights, untied.NumWeights)
	}
	metas, err := CollectWeightShapesFromConfig(cfg)
	if err != nil {
		t.Fatalf("CollectWeightShapesFromConfig: %v", err)
	}
	if len(metas) != tied.NumWeights {
		t.Fatalf("weight metadata count = %d, program weights = %d", len(metas), tied.NumWeights)
	}
	if len(metas) < 2 || metas[1].Name != "head" {
		limit := len(metas)
		if limit > 3 {
			limit = 3
		}
		t.Fatalf("reserved head metadata missing at fixed slot: %+v", metas[:limit])
	}

	var tiedUsesEmbed, untiedUsesHead bool
	for _, op := range tied.Ops {
		if op.Code == OpMatMul && len(op.Inputs) == 2 && op.Inputs[1] == "tied_head" && len(op.Outputs) == 1 && op.Outputs[0] == "logits" {
			tiedUsesEmbed = true
		}
	}
	for _, op := range untied.Ops {
		if op.Code == OpMatMul && len(op.Inputs) == 2 && op.Inputs[1] == "w1" && len(op.Outputs) == 1 && op.Outputs[0] == "logits" {
			untiedUsesHead = true
		}
	}
	if !tiedUsesEmbed {
		t.Fatal("pre-untie program does not use tied embedding head")
	}
	if !untiedUsesHead {
		t.Fatal("post-untie program does not use reserved head weight")
	}
}
