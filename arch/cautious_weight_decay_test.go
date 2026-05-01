package arch

import (
	"encoding/json"
	"strings"
	"testing"
)

func cautiousWeightDecayTestConfig() ArchConfig {
	return ArchConfig{
		Name:      "cautious_weight_decay",
		ModelDim:  64,
		VocabSize: 256,
		SeqLen:    32,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 4}},
		Training: TrainingSpec{
			Steps:                             100,
			LR:                                3e-4,
			CautiousWeightDecay:               true,
			CautiousWeightDecayActivationFrac: 0.35,
		},
	}
}

func TestParseArchConfig_CautiousWeightDecay(t *testing.T) {
	cfg := cautiousWeightDecayTestConfig()
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got, err := ParseArchConfig(data, "cautious_weight_decay")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	if !got.Training.CautiousWeightDecay {
		t.Fatal("cautious_weight_decay=false, want true")
	}
	if got.Training.CautiousWeightDecayActivationFrac != 0.35 {
		t.Fatalf("cautious_weight_decay_activation_frac=%g want 0.35", got.Training.CautiousWeightDecayActivationFrac)
	}
	if step := got.Training.EffectiveCautiousWeightDecayActivationStep(); step != 35 {
		t.Fatalf("activation step=%d want 35", step)
	}
	out, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("round-trip marshal: %v", err)
	}
	if !strings.Contains(string(out), "cautious_weight_decay") ||
		!strings.Contains(string(out), "cautious_weight_decay_activation_frac") {
		t.Fatalf("round-trip JSON missing cautious weight decay fields: %s", out)
	}
}

func TestParseArchConfig_RejectsBadCautiousWeightDecayActivationFrac(t *testing.T) {
	for _, frac := range []float64{-0.1, 1.1} {
		cfg := cautiousWeightDecayTestConfig()
		cfg.Training.CautiousWeightDecayActivationFrac = frac
		data, err := json.Marshal(cfg)
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		_, err = ParseArchConfig(data, "bad_cautious_weight_decay")
		if err == nil {
			t.Fatalf("expected error for frac=%g", frac)
		}
		if !strings.Contains(err.Error(), "training.cautious_weight_decay_activation_frac") {
			t.Fatalf("error %q does not mention cautious_weight_decay_activation_frac", err)
		}
	}
}
