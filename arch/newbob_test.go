package arch

import (
	"encoding/json"
	"strings"
	"testing"
)

const newBobClassificationConfig = `{
	"name":"newbob_classification",
	"model_dim":8,
	"vocab_size":16,
	"seq_len":4,
	"tie_embeddings":true,
	"blocks":[{"type":"plain","heads":2,"attention_mask":"bidirectional"}],
	"training":{
		"objective":"classification",
		"classification":{"num_labels":2,"pooling":"mean"},
		"steps":20,
		"lr":0.0002,
		"batch_tokens":8,
		"lr_schedule":"newbob",
		"newbob":{"annealing_factor":0.9,"improvement_threshold":0.0025,"patient":0,"metric":"val_error_rate"},
		"val_every_steps":5,
		"val_examples":0
	}
}`

func TestNewBobConfigParsesAndPreservesReferenceControls(t *testing.T) {
	cfg, err := ParseArchConfig([]byte(newBobClassificationConfig), "newbob")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Training.EffectiveLRSchedule() != LRScheduleNewBob || cfg.Training.NewBob == nil {
		t.Fatalf("schedule=%q newbob=%+v", cfg.Training.EffectiveLRSchedule(), cfg.Training.NewBob)
	}
	got := cfg.Training.NewBob
	if got.AnnealingFactor != 0.9 || got.ImprovementThreshold != 0.0025 || got.Patient != 0 || got.Metric != NewBobMetricValErrorRate {
		t.Fatalf("newbob=%+v", got)
	}
	if !cfg.Training.ConfiguredMetricValidation() || cfg.Training.ValEverySteps != 5 || cfg.Training.ValExamples != 0 {
		t.Fatalf("validation cadence/examples=%d/%d configured=%t", cfg.Training.ValEverySteps, cfg.Training.ValExamples, cfg.Training.ConfiguredMetricValidation())
	}
}

func TestNewBobDefaultsAndExplicitZeroThreshold(t *testing.T) {
	raw := strings.Replace(newBobClassificationConfig,
		`"newbob":{"annealing_factor":0.9,"improvement_threshold":0.0025,"patient":0,"metric":"val_error_rate"}`,
		`"newbob":{}`, 1)
	cfg, err := ParseArchConfig([]byte(raw), "newbob-defaults")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Training.NewBob.AnnealingFactor != 0.5 || cfg.Training.NewBob.ImprovementThreshold != 0.0025 || cfg.Training.NewBob.Metric != NewBobMetricValLoss {
		t.Fatalf("defaults=%+v", cfg.Training.NewBob)
	}
	raw = strings.Replace(newBobClassificationConfig, `"improvement_threshold":0.0025`, `"improvement_threshold":0`, 1)
	cfg, err = ParseArchConfig([]byte(raw), "newbob-zero-threshold")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Training.NewBob.ImprovementThreshold != 0 {
		t.Fatalf("explicit zero threshold defaulted to %g", cfg.Training.NewBob.ImprovementThreshold)
	}
	blob, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	roundTripped, err := ParseArchConfig(blob, "newbob-zero-threshold-round-trip")
	if err != nil {
		t.Fatal(err)
	}
	if roundTripped.Training.NewBob.ImprovementThreshold != 0 {
		t.Fatalf("explicit zero threshold changed after round trip to %g", roundTripped.Training.NewBob.ImprovementThreshold)
	}
}

func TestNewBobConfigValidation(t *testing.T) {
	tests := []struct {
		name string
		old  string
		new  string
		want string
	}{
		{name: "missing object", old: `"newbob":{"annealing_factor":0.9,"improvement_threshold":0.0025,"patient":0,"metric":"val_error_rate"},`, want: "requires training.newbob"},
		{name: "missing cadence", old: `"val_every_steps":5,`, want: "requires training.val_every_steps > 0"},
		{name: "invalid factor", old: `"annealing_factor":0.9`, new: `"annealing_factor":1.1`, want: "annealing_factor"},
		{name: "negative threshold", old: `"improvement_threshold":0.0025`, new: `"improvement_threshold":-0.1`, want: "improvement_threshold"},
		{name: "negative patient", old: `"patient":0`, new: `"patient":-1`, want: "patient"},
		{name: "bad metric", old: `"metric":"val_error_rate"`, new: `"metric":"accuracy"`, want: "metric"},
		{name: "warmup conflict", old: `"lr_schedule":"newbob"`, new: `"lr_schedule":"newbob","warmup_steps":10`, want: "cannot be combined"},
		{name: "causal objective", old: `"objective":"classification"`, new: `"objective":"causal"`, want: "classification settings require"},
		{name: "negative examples", old: `"val_examples":0`, new: `"val_examples":-1`, want: "val_examples"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			raw := newBobClassificationConfig
			if tt.new == "" {
				raw = strings.Replace(raw, tt.old, "", 1)
			} else {
				raw = strings.Replace(raw, tt.old, tt.new, 1)
			}
			if _, err := ParseArchConfig([]byte(raw), tt.name); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error=%v want substring %q", err, tt.want)
			}
		})
	}
}

func TestOmittedLRScheduleKeepsCosineDefault(t *testing.T) {
	raw := strings.Replace(newBobClassificationConfig,
		`,
		"lr_schedule":"newbob",
		"newbob":{"annealing_factor":0.9,"improvement_threshold":0.0025,"patient":0,"metric":"val_error_rate"},
		"val_every_steps":5,
		"val_examples":0`, "", 1)
	cfg, err := ParseArchConfig([]byte(raw), "cosine-default")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Training.EffectiveLRSchedule() != LRScheduleCosine || cfg.Training.ConfiguredMetricValidation() {
		t.Fatalf("schedule=%q configured_validation=%t", cfg.Training.EffectiveLRSchedule(), cfg.Training.ConfiguredMetricValidation())
	}
}
