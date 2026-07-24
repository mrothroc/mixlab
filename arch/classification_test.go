package arch

import (
	"encoding/json"
	"strings"
	"testing"
)

func classificationTestConfig(block BlockSpec) ArchConfig {
	return ArchConfig{
		Name: "classification", ModelDim: 16, VocabSize: 32, SeqLen: 6,
		TieEmbeddings: true,
		Blocks:        []BlockSpec{block},
		Training: TrainingSpec{
			Objective: ObjectiveClassification, BatchTokens: 12, Steps: 2, LR: 1e-3,
			Classification: &ClassificationSpec{NumLabels: 3},
		},
	}
}

func parseClassificationTestConfig(t *testing.T, cfg ArchConfig) *ArchConfig {
	t.Helper()
	raw, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseArchConfig(raw, cfg.Name)
	if err != nil {
		t.Fatal(err)
	}
	return parsed
}

func TestClassificationConfigDefaultsAndValidation(t *testing.T) {
	causal := parseClassificationTestConfig(t, classificationTestConfig(BlockSpec{Type: "plain", Heads: 2}))
	if got := causal.EffectiveClassificationPooling(); got != ClassificationPoolingLast {
		t.Fatalf("causal pooling=%q, want last", got)
	}
	bidirectionalCfg := classificationTestConfig(BlockSpec{Type: "plain", Heads: 2, AttentionMask: AttentionMaskBidirectional})
	bidirectionalCfg.HiddenDropout = 0.2
	bidirectional := parseClassificationTestConfig(t, bidirectionalCfg)
	if got := bidirectional.EffectiveClassificationPooling(); got != ClassificationPoolingMean {
		t.Fatalf("bidirectional pooling=%q, want mean", got)
	}
	if got := bidirectional.EffectiveClassifierDropout(); got != 0.2 {
		t.Fatalf("classifier dropout=%g, want hidden dropout 0.2", got)
	}

	tests := []struct {
		name string
		edit func(*ArchConfig)
		want string
	}{
		{"missing spec", func(c *ArchConfig) { c.Training.Classification = nil }, "requires training.classification"},
		{"one label", func(c *ArchConfig) { c.Training.Classification.NumLabels = 1 }, "must be >= 2"},
		{"bad pooling", func(c *ArchConfig) { c.Training.Classification.Pooling = "cls" }, "pooling"},
		{"spec on causal", func(c *ArchConfig) { c.Training.Objective = ObjectiveCausal }, "classification settings require"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := classificationTestConfig(BlockSpec{Type: "plain", Heads: 2})
			tt.edit(&cfg)
			raw, _ := json.Marshal(cfg)
			if _, err := ParseArchConfig(raw, tt.name); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error=%v, want substring %q", err, tt.want)
			}
		})
	}
}

func TestClassificationWeightsAppendWithoutChangingLMPrefix(t *testing.T) {
	cfg := parseClassificationTestConfig(t, classificationTestConfig(BlockSpec{Type: "plain", Heads: 2}))
	classificationWeights, err := CollectWeightShapesFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	base := *cfg
	base.Training.Objective = ObjectiveCausal
	base.Training.Classification = nil
	baseWeights, err := CollectWeightShapesFromConfig(&base)
	if err != nil {
		t.Fatal(err)
	}
	if len(classificationWeights) != len(baseWeights)+2 {
		t.Fatalf("classification weights=%d base=%d", len(classificationWeights), len(baseWeights))
	}
	for i := range baseWeights {
		if classificationWeights[i].Name != baseWeights[i].Name {
			t.Fatalf("weight %d name=%q, base=%q", i, classificationWeights[i].Name, baseWeights[i].Name)
		}
	}
	proj := classificationWeights[len(baseWeights)]
	bias := classificationWeights[len(baseWeights)+1]
	if proj.Name != "head_classifier_proj" || !equalShape(proj.Shape, []int{16, 3}) {
		t.Fatalf("classifier projection=%+v", proj)
	}
	if bias.Name != "head_classifier_bias" || !equalShape(bias.Shape, []int{3}) || !bias.InitZero {
		t.Fatalf("classifier bias=%+v", bias)
	}
}

func TestClassificationAllowsReverseComplementAugmentationForRuntimeDNAValidation(t *testing.T) {
	cfg := classificationTestConfig(BlockSpec{Type: "plain", Heads: 2, AttentionMask: AttentionMaskBidirectional})
	cfg.Training.Classification.Pooling = ClassificationPoolingMean
	cfg.Training.ReverseComplementProb = 0.5
	_ = parseClassificationTestConfig(t, cfg)
}

func TestClassificationIRUsesPaddingAwarePoolingAndNoLMProjection(t *testing.T) {
	for _, tt := range []struct {
		name    string
		block   BlockSpec
		pooling string
		input   string
	}{
		{"last", BlockSpec{Type: "plain", Heads: 2}, ClassificationPoolingLast, "classification_positions"},
		{"mean", BlockSpec{Type: "plain", Heads: 2, AttentionMask: AttentionMaskBidirectional}, ClassificationPoolingMean, "classification_valid_mask"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			cfg := parseClassificationTestConfig(t, classificationTestConfig(tt.block))
			prog, err := BuildIRProgramFromConfig(cfg)
			if err != nil {
				t.Fatal(err)
			}
			for _, name := range []string{"classification_labels", tt.input, "segment_ids"} {
				if !programHasInput(prog, name) {
					t.Fatalf("classification program missing input %q: %+v", name, prog.Inputs)
				}
			}
			if programHasInput(prog, "targets") {
				t.Fatal("classification graph retains unused LM targets")
			}
			if !programHasOutput(prog, "classification_logits") || programHasOutput(prog, "logits") {
				t.Fatalf("classification outputs=%+v", prog.Outputs)
			}
			for _, op := range prog.Ops {
				for _, output := range op.Outputs {
					if output == "logits" {
						t.Fatal("classification graph retains LM vocabulary projection")
					}
				}
			}
		})
	}
}

func equalShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func programHasInput(prog *Program, name string) bool {
	for _, input := range prog.Inputs {
		if input.Name == name {
			return true
		}
	}
	return false
}

func programHasOutput(prog *Program, name string) bool {
	for _, output := range prog.Outputs {
		if output.Name == name {
			return true
		}
	}
	return false
}
