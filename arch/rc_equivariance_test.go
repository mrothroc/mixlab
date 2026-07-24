package arch

import (
	"encoding/json"
	"strings"
	"testing"
)

func rcEquivarianceMLMConfig() ArchConfig {
	return ArchConfig{
		Name:          "rc_mlm",
		ModelDim:      8,
		VocabSize:     9,
		SeqLen:        6,
		TieEmbeddings: true,
		RCEquivariant: true,
		Blocks: []BlockSpec{
			{Type: "plain", Heads: 2, AttentionMask: AttentionMaskBidirectional},
			{Type: "swiglu"},
		},
		Training: TrainingSpec{
			Objective:      ObjectiveMLM,
			MLMMaskTokenID: 3,
			BatchTokens:    12,
			Steps:          2,
			LR:             1e-3,
		},
	}
}

func parseRCEquivarianceConfig(t *testing.T, cfg ArchConfig) (*ArchConfig, error) {
	t.Helper()
	raw, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return ParseArchConfig(raw, cfg.Name)
}

func TestRCEquivarianceConfigAndDisabledWeightParity(t *testing.T) {
	cfg, err := parseRCEquivarianceConfig(t, rcEquivarianceMLMConfig())
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.RCEquivarianceEnabled() {
		t.Fatal("rc_equivariant was not preserved")
	}
	rcWeights, err := CollectWeightShapesFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	baseline := *cfg
	baseline.RCEquivariant = false
	baseWeights, err := CollectWeightShapesFromConfig(&baseline)
	if err != nil {
		t.Fatal(err)
	}
	if len(rcWeights) != len(baseWeights) {
		t.Fatalf("RC weights=%d baseline=%d", len(rcWeights), len(baseWeights))
	}
	for i := range rcWeights {
		if rcWeights[i].Name != baseWeights[i].Name || !equalShape(rcWeights[i].Shape, baseWeights[i].Shape) {
			t.Fatalf("weight %d RC=%+v baseline=%+v", i, rcWeights[i], baseWeights[i])
		}
	}
	rcFLOPs := EstimateFLOPs(cfg)
	baseFLOPs := EstimateFLOPs(&baseline)
	if rcFLOPs.ForwardFLOPs != 2*baseFLOPs.ForwardFLOPs {
		t.Fatalf("RC forward FLOPs=%d baseline=%d, want exactly 2x", rcFLOPs.ForwardFLOPs, baseFLOPs.ForwardFLOPs)
	}
	if rcFLOPs.ParamCount != baseFLOPs.ParamCount {
		t.Fatalf("RC params=%d baseline=%d", rcFLOPs.ParamCount, baseFLOPs.ParamCount)
	}
}

func TestRCEquivarianceValidation(t *testing.T) {
	tests := []struct {
		name string
		edit func(*ArchConfig)
		want string
	}{
		{"causal", func(c *ArchConfig) { c.Training.Objective = ObjectiveCausal }, "supports training.objective"},
		{"augmentation", func(c *ArchConfig) { c.Training.ReverseComplementProb = 0.5 }, "reverse_complement_prob"},
		{"causal attention", func(c *ArchConfig) { c.Blocks[0].AttentionMask = AttentionMaskCausal }, "bidirectional or none"},
		{"dropout", func(c *ArchConfig) { c.Dropout = 0.1 }, "requires dropout"},
		{"feature embeddings", func(c *ArchConfig) { c.BigramVocabSize = 16 }, "embedding features"},
		{"learned positions", func(c *ArchConfig) { c.PositionalEmbedding = PositionalEmbeddingLearnedAbsolute }, "learned_absolute"},
		{"unsupported mixer", func(c *ArchConfig) { c.Blocks[0] = BlockSpec{Type: "hgrn2", Heads: 2} }, "does not support"},
		{"no mixer", func(c *ArchConfig) { c.Blocks = []BlockSpec{{Type: "swiglu"}} }, "at least one"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := rcEquivarianceMLMConfig()
			tt.edit(&cfg)
			_, err := parseRCEquivarianceConfig(t, cfg)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error=%v, want substring %q", err, tt.want)
			}
		})
	}
}

func TestRCEquivarianceClassificationRequiresMeanPooling(t *testing.T) {
	cfg := rcEquivarianceMLMConfig()
	cfg.Training.Objective = ObjectiveClassification
	cfg.Training.Classification = &ClassificationSpec{NumLabels: 2, Pooling: ClassificationPoolingMean}
	cfg.Blocks = []BlockSpec{{Type: "gated_deltanet", Heads: 2, DK: 4}}
	parsed, err := parseRCEquivarianceConfig(t, cfg)
	if err != nil {
		t.Fatal(err)
	}
	prog, err := BuildIRProgramFromConfig(parsed)
	if err != nil {
		t.Fatal(err)
	}
	if !programHasOutput(prog, "classification_logits") || !programHasOutput(prog, "rc_equivariant_hidden") {
		t.Fatalf("classification outputs=%+v", prog.Outputs)
	}

	cfg.Training.Classification.Pooling = ClassificationPoolingLast
	if _, err := parseRCEquivarianceConfig(t, cfg); err == nil || !strings.Contains(err.Error(), "pooling=\"mean\"") {
		t.Fatalf("last-pooling error=%v", err)
	}
}

func TestRCEquivarianceIREmitsSharedPairedBackboneAndLogicalOutputs(t *testing.T) {
	cfg, err := parseRCEquivarianceConfig(t, rcEquivarianceMLMConfig())
	if err != nil {
		t.Fatal(err)
	}
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"tokens", "rc_tokens", "rc_alignment_positions", "rc_complement_ids", "targets", "loss_mask"} {
		if !programHasInput(prog, name) {
			t.Fatalf("missing input %q", name)
		}
	}
	for _, name := range []string{"logits", "x_hidden", "rc_equivariant_hidden"} {
		if !programHasOutput(prog, name) {
			t.Fatalf("missing output %q", name)
		}
	}
	wantShapes := map[string][]int{
		"logits":                {12, 9},
		"x_hidden":              {2, 6, 8},
		"rc_equivariant_hidden": {2, 6, 16},
	}
	for _, output := range prog.Outputs {
		if want, ok := wantShapes[output.Name]; ok && !equalShape(output.Shape, want) {
			t.Fatalf("output %s shape=%v want=%v", output.Name, output.Shape, want)
		}
	}
	embedCount := 0
	hasPairedConcat := false
	hasVocabComplementGather := false
	for _, op := range prog.Ops {
		if op.Code == OpEmbed && len(op.Inputs) > 0 && op.Inputs[0] == weightName(0) {
			embedCount++
		}
		if op.Code == OpConcat && len(op.Outputs) == 1 && op.Outputs[0] == "x" {
			hasPairedConcat = true
		}
		if op.Code == OpEmbed && len(op.Inputs) == 2 && op.Inputs[1] == "rc_complement_ids" {
			hasVocabComplementGather = true
		}
	}
	if embedCount != 2 {
		t.Fatalf("shared token embedding lookups=%d, want 2", embedCount)
	}
	if !hasPairedConcat || !hasVocabComplementGather {
		t.Fatalf("paired concat=%t vocabulary complement gather=%t", hasPairedConcat, hasVocabComplementGather)
	}
}

func TestRCEquivarianceBERTMLMHeadSharesOneWeightSet(t *testing.T) {
	cfg := rcEquivarianceMLMConfig()
	cfg.MLMHead = MLMHeadBERT
	parsed, err := parseRCEquivarianceConfig(t, cfg)
	if err != nil {
		t.Fatal(err)
	}
	shapes, err := CollectWeightShapesFromConfig(parsed)
	if err != nil {
		t.Fatal(err)
	}
	prog, err := BuildIRProgramFromConfig(parsed)
	if err != nil {
		t.Fatal(err)
	}
	if prog.NumWeights != len(shapes) {
		t.Fatalf("program weights=%d shapes=%d", prog.NumWeights, len(shapes))
	}
	denseRefs := 0
	for _, op := range prog.Ops {
		if op.Code == OpMatMul && len(op.Inputs) == 2 && op.Inputs[1] == weightName(len(shapes)-3) {
			denseRefs++
		}
	}
	if denseRefs != 2 {
		t.Fatalf("shared BERT MLM dense weight references=%d, want 2", denseRefs)
	}
}

func TestRCEquivarianceRejectsGeneration(t *testing.T) {
	cfg, err := parseRCEquivarianceConfig(t, rcEquivarianceMLMConfig())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := BuildGenerationIRProgramFromConfig(cfg); err == nil || !strings.Contains(err.Error(), "rc_equivariant") {
		t.Fatalf("generation error=%v", err)
	}
}
