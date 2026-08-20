package arch

import (
	"encoding/json"
	"strings"
	"testing"
)

func lengthBucketConfig() ArchConfig {
	return ArchConfig{
		Name: "bucketed_codebooks", ModelDim: 8, SeqLen: 8,
		PositionalEmbedding: PositionalEmbeddingNone,
		InputAdapter: &InputAdapterSpec{
			Kind: InputAdapterDiscreteCodebooks, NumCodebooks: 2, CodebookVocabSize: 16,
			Fusion: InputAdapterFusionMean, Norm: InputAdapterNormNone,
		},
		Blocks: []BlockSpec{{Type: "swiglu"}},
		Training: TrainingSpec{
			Objective: ObjectiveClassification, BatchTokens: 10, Steps: 2, LR: 1e-3,
			LengthBuckets:  []int{2, 4, 8},
			Classification: &ClassificationSpec{NumLabels: 3, Pooling: ClassificationPoolingMean},
		},
	}
}

func parseLengthBucketConfig(t *testing.T, cfg ArchConfig) (*ArchConfig, error) {
	t.Helper()
	raw, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return ParseArchConfig(raw, cfg.Name)
}

func TestLengthBucketsConfigShapesAndClassificationIR(t *testing.T) {
	cfg, err := parseLengthBucketConfig(t, lengthBucketConfig())
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct{ width, rows, tokens int }{{2, 5, 10}, {4, 2, 8}, {8, 1, 8}} {
		rows, tokens := cfg.Training.LengthBucketBatchShape(tt.width)
		if rows != tt.rows || tokens != tt.tokens {
			t.Fatalf("bucket %d shape=%dx%d want rows=%d tokens=%d", tt.width, rows, tokens, tt.rows, tt.tokens)
		}
	}
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if !programHasInput(prog, "classification_example_mask") || countOps(prog, OpMaskedCrossEntropy) != 1 || countOps(prog, OpCrossEntropy) != 0 {
		t.Fatalf("bucketed classification inputs=%v ops=%v", prog.Inputs, prog.Ops)
	}
	without := *cfg
	without.Training = cfg.Training
	without.Training.LengthBuckets = nil
	plain, err := BuildIRProgramFromConfig(&without)
	if err != nil {
		t.Fatal(err)
	}
	if prog.NumWeights != plain.NumWeights {
		t.Fatalf("bucketed weights=%d non-bucketed=%d", prog.NumWeights, plain.NumWeights)
	}
	if programHasInput(plain, "classification_example_mask") || countOps(plain, OpCrossEntropy) != 1 {
		t.Fatalf("non-bucketed classification behavior changed")
	}
}

func TestSingleFullWidthLengthBucketUsesLegacyIR(t *testing.T) {
	cfg := lengthBucketConfig()
	cfg.Training.BatchTokens = 16
	cfg.Training.LengthBuckets = []int{cfg.SeqLen}
	parsed, err := parseLengthBucketConfig(t, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if parsed.Training.LengthBucketsChangeShape(parsed.SeqLen) {
		t.Fatal("single full-width bucket should be a fixed-shape no-op")
	}
	prog, err := BuildIRProgramFromConfig(parsed)
	if err != nil {
		t.Fatal(err)
	}
	if programHasInput(prog, "classification_example_mask") || countOps(prog, OpCrossEntropy) != 1 || countOps(prog, OpMaskedCrossEntropy) != 0 {
		t.Fatalf("single full-width bucket did not preserve legacy IR")
	}
}

func TestLengthBucketsValidation(t *testing.T) {
	tests := []struct {
		name string
		edit func(*ArchConfig)
		want string
	}{
		{"not increasing", func(c *ArchConfig) { c.Training.LengthBuckets = []int{4, 4} }, "strictly greater"},
		{"over seq len", func(c *ArchConfig) { c.Training.LengthBuckets = []int{4, 9} }, "exceeds seq_len"},
		{"over token ceiling", func(c *ArchConfig) { c.Training.BatchTokens = 7 }, "must be >= largest"},
		{"non classification", func(c *ArchConfig) { c.Training.Objective = ObjectiveCausal; c.Training.Classification = nil }, "requires training.objective"},
		{"batchnorm", func(c *ArchConfig) { c.NormType = NormTypeBatchNorm }, "does not support norm_type"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := lengthBucketConfig()
			tt.edit(&cfg)
			_, err := parseLengthBucketConfig(t, cfg)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error=%v want substring %q", err, tt.want)
			}
		})
	}
	cfg := lengthBucketConfig()
	cfg.Blocks = []BlockSpec{{Type: "plain", Heads: 2, AttentionMask: AttentionMaskBidirectional}}
	if _, err := parseLengthBucketConfig(t, cfg); err != nil {
		t.Fatalf("bidirectional classification should use padding-derived segment IDs: %v", err)
	}
}
