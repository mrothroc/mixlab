package arch

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestCLSPoolingWeightsAndIR(t *testing.T) {
	for _, adapter := range []string{"token", "linear_frames", "discrete_codebooks"} {
		for _, pos := range []string{"none", "rope", "learned_absolute"} {
			t.Run(adapter+"/"+pos, func(t *testing.T) {
				c := classificationTestConfig(BlockSpec{Type: "plain", Heads: 2, AttentionMask: "bidirectional"})
				c.PositionalEmbedding = pos
				if adapter != "token" {
					c.VocabSize = 0
					c.TieEmbeddings = false
					c.InputAdapter = &InputAdapterSpec{Kind: adapter, FeatureDim: 2, NumCodebooks: 2, CodebookVocabSize: 8, Fusion: "mean", Norm: "none"}
					if adapter == "linear_frames" {
						c.InputAdapter.NumCodebooks = 0
						c.InputAdapter.CodebookVocabSize = 0
						c.InputAdapter.Fusion = ""
					} else {
						c.InputAdapter.FeatureDim = 0
					}
				}
				c.Training.Classification.Pooling = "mean"
				base := parseClassificationTestConfig(t, c)
				before, _, err := ParameterCountsFromConfig(base)
				if err != nil {
					t.Fatal(err)
				}
				c.Training.Classification.Pooling = "cls"
				cfg := parseClassificationTestConfig(t, c)
				after, _, err := ParameterCountsFromConfig(cfg)
				if err != nil {
					t.Fatal(err)
				}
				want := int64(c.ModelDim)
				if pos == "learned_absolute" {
					want *= 2
				}
				if after-before != want {
					t.Fatalf("parameter delta %d want %d", after-before, want)
				}
				metas, err := CollectWeightShapesFromConfig(cfg)
				if err != nil {
					t.Fatal(err)
				}
				if metas[len(metas)-3].Name != "cls_token" || !reflect.DeepEqual(metas[len(metas)-3].Shape, []int{1, c.ModelDim}) {
					t.Fatal("CLS layout")
				}
				p, err := BuildEvalIRProgramFromConfig(cfg)
				if err != nil {
					t.Fatal(err)
				}
				if p.NumWeights != len(metas) {
					t.Fatalf("weights %d != %d", p.NumWeights, len(metas))
				}
				for _, in := range p.Inputs {
					if in.Name == "segment_ids" && !reflect.DeepEqual(in.Shape, []int{2, 6}) {
						t.Fatalf("raw mask changed: %v", in.Shape)
					}
				}
				for _, out := range p.Outputs {
					if out.Name == "x_hidden" && !reflect.DeepEqual(out.Shape, []int{2, 7, 16}) {
						t.Fatalf("hidden %v", out.Shape)
					}
				}
				seenPrepend, seenMask := false, false
				for _, op := range p.Ops {
					for _, out := range op.Outputs {
						if out == "cls_input" {
							seenPrepend = true
						}
						if out == "cls_segment" {
							seenMask = true
						}
					}
				}
				if !seenPrepend || !seenMask {
					t.Fatal("missing prepend/mask")
				}
			})
		}
	}
}

func TestCLSPoolingValidation(t *testing.T) {
	for _, tc := range []struct {
		name string
		edit func(*ArchConfig)
	}{
		{"empty", func(c *ArchConfig) { c.Blocks = nil }},
		{"ffn_only", func(c *ArchConfig) { c.Blocks = []BlockSpec{{Type: "geglu"}} }},
		{"causal", func(c *ArchConfig) { c.Blocks[0].AttentionMask = "causal" }},
		{"skipped_attention", func(c *ArchConfig) { c.Blocks[0].SkipAttention = true }},
		{"recurrent", func(c *ArchConfig) { c.Blocks = []BlockSpec{{Type: "s4d", Bidirectional: true}} }},
		{"capacity", func(c *ArchConfig) { c.MaxPositions = c.SeqLen }},
		{"features", func(c *ArchConfig) { c.BigramVocabSize = 8 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := classificationTestConfig(BlockSpec{Type: "plain", Heads: 2, AttentionMask: "bidirectional"})
			c.Training.Classification.Pooling = "cls"
			tc.edit(&c)
			raw, _ := json.Marshal(c)
			if _, err := ParseArchConfig(raw, tc.name); err == nil {
				t.Fatal("expected rejection")
			}
		})
	}
}

func TestCLSPoolingSparePositionCapacity(t *testing.T) {
	c := classificationTestConfig(BlockSpec{Type: "plain", Heads: 2, AttentionMask: "bidirectional"})
	c.PositionalEmbedding = "learned_absolute"
	c.MaxPositions = 20
	c.Training.Classification.Pooling = "mean"
	base := parseClassificationTestConfig(t, c)
	before, _, err := ParameterCountsFromConfig(base)
	if err != nil {
		t.Fatal(err)
	}
	c.Training.Classification.Pooling = "cls"
	cfg := parseClassificationTestConfig(t, c)
	after, _, err := ParameterCountsFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if after-before != int64(c.ModelDim) {
		t.Fatalf("spare capacity parameter delta=%d", after-before)
	}
}
