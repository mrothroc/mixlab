package arch

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func TestCLSPositionDefaultsAndLayout(t *testing.T) {
	c := classificationTestConfig(BlockSpec{Type: "plain", Heads: 2, AttentionMask: "bidirectional"})
	c.Training.Classification.Pooling = "cls"
	base := parseClassificationTestConfig(t, c)
	want, err := BuildEvalIRProgramFromConfig(base)
	if err != nil {
		t.Fatal(err)
	}
	weights, err := CollectWeightShapesFromConfig(base)
	if err != nil {
		t.Fatal(err)
	}
	for _, position := range []string{"head", "middle", "tail"} {
		c.Training.Classification.CLSPosition = position
		cfg := parseClassificationTestConfig(t, c)
		p, err := BuildEvalIRProgramFromConfig(cfg)
		if err != nil {
			t.Fatal(err)
		}
		got, err := CollectWeightShapesFromConfig(cfg)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, weights) {
			t.Fatal("CLS position changed weight layout")
		}
		if position == "head" && !reflect.DeepEqual(p, want) {
			t.Fatal("explicit head changed default graph")
		}
		if position != "head" && (!programDeclaresInput(p, clsInsertionInput) || !programDeclaresInput(p, "classification_positions")) {
			t.Fatal("missing placement/readout input")
		}
	}
}

func TestCLSRecurrentValidationAndMask(t *testing.T) {
	for _, block := range []BlockSpec{
		{Type: "s4d", StateSize: 4, Bidirectional: true},
		{Type: "gated_deltanet", Heads: 2, DK: 4, DV: 4, Bidirectional: true},
		{Type: "mamba3-canonical", InnerDim: 16, StateSize: 4, NGroups: 2, DTRank: 2, Bidirectional: true},
	} {
		t.Run(block.Type, func(t *testing.T) {
			c := classificationTestConfig(block)
			c.Training.Classification.Pooling = "cls"
			for _, position := range []string{"head", "middle", "tail"} {
				c.Training.Classification.CLSPosition = position
				cfg := parseClassificationTestConfig(t, c)
				p, err := BuildEvalIRProgramFromConfig(cfg)
				if err != nil {
					t.Fatal(err)
				}
				found := false
				for _, op := range p.Ops {
					for _, out := range op.Outputs {
						if out == "cls_valid" {
							found = true
						}
					}
				}
				if !found {
					t.Fatal("CLS not added to recurrent valid mask")
				}
			}
			c.Blocks[0].Bidirectional = false
			raw, _ := json.Marshal(c)
			if _, err := ParseArchConfig(raw, "unidirectional"); err == nil || !strings.Contains(err.Error(), "bidirectional") {
				t.Fatalf("error=%v", err)
			}
		})
	}
}

func TestCLSPositionValidation(t *testing.T) {
	for _, tc := range []struct{ pool, position string }{{"cls", "invalid"}, {"mean", "tail"}, {"last", "head"}} {
		c := classificationTestConfig(BlockSpec{Type: "plain", Heads: 2, AttentionMask: "bidirectional"})
		c.Training.Classification.Pooling = tc.pool
		c.Training.Classification.CLSPosition = tc.position
		raw, _ := json.Marshal(c)
		if _, err := ParseArchConfig(raw, "bad_position"); err == nil || !strings.Contains(err.Error(), "cls_position") {
			t.Fatalf("error=%v", err)
		}
	}
	c := classificationTestConfig(BlockSpec{Type: "s4d", StateSize: 4, Bidirectional: true})
	c.VocabSize = 0
	c.TieEmbeddings = false
	c.InputAdapter = &InputAdapterSpec{Kind: "linear_frames", FeatureDim: 1, Norm: "none"}
	c.Training.Classification.Pooling = "cls"
	c.Training.Classification.CLSPosition = "middle"
	c.Training.LengthBuckets = []int{3, 6}
	raw, _ := json.Marshal(c)
	if _, err := ParseArchConfig(raw, "middle_buckets"); err == nil || !strings.Contains(err.Error(), "cls_position=middle") {
		t.Fatalf("middle bucketing error=%v", err)
	}
}
